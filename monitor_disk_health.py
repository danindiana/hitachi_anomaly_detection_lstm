import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import subprocess
import csv
import syslog
from datetime import datetime
from collections import deque
from config import (
    DISK, DEV_PATH, MODEL_PATH, SCALER_PATH, THRESHOLDS_PATH, FEATURE_NAMES_PATH,
    FEATURES, WINDOW_SIZE, HIDDEN_DIM, LATENT_DIM, 
    POLL_INTERVAL, SMART_INTERVAL, REALLOCATED_BASELINE, PENDING_BASELINE
)

# Import the model architecture
from train_disk_model import LSTMAutoencoder

def get_disk_stats():
    """Simplified stats fetch for the monitor."""
    try:
        with open("/proc/diskstats", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 2 and parts[2] == DISK:
                    return {
                        "reads": int(parts[3]),
                        "read_bytes": int(parts[5]) * 512,
                        "writes": int(parts[7]),
                        "write_bytes": int(parts[9]) * 512,
                        "time_reading": int(parts[6]),
                        "time_writing": int(parts[10]),
                        "io_time": int(parts[12])
                    }
    except Exception as e:
        pass
    return None

def get_smart_metrics():
    """Polls smartctl for critical health indicators."""
    try:
        res = subprocess.check_output(["sudo", "smartctl", "-A", DEV_PATH], stderr=subprocess.STDOUT).decode()
        metrics = {}
        for line in res.splitlines():
            parts = line.split()
            if not parts or len(parts) < 10:
                continue
            
            attr_name = parts[1]
            raw_value = parts[9]
            
            if attr_name == "Reallocated_Sector_Ct":
                metrics["reallocated_sectors"] = int(raw_value)
            elif attr_name == "Current_Pending_Sector":
                metrics["pending_sectors"] = int(raw_value)
            elif attr_name == "Offline_Uncorrectable":
                metrics["offline_uncorrectable"] = int(raw_value)
            elif attr_name == "Temperature_Celsius":
                metrics["temperature"] = int(raw_value.split('(')[0].strip())
        return metrics
    except:
        return None

def run_monitor():
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, THRESHOLDS_PATH]):
        print("Error: Model, Scaler, or Thresholds not found. Please train the model first.")
        return

    # 1. Load Scaler, Model, and Thresholds
    scaler_params = np.load(SCALER_PATH)
    threshold_warn, threshold_crit = np.load(THRESHOLDS_PATH)
    
    if os.path.exists(FEATURE_NAMES_PATH):
        with open(FEATURE_NAMES_PATH, "r") as f:
            actual_features = f.read().splitlines()
    else:
        actual_features = FEATURES

    assert scaler_params.shape == (2, len(actual_features)), \
        f"Scaler mismatch: expected (2, {len(actual_features)}), got {scaler_params.shape}. Retrain."
    
    scaler_mean, scaler_scale = scaler_params
    
    input_dim = len(actual_features)
    model = LSTMAutoencoder(input_dim, HIDDEN_DIM, LATENT_DIM, WINDOW_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # 2. Setup real-time window with Startup Retry Cap
    window = deque(maxlen=WINDOW_SIZE)
    last_stats = get_disk_stats()
    last_smart = get_smart_metrics()
    
    retries = 0
    MAX_RETRIES = 15
    while last_smart is None:
        retries += 1
        if retries > MAX_RETRIES:
            msg = f"CRITICAL: Cannot get initial SMART data after {MAX_RETRIES} attempts. Check sudoers rule for smartctl."
            syslog.syslog(syslog.LOG_ERR, msg)
            raise RuntimeError(msg)
        print(f"Waiting for initial SMART data... (attempt {retries}/{MAX_RETRIES})")
        time.sleep(2)
        last_smart = get_smart_metrics()
        
    # Initial Baseline Check
    if last_smart["reallocated_sectors"] > REALLOCATED_BASELINE:
        syslog.syslog(syslog.LOG_CRIT, f"Disk {DISK} STARTUP ALERT: Current reallocated sectors ({last_smart['reallocated_sectors']}) already exceed baseline ({REALLOCATED_BASELINE})")
    
    if last_smart["pending_sectors"] > PENDING_BASELINE:
        syslog.syslog(syslog.LOG_WARNING, f"Disk {DISK} STARTUP ALERT: Current pending sectors ({last_smart['pending_sectors']}) exceed baseline ({PENDING_BASELINE})")

    # Initialize Edge-Trigger States
    prev_reallocated = last_smart["reallocated_sectors"]
    prev_pending = last_smart["pending_sectors"]
    
    last_crit_log_time = 0
    CRIT_COOLDOWN = 300
        
    last_smart_time = time.time()
    last_poll_time = time.time()
    
    print(f"--- Real-time Health Monitor for {DISK} ---")
    print(f"Thresholds: WARN={threshold_warn:.6f}, CRIT={threshold_crit:.6f}")
    print("-" * 60)
    print("Initializing data window...")

    try:
        while True:
            time.sleep(max(0, POLL_INTERVAL - (time.time() - last_poll_time)))
            curr_time = time.time()
            dt = curr_time - last_poll_time
            last_poll_time = curr_time
            
            curr_stats = get_disk_stats()
            if not curr_stats or not last_stats: 
                continue

            if curr_time - last_smart_time > SMART_INTERVAL:
                curr_smart = get_smart_metrics()
                if curr_smart:
                    if curr_smart["reallocated_sectors"] > prev_reallocated:
                        syslog.syslog(syslog.LOG_CRIT, 
                            f"Disk {DISK} REALLOCATED sectors GREW: {prev_reallocated} -> {curr_smart['reallocated_sectors']}")
                        prev_reallocated = curr_smart["reallocated_sectors"]
                    
                    if curr_smart["pending_sectors"] > prev_pending:
                        syslog.syslog(syslog.LOG_WARNING, 
                            f"Disk {DISK} PENDING sectors GREW: {prev_pending} -> {curr_smart['pending_sectors']}")
                        prev_pending = curr_smart["pending_sectors"]
                        
                    last_smart = curr_smart
                    last_smart_time = curr_time

            read_kb_s = ((curr_stats["read_bytes"] - last_stats["read_bytes"]) / 1024) / dt
            write_kb_s = ((curr_stats["write_bytes"] - last_stats["write_bytes"]) / 1024) / dt
            
            reads_diff = curr_stats["reads"] - last_stats["reads"]
            writes_diff = curr_stats["writes"] - last_stats["writes"]
            total_ops = reads_diff + writes_diff
            
            avg_lat = 0
            if total_ops > 0:
                total_time = (curr_stats["time_reading"] - last_stats["time_reading"]) + \
                             (curr_stats["time_writing"] - last_stats["time_writing"])
                avg_lat = total_time / total_ops
                
            util = ((curr_stats["io_time"] - last_stats["io_time"]) / (dt * 1000)) * 100

            feat_dict = {
                "read_kb_s": read_kb_s,
                "write_kb_s": write_kb_s,
                "avg_latency_ms": avg_lat,
                "io_utilization_pct": util,
                "temperature": last_smart["temperature"],
                "reallocated_sectors": last_smart["reallocated_sectors"],
                "pending_sectors": last_smart["pending_sectors"],
                "offline_uncorrectable": last_smart.get("offline_uncorrectable", 0)
            }
            
            missing = [f for f in actual_features if f not in feat_dict]
            if missing:
                raise RuntimeError(f"Monitor feat_dict missing keys: {missing}.")

            feat_vec = np.array([feat_dict[f] for f in actual_features])
            scaled_vec = (feat_vec - scaler_mean) / scaler_scale
            window.append(scaled_vec)
            
            if len(window) == WINDOW_SIZE:
                input_tensor = torch.FloatTensor(np.array([list(window)]))
                with torch.no_grad():
                    reconstruction = model(input_tensor)
                    loss = torch.mean((reconstruction - input_tensor)**2).item()
                
                status = "NORMAL"
                if loss > threshold_warn: status = "WARNING"
                if loss > threshold_crit: 
                    status = "CRITICAL"
                    if time.time() - last_crit_log_time > CRIT_COOLDOWN:
                        syslog.syslog(syslog.LOG_CRIT, f"Disk {DISK} CRITICAL Health Anomaly: score {loss:.6f}")
                        last_crit_log_time = time.time()

                ts = datetime.now().strftime('%H:%M:%S')
                print(f"[{ts}] Loss: {loss:.6f} | Status: {status:8} | Lat: {avg_lat:5.2f}ms | Util: {util:5.1f}% | Temp: {last_smart['temperature']}C")

            last_stats = curr_stats
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    run_monitor()
