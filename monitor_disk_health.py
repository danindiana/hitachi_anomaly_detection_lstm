import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import subprocess
import csv
from datetime import datetime
from collections import deque

# Import the model architecture from our training script
from train_disk_model import LSTMAutoencoder, WINDOW_SIZE, HIDDEN_DIM, LATENT_DIM

# Configuration
DISK = "sdb"
# Use absolute paths or paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "disk_model.pth")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler_params.npy")
FEATURES = ["read_kb_s", "write_kb_s", "avg_latency_ms", "io_utilization_pct", "temperature"]

def get_disk_stats():
    """Simplified stats fetch for the monitor."""
    try:
        with open("/proc/diskstats", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 2 and parts[2] == DISK:
                    return {
                        "reads": int(parts[3]),
                        "writes": int(parts[7]),
                        "read_bytes": int(parts[5]) * 512,
                        "write_bytes": int(parts[9]) * 512,
                        "time_reading": int(parts[6]),
                        "time_writing": int(parts[10]),
                        "io_time": int(parts[12])
                    }
    except Exception as e:
        print(f"Error reading diskstats: {e}")
    return None

def get_temp():
    """Quick temp fetch."""
    try:
        res = subprocess.check_output(["sudo", "smartctl", "-A", f"/dev/{DISK}"], stderr=subprocess.DEVNULL).decode()
        for line in res.splitlines():
            if "Temperature_Celsius" in line:
                return int(line.split()[9].split('(')[0])
    except:
        pass
    return 39

def run_monitor():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or Scaler not found. Please train the model first.")
        return

    # 1. Load Scaler and Model
    scaler_mean, scaler_scale = np.load(SCALER_PATH)
    input_dim = len(FEATURES)
    model = LSTMAutoencoder(input_dim, HIDDEN_DIM, LATENT_DIM, WINDOW_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # 2. Setup real-time window
    window = deque(maxlen=WINDOW_SIZE)
    last_stats = get_disk_stats()
    
    print(f"--- Real-time Health Monitor for {DISK} ---")
    print("Anomaly Score: 0 (Normal) -> Higher (Potential Issue)")
    print("-" * 50)
    print("Initializing data window (10 seconds)...")

    try:
        while True:
            time.sleep(1.0)
            curr_stats = get_disk_stats()
            if not curr_stats or not last_stats: 
                print("DEBUG: Failed to get stats")
                continue

            # Calculate metrics
            read_kb = (curr_stats["read_bytes"] - last_stats["read_bytes"]) / 1024
            write_kb = (curr_stats["write_bytes"] - last_stats["write_bytes"]) / 1024
            
            reads_diff = curr_stats["reads"] - last_stats["reads"]
            writes_diff = curr_stats["writes"] - last_stats["writes"]
            total_ops = reads_diff + writes_diff
            
            avg_lat = 0
            if total_ops > 0:
                total_time = (curr_stats["time_reading"] - last_stats["time_reading"]) + \
                             (curr_stats["time_writing"] - last_stats["time_writing"])
                avg_lat = total_time / total_ops
                
            util = ((curr_stats["io_time"] - last_stats["io_time"]) / 1000) * 100
            temp = get_temp()

            # Create feature vector
            feat_vec = np.array([read_kb, write_kb, avg_lat, util, temp])
            # Scale
            scaled_vec = (feat_vec - scaler_mean) / scaler_scale
            window.append(scaled_vec)
            
            # print(f"DEBUG: Window size: {len(window)}")

            # Inference
            if len(window) == WINDOW_SIZE:
                input_tensor = torch.FloatTensor(np.array([list(window)]))
                with torch.no_grad():
                    reconstruction = model(input_tensor)
                    loss = torch.mean((reconstruction - input_tensor)**2).item()
                
                # Normalize loss to a 0-100 scale (heuristic based on training loss)
                anomaly_score = min(100, loss * 1000) 
                
                status = "NORMAL"
                if anomaly_score > 50: status = "WARNING"
                if anomaly_score > 80: status = "CRITICAL"

                print(f"[{datetime.now().strftime('%H:%M:%S')}] Score: {anomaly_score:5.2f} | Status: {status:8} | Lat: {avg_lat:5.2f}ms | Util: {util:4.1f}%")

            last_stats = curr_stats
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    run_monitor()
