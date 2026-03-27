import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import os
import subprocess
from datetime import datetime
from collections import deque

# Import architecture
from train_system_model import SystemAutoencoder, WINDOW_SIZE, HIDDEN_DIM, LATENT_DIM

# Configuration
DISKS = ["sda", "sdb", "sdc", "sdd", "sde", "sdf", "sdg", "nvme0n1", "nvme1n1", "md0"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "system_model.pth")
SCALER_PATH = os.path.join(SCRIPT_DIR, "system_scaler_params.npy")
FEATURE_NAMES_PATH = os.path.join(SCRIPT_DIR, "feature_names.txt")

def get_disk_stats():
    stats = {}
    with open("/proc/diskstats", "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 14: continue
            name = parts[2]
            if name in DISKS:
                stats[name] = {
                    "reads": int(parts[3]),
                    "read_bytes": int(parts[5]) * 512,
                    "time_reading": int(parts[6]),
                    "writes": int(parts[7]),
                    "write_bytes": int(parts[9]) * 512,
                    "time_writing": int(parts[10]),
                    "io_time": int(parts[12])
                }
    return stats

def get_mem_stats():
    mem = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("Slab:"):
                mem["slab_kb"] = int(line.split()[1])
            elif line.startswith("Percpu:"):
                mem["percpu_kb"] = int(line.split()[1])
    return mem

def run_monitor():
    if not os.path.exists(MODEL_PATH):
        print("Error: System model not found.")
        return

    scaler_mean, scaler_scale = np.load(SCALER_PATH)
    with open(FEATURE_NAMES_PATH, "r") as f:
        features = f.read().splitlines()
    
    input_dim = len(features)
    model = SystemAutoencoder(input_dim, HIDDEN_DIM, LATENT_DIM, WINDOW_SIZE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    window = deque(maxlen=WINDOW_SIZE)
    last_stats = get_disk_stats()
    
    print(f"--- System-Wide I/O Guard (Monitoring {len(DISKS)} devices) ---")
    print("Detecting cross-disk latency correlations and memory pressure...")
    print("-" * 60)

    try:
        while True:
            time.sleep(1.0)
            curr_stats = get_disk_stats()
            mem = get_mem_stats()
            if not curr_stats or not last_stats: continue

            row_data = {
                "slab_kb": mem.get("slab_kb", 0),
                "percpu_kb": mem.get("percpu_kb", 0)
            }
            
            for d in DISKS:
                if d in curr_stats and d in last_stats:
                    c, l = curr_stats[d], last_stats[d]
                    read_kb = (c["read_bytes"] - l["read_bytes"]) / 1024
                    write_kb = (c["write_bytes"] - l["write_bytes"]) / 1024
                    total_ops = (c["reads"] - l["reads"]) + (c["writes"] - l["writes"])
                    lat = (c["time_reading"] - l["time_reading"] + c["time_writing"] - l["time_writing"]) / total_ops if total_ops > 0 else 0
                    util = ((c["io_time"] - l["io_time"]) / 1000) * 100
                    row_data[f"{d}_read_kb_s"] = read_kb
                    row_data[f"{d}_write_kb_s"] = write_kb
                    row_data[f"{d}_latency_ms"] = lat
                    row_data[f"{d}_util_pct"] = util
                else:
                    row_data[f"{d}_read_kb_s"] = 0
                    row_data[f"{d}_write_kb_s"] = 0
                    row_data[f"{d}_latency_ms"] = 0
                    row_data[f"{d}_util_pct"] = 0

            # Map to feature order
            feat_vec = np.array([row_data[f] for f in features])
            scaled_vec = (feat_vec - scaler_mean) / scaler_scale
            window.append(scaled_vec)

            if len(window) == WINDOW_SIZE:
                input_tensor = torch.FloatTensor(np.array([list(window)]))
                with torch.no_grad():
                    reconstruction = model(input_tensor)
                    loss = torch.mean((reconstruction - input_tensor)**2).item()
                
                anomaly_score = min(100, loss * 100) # Heuristic
                status = "STABLE"
                if anomaly_score > 30: status = "UNSTABLE"
                if anomaly_score > 70: status = "CRITICAL"

                # Show score and top pressure points
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Score: {anomaly_score:5.2f} | Status: {status:8} | Slab: {row_data['slab_kb']/1024:7.1f}MB | md0 Util: {row_data['md0_util_pct']:5.1f}%")

            last_stats = curr_stats
            
    except KeyboardInterrupt:
        print("\nMonitor stopped.")

if __name__ == "__main__":
    run_monitor()
