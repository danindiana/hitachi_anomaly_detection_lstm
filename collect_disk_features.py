import os
import time
import subprocess
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from config import DISK, DEV_PATH, POLL_INTERVAL, SMART_INTERVAL, DATA_FILE

MAX_ROWS = 100000  # Approx 27 hours of data at 1Hz

def trim_csv(file_path, max_rows):
    """Keep only the last max_rows in the CSV to prevent unbound growth."""
    try:
        if not os.path.exists(file_path):
            return
        
        # Read the file
        df = pd.read_csv(file_path)
        if len(df) > max_rows * 1.2:  # Only trim if it's 20% over limit to avoid frequent I/O
            print(f"\nTrimming {file_path} (current rows: {len(df)})")
            df = df.tail(max_rows)
            df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"\nError trimming CSV: {e}")

def get_disk_stats():
    """Reads /proc/diskstats for the specific disk."""
    try:
        with open("/proc/diskstats", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) > 2 and parts[2] == DISK:
                    return {
                        "reads": int(parts[3]),
                        "sectors_read": int(parts[5]),
                        "time_reading": int(parts[6]),
                        "writes": int(parts[7]),
                        "sectors_written": int(parts[9]),
                        "time_writing": int(parts[10]),
                        "io_time": int(parts[12])
                    }
    except Exception as e:
        print(f"Error reading diskstats: {e}")
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
            elif attr_name == "Power_On_Hours":
                metrics["power_on_hours"] = int(raw_value)
        return metrics
    except Exception as e:
        return None

def main():
    print(f"Starting feature collection for {DEV_PATH}...")
    print(f"Outputting to {DATA_FILE}")
    
    headers = [
        "timestamp", "unix_time", 
        "read_kb_s", "write_kb_s", "avg_latency_ms", "io_utilization_pct",
        "reallocated_sectors", "pending_sectors", "offline_uncorrectable", 
        "temperature", "power_on_hours"
    ]
    
    file_exists = os.path.isfile(DATA_FILE)
    
    # Trim on startup
    if file_exists:
        trim_csv(DATA_FILE, MAX_ROWS)
    
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
            
        last_stats = get_disk_stats()
        last_smart = get_smart_metrics()
        while last_smart is None:
            print("Waiting for initial SMART data...")
            time.sleep(2)
            last_smart = get_smart_metrics()
            
        last_smart_time = time.time()
        last_poll_time = time.time()
        last_trim_time = time.time()
        
        try:
            while True:
                time.sleep(max(0, POLL_INTERVAL - (time.time() - last_poll_time)))
                
                curr_time = time.time()
                dt = curr_time - last_poll_time
                last_poll_time = curr_time
                
                curr_stats = get_disk_stats()
                if not curr_stats or not last_stats:
                    continue
                
                reads_diff = curr_stats["reads"] - last_stats["reads"]
                writes_diff = curr_stats["writes"] - last_stats["writes"]
                sectors_read = curr_stats["sectors_read"] - last_stats["sectors_read"]
                sectors_written = curr_stats["sectors_written"] - last_stats["sectors_written"]
                time_io = (curr_stats["io_time"] - last_stats["io_time"])
                
                read_kb_s = (sectors_read * 512) / 1024 / dt
                write_kb_s = (sectors_written * 512) / 1024 / dt
                
                total_ops = reads_diff + writes_diff
                if total_ops > 0:
                    total_time = (curr_stats["time_reading"] - last_stats["time_reading"]) + \
                                 (curr_stats["time_writing"] - last_stats["time_writing"])
                    avg_latency = total_time / total_ops
                else:
                    avg_latency = 0
                    
                utilization = (time_io / (dt * 1000)) * 100
                
                if curr_time - last_smart_time > SMART_INTERVAL:
                    curr_smart = get_smart_metrics()
                    if curr_smart:
                        last_smart = curr_smart
                        last_smart_time = curr_time
                
                # Periodic trim (every 1 hour)
                if curr_time - last_trim_time > 3600:
                    f.flush()
                    trim_csv(DATA_FILE, MAX_ROWS)
                    last_trim_time = curr_time
                
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "unix_time": curr_time,
                    "read_kb_s": round(read_kb_s, 2),
                    "write_kb_s": round(write_kb_s, 2),
                    "avg_latency_ms": round(avg_latency, 2),
                    "io_utilization_pct": round(utilization, 2),
                    "reallocated_sectors": last_smart["reallocated_sectors"],
                    "pending_sectors": last_smart["pending_sectors"],
                    "offline_uncorrectable": last_smart.get("offline_uncorrectable", 0),
                    "temperature": last_smart["temperature"],
                    "power_on_hours": last_smart["power_on_hours"]
                }
                writer.writerow(row)
                f.flush()
                
                last_stats = curr_stats
                print(f"[{row['timestamp']}] Lat: {row['avg_latency_ms']:5.2f}ms | Util: {row['io_utilization_pct']:5.2f}% | Temp: {last_smart['temperature']}C", end='\r')
                
        except KeyboardInterrupt:
            print("\nCollection stopped.")

if __name__ == "__main__":
    main()
