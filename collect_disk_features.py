import os
import time
import subprocess
import csv
from datetime import datetime

# Configuration
DISK = "sdb"
DEV_PATH = f"/dev/{DISK}"
POLL_INTERVAL = 1.0  # Seconds between I/O samples
SMART_INTERVAL = 60.0 # Seconds between SMART samples (don't over-poll the controller)
OUTPUT_FILE = "disk_features.csv"

def get_disk_stats():
    """Reads /proc/diskstats for the specific disk."""
    with open("/proc/diskstats", "r") as f:
        for line in f:
            parts = line.split()
            if parts[2] == DISK:
                # https://www.kernel.org/doc/Documentation/iostats.txt
                return {
                    "reads": int(parts[3]),
                    "reads_merged": int(parts[4]),
                    "sectors_read": int(parts[5]),
                    "time_reading": int(parts[6]),
                    "writes": int(parts[7]),
                    "writes_merged": int(parts[8]),
                    "sectors_written": int(parts[9]),
                    "time_writing": int(parts[10]),
                    "io_in_progress": int(parts[11]),
                    "io_time": int(parts[12]),
                    "weighted_io_time": int(parts[13])
                }
    return None

def get_smart_metrics():
    """Polls smartctl for critical health indicators."""
    try:
        # We use -A for attributes only, much faster than -a
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
            elif attr_name == "Temperature_Celsius":
                # Temperature often has (Min/Max) appended, take first number
                metrics["temperature"] = int(raw_value.split('(')[0].strip())
            elif attr_name == "Power_On_Hours":
                metrics["power_on_hours"] = int(raw_value)
        return metrics
    except Exception as e:
        print(f"Error reading SMART: {e}")
        return None

def main():
    print(f"Starting feature collection for {DEV_PATH}...")
    print(f"Outputting to {OUTPUT_FILE}")
    
    headers = [
        "timestamp", "unix_time", 
        "read_kb_s", "write_kb_s", "avg_latency_ms", "io_utilization_pct",
        "reallocated_sectors", "pending_sectors", "temperature", "power_on_hours"
    ]
    
    file_exists = os.path.isfile(OUTPUT_FILE)
    
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
            
        last_stats = get_disk_stats()
        last_smart = get_smart_metrics() or {}
        last_smart_time = time.time()
        
        try:
            while True:
                time.sleep(POLL_INTERVAL)
                curr_time = time.time()
                curr_stats = get_disk_stats()
                
                if not curr_stats or not last_stats:
                    continue
                
                # Calculate I/O deltas
                dt = POLL_INTERVAL
                reads_diff = curr_stats["reads"] - last_stats["reads"]
                writes_diff = curr_stats["writes"] - last_stats["writes"]
                sectors_read = curr_stats["sectors_read"] - last_stats["sectors_read"]
                sectors_written = curr_stats["sectors_written"] - last_stats["sectors_written"]
                time_io = (curr_stats["io_time"] - last_stats["io_time"])
                
                # Feature calculation
                read_kb_s = (sectors_read * 512) / 1024 / dt
                write_kb_s = (sectors_written * 512) / 1024 / dt
                
                total_ops = reads_diff + writes_diff
                if total_ops > 0:
                    # Avg latency in ms
                    total_time = (curr_stats["time_reading"] - last_stats["time_reading"]) + \
                                 (curr_stats["time_writing"] - last_stats["time_writing"])
                    avg_latency = total_time / total_ops
                else:
                    avg_latency = 0
                    
                utilization = (time_io / (dt * 1000)) * 100
                
                # Periodically update SMART
                if curr_time - last_smart_time > SMART_INTERVAL:
                    curr_smart = get_smart_metrics()
                    if curr_smart:
                        last_smart = curr_smart
                        last_smart_time = curr_time
                
                # Write row
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "unix_time": curr_time,
                    "read_kb_s": round(read_kb_s, 2),
                    "write_kb_s": round(write_kb_s, 2),
                    "avg_latency_ms": round(avg_latency, 2),
                    "io_utilization_pct": round(utilization, 2),
                    "reallocated_sectors": last_smart.get("reallocated_sectors", 0),
                    "pending_sectors": last_smart.get("pending_sectors", 0),
                    "temperature": last_smart.get("temperature", 0),
                    "power_on_hours": last_smart.get("power_on_hours", 0)
                }
                writer.writerow(row)
                f.flush()
                
                last_stats = curr_stats
                print(f"[{row['timestamp']}] Latency: {row['avg_latency_ms']}ms | Util: {row['io_utilization_pct']}%", end='\r')
                
        except KeyboardInterrupt:
            print("\nCollection stopped.")

if __name__ == "__main__":
    main()
