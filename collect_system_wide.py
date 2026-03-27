import os
import time
import subprocess
import csv
from datetime import datetime

# Configuration
# Monitoring all physical disks + RAID0
DISKS = ["sda", "sdb", "sdc", "sdd", "sde", "sdf", "sdg", "nvme0n1", "nvme1n1", "md0"]
POLL_INTERVAL = 1.0
SMART_INTERVAL = 60.0
OUTPUT_FILE = "system_wide_features.csv"

def get_disk_stats():
    """Reads /proc/diskstats for all tracked disks."""
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
    """Reads Slab and Percpu memory from /proc/meminfo."""
    mem = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("Slab:"):
                mem["slab_kb"] = int(line.split()[1])
            elif line.startswith("Percpu:"):
                mem["percpu_kb"] = int(line.split()[1])
    return mem

def get_smart_temp(disk):
    """Fetches temperature for a disk via smartctl."""
    # Skip md0 and partitions
    if disk.startswith("md"): return 0
    try:
        res = subprocess.check_output(["sudo", "smartctl", "-A", f"/dev/{disk}"], stderr=subprocess.DEVNULL).decode()
        for line in res.splitlines():
            if "Temperature_Celsius" in line:
                return int(line.split()[9].split('(')[0])
    except:
        pass
    return 0

def main():
    print(f"Starting system-wide feature collection for {len(DISKS)} devices...")
    
    # Dynamically build headers
    headers = ["timestamp", "unix_time", "slab_kb", "percpu_kb"]
    for d in DISKS:
        headers += [f"{d}_read_kb_s", f"{d}_write_kb_s", f"{d}_latency_ms", f"{d}_util_pct"]
    
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
            
        last_stats = get_disk_stats()
        last_smart_time = 0
        temps = {d: 0 for d in DISKS}
        
        try:
            while True:
                time.sleep(POLL_INTERVAL)
                curr_time = time.time()
                curr_stats = get_disk_stats()
                mem = get_mem_stats()
                
                if not curr_stats or not last_stats: continue
                
                row = {
                    "timestamp": datetime.now().isoformat(),
                    "unix_time": curr_time,
                    "slab_kb": mem.get("slab_kb", 0),
                    "percpu_kb": mem.get("percpu_kb", 0)
                }
                
                # Periodically update temps for physical disks
                if curr_time - last_smart_time > SMART_INTERVAL:
                    # Just sample sdb (Hitachi) for now to keep it fast, or loop all
                    temps["sdb"] = get_smart_temp("sdb")
                    last_smart_time = curr_time
                
                for d in DISKS:
                    if d in curr_stats and d in last_stats:
                        c = curr_stats[d]
                        l = last_stats[d]
                        
                        dt = POLL_INTERVAL
                        read_kb = (c["read_bytes"] - l["read_bytes"]) / 1024
                        write_kb = (c["write_bytes"] - l["write_bytes"]) / 1024
                        
                        total_ops = (c["reads"] - l["reads"]) + (c["writes"] - l["writes"])
                        lat = 0
                        if total_ops > 0:
                            total_time = (c["time_reading"] - l["time_reading"]) + \
                                         (c["time_writing"] - l["time_writing"])
                            lat = total_time / total_ops
                            
                        util = ((c["io_time"] - l["io_time"]) / 1000) * 100
                        
                        row[f"{d}_read_kb_s"] = round(read_kb, 2)
                        row[f"{d}_write_kb_s"] = round(write_kb, 2)
                        row[f"{d}_latency_ms"] = round(lat, 2)
                        row[f"{d}_util_pct"] = round(util, 1)
                    else:
                        # Placeholder for missing disks
                        row[f"{d}_read_kb_s"] = 0
                        row[f"{d}_write_kb_s"] = 0
                        row[f"{d}_latency_ms"] = 0
                        row[f"{d}_util_pct"] = 0
                        
                writer.writerow(row)
                f.flush()
                last_stats = curr_stats
                
                # Live status for sdb and md0
                print(f"[{row['timestamp']}] sdb Lat: {row['sdb_latency_ms']}ms | md0 Util: {row['md0_util_pct']}% | Slab: {row['slab_kb']}KB", end='\r')
                
        except KeyboardInterrupt:
            print("\nCollection stopped.")

if __name__ == "__main__":
    main()
