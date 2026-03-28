import os

# Disk Configuration
DISK = "sdb"
DEV_PATH = f"/dev/{DISK}"

# Data Collection Configuration
POLL_INTERVAL = 1.0    # Seconds between I/O samples
SMART_INTERVAL = 60.0  # Seconds between SMART samples
DATA_FILE = "disk_features.csv"

# Model Configuration
WINDOW_SIZE = 10
HIDDEN_DIM = 64        
LATENT_DIM = 16        
EPOCHS = 100           
BATCH_SIZE = 32        
LEARNING_RATE = 0.001
PATIENCE = 10          
MIN_EPOCHS = 20        

# SMART Thresholds for Alerting (Edge-trigger baseline)
REALLOCATED_BASELINE = 541  
PENDING_BASELINE = 0

# Paths (Relative to this config file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "disk_model.pth")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler_params.npy")
THRESHOLDS_PATH = os.path.join(SCRIPT_DIR, "thresholds.npy")
FEATURE_NAMES_PATH = os.path.join(SCRIPT_DIR, "feature_names.txt")

# Features to use for training/inference
FEATURES = [
    "read_kb_s", 
    "write_kb_s", 
    "avg_latency_ms", 
    "io_utilization_pct", 
    "temperature",
    "reallocated_sectors",
    "pending_sectors",
    "offline_uncorrectable"
]
