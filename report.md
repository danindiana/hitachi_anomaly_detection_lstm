# Hitachi Disk Analysis Report

- **Device:** /dev/sdb
- **Model:** Hitachi HUA723020ALA640
- **Serial Number:** MK0171YFJHSSDA
- **Current Temperature:** 39°C (Min/Max 13/66)
- **SMART Health Status:** PASSED
- **Power-on Hours:** 42,142 (approx. 4.8 years)
- **Reallocated Sector Count:** 541 (RAW)
- **Reallocated Event Count:** 717 (RAW)
- **SATA Version:** 2.6, 6.0 Gb/s

## Observations

- The drive is currently operating at a stable temperature of 39°C.
- While the overall health is 'PASSED', the Reallocated Sector Count is 541, which suggests some physical surface degradation over its 4.8 years of service. 
- No current pending sectors or offline uncorrectables were detected.

## Deep Dive Metrics Breakdown

- **Total Lifespan:** 1,755.9 days (~4 years, 9 months, 20 days).
- **Spin-up Frequency:** 18,137 total start/stops. On average, this disk has spun up **10.3 times per day**.
- **Power Cycle Persistence:** Only 725 hard power cycles. This means the system stays on for an average of **58 hours (2.4 days) per boot**.
- **Sleep vs. Power:** With 18,137 start/stops vs 725 power cycles, the drive has entered a standby/sleep state roughly **25 times** for every 1 time the computer was actually turned off.
- **Head Health (Load Cycles):** 18,621 load/unload cycles. For an enterprise drive, this is very low (many are rated for 300,000+), suggesting it doesn't park its heads aggressively.
- **Surface Degradation:** 541 reallocated sectors. Each sector is 512 bytes, so only **277 KB** of the 2 TB has been moved to the spare area due to physical flaws. While the number is high, it represents a tiny fraction of the disk.

## Feature Collection & Stress Test (Mar 26)

- **Test Load:** 512MB write via `dd` (direct I/O).
- **Peak Throughput Observed:** 126.9 MB/s.
- **Peak Latency Observed:** 9.98ms.
- **Peak Utilization:** 71.2%.
- **Status:** Collector script (`collect_disk_features.py`) verified and generating `disk_features.csv`.

## Neural Network Model Design (Proposed)

### Architecture: LSTM (Long Short-Term Memory)

**Reasoning:** Disk failure is often preceded by temporal patterns (e.g., a gradual increase in latency jitter over several minutes). LSTMs are designed to capture these time-series dependencies.

### Input Features (Windowed):
- Read/Write KB/s
- Average Latency (ms)
- I/O Utilization (%)
- Temperature Δ (Rate of change)
- Reallocated Sector Count Δ

### Target:
- **Primary:** Anomaly Score (reconstruction error from an Autoencoder).
- **Secondary:** RUL (Remaining Useful Life) estimation based on degradation slope.
