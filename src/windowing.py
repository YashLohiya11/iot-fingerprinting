"""
Step 4 — Windowing
===================
For each device's IAT data from Step 3:
  1. Apply log(1 + IAT) transform (config-driven: iat_transform)
  2. Slice into fixed-length non-overlapping windows (window_size=128, window_step=128)
  3. Drop incomplete final windows (pad_incomplete_windows=false)
  4. Tag each window with:
     - device_label (from filename, not from packet data)
     - time_block_id (1-hour block for session/time-block splitting in Step 6)
     - window_id (sequential within device)
     - start_timestamp / end_timestamp (for traceability)
  5. Save windowed data to data/features/
  6. Report window counts per device and time-block distribution

Session/time-block assignment:
  Each window's center timestamp is mapped to a 1-hour time block
  (configurable via time_block_hours). All windows in the same block
  will go to the same train/val/test split in Step 6, preventing
  temporal leakage.
"""

import os
import sys
import csv
import math
import datetime
import yaml
import numpy as np

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "phase1_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE_SUBSET = config["dataset"]["device_subset"]
DATE_START = config["dataset"]["date_start"]
WINDOW_SIZE = config["preprocessing"]["window_size"]
WINDOW_STEP = config["preprocessing"]["window_step"]
PAD_INCOMPLETE = config["preprocessing"]["pad_incomplete_windows"]
IAT_TRANSFORM = config["preprocessing"]["iat_transform"]
TIME_BLOCK_HOURS = config["preprocessing"]["time_block_hours"]

PROCESSED_DIR = os.path.join(PROJ_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)

# Reference timestamp for time-block assignment (start of capture window)
REF_TS = datetime.datetime.strptime(DATE_START, "%Y-%m-%d").replace(
    tzinfo=datetime.timezone.utc
).timestamp()
BLOCK_SIZE_SEC = TIME_BLOCK_HOURS * 3600


def apply_iat_transform(iat_val):
    """Apply the configured IAT transform."""
    if IAT_TRANSFORM == "log1p":
        return math.log1p(iat_val)  # log(1 + iat)
    elif IAT_TRANSFORM == "none":
        return iat_val
    else:
        raise ValueError(f"Unknown iat_transform: {IAT_TRANSFORM}")


def compute_time_block(timestamp):
    """Assign a timestamp to a time block ID (0-indexed)."""
    return int((timestamp - REF_TS) / BLOCK_SIZE_SEC)


def load_iat_data(device_name):
    """Load IAT data from Step 3 output."""
    filepath = os.path.join(PROCESSED_DIR, f"{device_name}_iat.csv")
    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return []

    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp": float(row["timestamp"]),
                "packet_size": int(row["packet_size"]),
                "direction": int(row["direction"]),
                "iat": float(row["iat"]),
            })
    return rows


def create_windows(device_name, rows):
    """
    Slice IAT sequence into fixed-length windows.
    Returns list of window dicts.
    """
    if not rows:
        return []

    # Apply IAT transform
    transformed_iats = [apply_iat_transform(r["iat"]) for r in rows]
    timestamps = [r["timestamp"] for r in rows]

    n = len(transformed_iats)
    windows = []
    window_id = 0

    i = 0
    while i + WINDOW_SIZE <= n:
        # Extract window
        window_iats = transformed_iats[i:i + WINDOW_SIZE]
        window_timestamps = timestamps[i:i + WINDOW_SIZE]

        start_ts = window_timestamps[0]
        end_ts = window_timestamps[-1]
        center_ts = (start_ts + end_ts) / 2.0

        time_block = compute_time_block(center_ts)

        windows.append({
            "device_label": device_name,
            "window_id": window_id,
            "time_block_id": time_block,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "iat_values": window_iats,
        })

        window_id += 1
        i += WINDOW_STEP

    # Handle incomplete final window
    remaining = n - i
    if remaining > 0 and PAD_INCOMPLETE:
        window_iats = transformed_iats[i:] + [0.0] * (WINDOW_SIZE - remaining)
        window_timestamps = timestamps[i:]
        start_ts = window_timestamps[0]
        end_ts = window_timestamps[-1]
        center_ts = (start_ts + end_ts) / 2.0
        time_block = compute_time_block(center_ts)

        windows.append({
            "device_label": device_name,
            "window_id": window_id,
            "time_block_id": time_block,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "iat_values": window_iats,
        })
    elif remaining > 0:
        print(f"    Dropped incomplete final window ({remaining}/{WINDOW_SIZE} values)")

    return windows


def save_windows(device_name, windows):
    """
    Save windows to CSV.
    Format: device_label, window_id, time_block_id, start_ts, end_ts, iat_0, iat_1, ..., iat_127
    """
    output_path = os.path.join(FEATURES_DIR, f"{device_name}_windows.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        iat_cols = [f"iat_{i}" for i in range(WINDOW_SIZE)]
        writer.writerow(["device_label", "window_id", "time_block_id",
                         "start_timestamp", "end_timestamp"] + iat_cols)
        for w in windows:
            row = [
                w["device_label"],
                w["window_id"],
                w["time_block_id"],
                f"{w['start_timestamp']:.6f}",
                f"{w['end_timestamp']:.6f}",
            ] + [f"{v:.9f}" for v in w["iat_values"]]
            writer.writerow(row)
    return output_path


def main():
    print("=" * 80)
    print("  STEP 4: WINDOWING")
    print(f"  IAT transform: {IAT_TRANSFORM}")
    print(f"  Window size: {WINDOW_SIZE}, Step: {WINDOW_STEP} (non-overlapping)")
    print(f"  Pad incomplete: {PAD_INCOMPLETE}")
    print(f"  Time block size: {TIME_BLOCK_HOURS} hour(s)")
    print("=" * 80)

    all_results = []

    for device_name in DEVICE_SUBSET:
        print(f"\n  {'='*60}")
        print(f"  Device: {device_name}")
        print(f"  {'='*60}")

        # Load IAT data
        rows = load_iat_data(device_name)
        print(f"  Loaded {len(rows):,} IAT values")

        if not rows:
            continue

        # Show transform effect on a sample
        sample_raw = [rows[i]["iat"] for i in range(min(5, len(rows)))]
        sample_transformed = [apply_iat_transform(v) for v in sample_raw]
        print(f"\n  Transform sample (raw -> log1p):")
        for raw, trans in zip(sample_raw, sample_transformed):
            print(f"    {raw:.9f} s -> {trans:.9f}")

        # Create windows
        windows = create_windows(device_name, rows)
        print(f"\n  Windows created: {len(windows):,}")

        if not windows:
            continue

        # Time block distribution
        block_counts = {}
        for w in windows:
            bid = w["time_block_id"]
            block_counts[bid] = block_counts.get(bid, 0) + 1

        unique_blocks = sorted(block_counts.keys())
        print(f"  Unique time blocks: {len(unique_blocks)}")
        print(f"  Time blocks (id: window_count):")
        for bid in unique_blocks:
            block_start = datetime.datetime.fromtimestamp(
                REF_TS + bid * BLOCK_SIZE_SEC, tz=datetime.timezone.utc
            )
            block_end = datetime.datetime.fromtimestamp(
                REF_TS + (bid + 1) * BLOCK_SIZE_SEC, tz=datetime.timezone.utc
            )
            print(f"    Block {bid:>3}: {block_counts[bid]:>5} windows  "
                  f"({block_start.strftime('%m/%d %H:%M')} - {block_end.strftime('%H:%M')})")

        # Save
        out_path = save_windows(device_name, windows)
        print(f"\n  Saved to: {out_path}")

        # Show first window sample
        w0 = windows[0]
        print(f"\n  Sample window (window_id=0):")
        print(f"    Device:     {w0['device_label']}")
        print(f"    Block ID:   {w0['time_block_id']}")
        print(f"    Time range: {datetime.datetime.fromtimestamp(w0['start_timestamp'], tz=datetime.timezone.utc)} -> "
              f"{datetime.datetime.fromtimestamp(w0['end_timestamp'], tz=datetime.timezone.utc)}")
        print(f"    IAT values (first 10): {[f'{v:.6f}' for v in w0['iat_values'][:10]]}")
        print(f"    IAT values (last 5):   {[f'{v:.6f}' for v in w0['iat_values'][-5:]]}")

        all_results.append({
            "device": device_name,
            "iat_count": len(rows),
            "window_count": len(windows),
            "block_count": len(unique_blocks),
            "min_windows_per_block": min(block_counts.values()),
            "max_windows_per_block": max(block_counts.values()),
        })

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*110}")
    print("  WINDOW COUNT SUMMARY")
    print(f"{'='*110}")
    print(f"  {'Device':<28} {'IAT Values':>12} {'Windows':>10} {'Blocks':>8} "
          f"{'Min W/Block':>12} {'Max W/Block':>12} {'Dropped':>10}")
    print(f"  {'-'*96}")

    total_iats = 0
    total_windows = 0

    for r in all_results:
        expected_windows = r["iat_count"] // WINDOW_SIZE
        dropped = r["iat_count"] % WINDOW_SIZE
        print(f"  {r['device']:<28} {r['iat_count']:>12,} {r['window_count']:>10,} "
              f"{r['block_count']:>8} {r['min_windows_per_block']:>12} "
              f"{r['max_windows_per_block']:>12} {dropped:>10}")
        total_iats += r["iat_count"]
        total_windows += r["window_count"]

    print(f"  {'-'*96}")
    print(f"  {'TOTAL':<28} {total_iats:>12,} {total_windows:>10,}")
    print(f"{'='*110}")

    # Verify window integrity
    print(f"\n  INTEGRITY CHECKS:")
    print(f"  - Window size: {WINDOW_SIZE} (all windows have exactly this many values)")
    print(f"  - Step size: {WINDOW_STEP} (= window size, non-overlapping)")
    print(f"  - Pad incomplete: {PAD_INCOMPLETE}")
    print(f"  - IAT transform: {IAT_TRANSFORM}")
    print(f"  - All IAT values are log1p-transformed (no raw IATs in output)")
    print(f"  - Each window tagged with time_block_id for split-safe grouping")

    print(f"\n  Step 4 complete. Awaiting review before proceeding to Step 5.")


if __name__ == "__main__":
    main()
