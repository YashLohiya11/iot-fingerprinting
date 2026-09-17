"""
Step 3 — IAT (Inter-Arrival Time) Extraction
==============================================
For each device's cleaned CSV from Step 2:
  1. Load data, check for malformed/duplicate rows
  2. Sort by timestamp (verify already sorted)
  3. Compute IAT = timestamp[i+1] - timestamp[i] within device boundaries
  4. Report raw IAT distribution: min, max, mean, median, std, 99th percentile
  5. Save IAT sequences to data/processed/{device}_iat.csv

NO clipping, filtering, or transformation applied — raw IAT values only.
Outlier handling will be decided after reviewing the distribution.
"""

import os
import sys
import csv
import math
import statistics
import yaml

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "phase1_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE_SUBSET = config["dataset"]["device_subset"]
PROCESSED_DIR = os.path.join(PROJ_ROOT, "data", "processed")


def load_and_validate(device_name):
    """
    Load a device's cleaned CSV and validate for malformed/duplicate rows.
    Returns (valid_rows, stats_dict) where valid_rows is list of (timestamp, packet_size, direction).
    """
    filepath = os.path.join(PROCESSED_DIR, f"{device_name}.csv")
    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return [], {}

    total_lines = 0
    valid_rows = []
    malformed_count = 0
    duplicate_count = 0
    malformed_examples = []

    prev_ts = None

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        # Verify expected columns
        expected = ["timestamp", "packet_size", "direction"]
        if header != expected:
            print(f"  WARNING: Unexpected header: {header} (expected {expected})")

        for line_num, row in enumerate(reader, start=2):  # line 2 onwards (1-indexed, after header)
            total_lines += 1

            # Check for malformed rows
            if len(row) != 3:
                malformed_count += 1
                if len(malformed_examples) < 3:
                    malformed_examples.append((line_num, row))
                continue

            try:
                ts = float(row[0])
                pkt_size = int(row[1])
                direction = int(row[2])
            except (ValueError, IndexError):
                malformed_count += 1
                if len(malformed_examples) < 3:
                    malformed_examples.append((line_num, row))
                continue

            # Validate direction is 0 or 1
            if direction not in (0, 1):
                malformed_count += 1
                if len(malformed_examples) < 3:
                    malformed_examples.append((line_num, row))
                continue

            # Check for exact duplicate timestamps
            if prev_ts is not None and ts == prev_ts:
                duplicate_count += 1
                # Still include the row — duplicates are reported, not dropped
            prev_ts = ts

            valid_rows.append((ts, pkt_size, direction))

    stats = {
        "total_data_lines": total_lines,
        "valid_rows": len(valid_rows),
        "malformed_rows": malformed_count,
        "duplicate_ts_rows": duplicate_count,
        "malformed_examples": malformed_examples,
    }

    return valid_rows, stats


def compute_iat(rows):
    """
    Compute Inter-Arrival Times from sorted (timestamp, pkt_size, direction) rows.
    IAT[i] = timestamp[i+1] - timestamp[i]
    Returns list of IAT values (floats, in seconds).
    """
    if len(rows) < 2:
        return []

    # Verify timestamps are sorted
    is_sorted = all(rows[i][0] <= rows[i + 1][0] for i in range(len(rows) - 1))
    if not is_sorted:
        print("    WARNING: Timestamps not sorted! Sorting now...")
        rows.sort(key=lambda x: x[0])

    iats = []
    for i in range(len(rows) - 1):
        iat = rows[i + 1][0] - rows[i][0]
        iats.append(iat)

    return iats


def iat_statistics(iats):
    """Compute descriptive statistics for IAT values."""
    if not iats:
        return {}

    sorted_iats = sorted(iats)
    n = len(sorted_iats)

    # Percentile calculation
    def percentile(data, p):
        k = (len(data) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        return data[f] * (c - k) + data[c] * (k - f)

    return {
        "count": n,
        "min": sorted_iats[0],
        "max": sorted_iats[-1],
        "mean": statistics.mean(iats),
        "median": statistics.median(iats),
        "std": statistics.stdev(iats) if n > 1 else 0.0,
        "p1": percentile(sorted_iats, 1),
        "p5": percentile(sorted_iats, 5),
        "p25": percentile(sorted_iats, 25),
        "p75": percentile(sorted_iats, 75),
        "p95": percentile(sorted_iats, 95),
        "p99": percentile(sorted_iats, 99),
        "p999": percentile(sorted_iats, 99.9),
        "zero_count": sum(1 for x in iats if x == 0.0),
        "negative_count": sum(1 for x in iats if x < 0.0),
    }


def save_iat(device_name, rows, iats):
    """
    Save IAT data alongside original fields.
    Each row: timestamp, packet_size, direction, iat
    (First packet has no IAT predecessor — it is excluded from the IAT file.)
    """
    output_path = os.path.join(PROCESSED_DIR, f"{device_name}_iat.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "packet_size", "direction", "iat"])
        for i, iat_val in enumerate(iats):
            ts, pkt_size, direction = rows[i + 1]  # IAT[i] corresponds to rows[i+1]
            writer.writerow([f"{ts:.6f}", pkt_size, direction, f"{iat_val:.9f}"])
    return output_path


def format_time(seconds):
    """Format a duration in seconds to a human-readable string."""
    if seconds < 0.001:
        return f"{seconds*1_000_000:.2f} us"
    elif seconds < 1.0:
        return f"{seconds*1000:.3f} ms"
    elif seconds < 60:
        return f"{seconds:.3f} s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        return f"{seconds/3600:.2f} hrs"


def main():
    print("=" * 80)
    print("  STEP 3: IAT (INTER-ARRIVAL TIME) EXTRACTION")
    print("=" * 80)

    all_validation = []
    all_iat_stats = []

    for device_name in DEVICE_SUBSET:
        print(f"\n  {'='*60}")
        print(f"  Device: {device_name}")
        print(f"  {'='*60}")

        # ── Load and validate ──────────────────────────────────────────
        rows, validation = load_and_validate(device_name)
        all_validation.append((device_name, validation))

        print(f"  Data validation:")
        print(f"    Total data lines:     {validation['total_data_lines']:,}")
        print(f"    Valid rows:           {validation['valid_rows']:,}")
        print(f"    Malformed rows:       {validation['malformed_rows']:,}")
        print(f"    Duplicate timestamps: {validation['duplicate_ts_rows']:,}")
        if validation['malformed_examples']:
            print(f"    Malformed examples:")
            for line_num, row in validation['malformed_examples']:
                print(f"      Line {line_num}: {row}")

        if not rows:
            print(f"  ERROR: No valid data for {device_name}")
            continue

        # ── Compute IAT ────────────────────────────────────────────────
        iats = compute_iat(rows)
        stats = iat_statistics(iats)
        all_iat_stats.append((device_name, stats))

        print(f"\n  IAT Distribution (raw, unfiltered):")
        print(f"    Count:               {stats['count']:,}")
        print(f"    Min:                 {format_time(stats['min'])} ({stats['min']:.9f} s)")
        print(f"    Max:                 {format_time(stats['max'])} ({stats['max']:.9f} s)")
        print(f"    Mean:                {format_time(stats['mean'])} ({stats['mean']:.6f} s)")
        print(f"    Median:              {format_time(stats['median'])} ({stats['median']:.6f} s)")
        print(f"    Std Dev:             {format_time(stats['std'])} ({stats['std']:.6f} s)")
        print(f"    1st percentile:      {format_time(stats['p1'])}")
        print(f"    5th percentile:      {format_time(stats['p5'])}")
        print(f"    25th percentile:     {format_time(stats['p25'])}")
        print(f"    75th percentile:     {format_time(stats['p75'])}")
        print(f"    95th percentile:     {format_time(stats['p95'])}")
        print(f"    99th percentile:     {format_time(stats['p99'])}")
        print(f"    99.9th percentile:   {format_time(stats['p999'])}")
        print(f"    Zero IAT count:      {stats['zero_count']:,}")
        print(f"    Negative IAT count:  {stats['negative_count']:,}")

        # ── Save IAT ──────────────────────────────────────────────────
        out_path = save_iat(device_name, rows, iats)
        print(f"\n  IAT file saved: {out_path}")

        # ── Show first 5 IAT values ───────────────────────────────────
        print(f"\n  Sample IAT values (first 5):")
        print(f"    {'timestamp':<22} {'pkt_size':>10} {'dir':>5} {'iat':>18} {'iat_human':>14}")
        print(f"    {'-'*72}")
        for i in range(min(5, len(iats))):
            ts, pkt_size, direction = rows[i + 1]
            iat_val = iats[i]
            print(f"    {ts:<22.6f} {pkt_size:>10} {direction:>5} {iat_val:>18.9f} {format_time(iat_val):>14}")

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY TABLES
    # ══════════════════════════════════════════════════════════════════

    # ── Malformed/Duplicate Summary ────────────────────────────────────
    print(f"\n\n{'='*90}")
    print("  MALFORMED & DUPLICATE ROW SUMMARY (from Step 2 output)")
    print(f"{'='*90}")
    print(f"  {'Device':<28} {'Total Lines':>12} {'Valid':>12} {'Malformed':>12} {'Dup TS':>12}")
    print(f"  {'-'*78}")
    for device_name, v in all_validation:
        print(f"  {device_name:<28} {v['total_data_lines']:>12,} {v['valid_rows']:>12,} "
              f"{v['malformed_rows']:>12,} {v['duplicate_ts_rows']:>12,}")
    print(f"{'='*90}")

    # ── IAT Distribution Summary ──────────────────────────────────────
    print(f"\n{'='*130}")
    print("  IAT DISTRIBUTION SUMMARY (raw, unfiltered)")
    print(f"{'='*130}")
    print(f"  {'Device':<28} {'Count':>10} {'Min':>14} {'Median':>14} {'Mean':>14} "
          f"{'Std':>14} {'P99':>14} {'Max':>14} {'Zero':>8} {'Neg':>6}")
    print(f"  {'-'*128}")
    for device_name, s in all_iat_stats:
        print(f"  {device_name:<28} {s['count']:>10,} {format_time(s['min']):>14} "
              f"{format_time(s['median']):>14} {format_time(s['mean']):>14} "
              f"{format_time(s['std']):>14} {format_time(s['p99']):>14} "
              f"{format_time(s['max']):>14} {s['zero_count']:>8,} {s['negative_count']:>6,}")
    print(f"{'='*130}")

    print(f"\n  Step 3 complete. Awaiting review on IAT distribution before deciding")
    print(f"  on outlier handling / clipping strategy for Step 4.")


if __name__ == "__main__":
    main()
