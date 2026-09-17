"""
Step 5 — DWT Feature Extraction + Raw Statistical Features
============================================================
For each windowed log1p-IAT file from Step 4:

A) DWT Features (db4, level 3):
   - Decompose each 128-length window into [cA3, cD3, cD2, cD1]
   - For each coefficient array: mean, variance, std, energy, entropy
   - Result: 4 levels × 5 stats = 20 DWT features per window

B) Raw Statistical Features (no DWT, same log1p windows):
   - mean, std, min, max, median, skew, kurtosis
   - P25, P75, P95, P99
   - energy, entropy
   - Result: 13 features per window

Both saved to data/features/ with device_label and time_block_id carried through.
No scaling/normalization — that happens in Step 6, fit on training split only.
"""

import os
import csv
import math
import statistics
import yaml
import pywt
import numpy as np

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "phase1_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE_SUBSET = config["dataset"]["device_subset"]
WAVELET_FAMILY = config["features"]["wavelet_family"]
DECOMP_LEVEL = config["features"]["decomposition_level"]
WINDOW_SIZE = config["preprocessing"]["window_size"]

FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")


# ══════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def shannon_entropy(values):
    """Compute Shannon entropy from an array of values.
    Uses normalized squared magnitudes as a probability distribution."""
    sq = np.array(values) ** 2
    total = np.sum(sq)
    if total == 0:
        return 0.0
    probs = sq / total
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def compute_dwt_features(window_values):
    """
    Apply DWT decomposition and extract statistical features per level.
    Returns: (feature_dict, coeff_names)
    """
    coeffs = pywt.wavedec(window_values, WAVELET_FAMILY, level=DECOMP_LEVEL)
    # coeffs = [cA3, cD3, cD2, cD1] for level=3

    level_names = [f"cA{DECOMP_LEVEL}"]
    for i in range(DECOMP_LEVEL, 0, -1):
        level_names.append(f"cD{i}")

    features = {}
    for name, coeff_arr in zip(level_names, coeffs):
        arr = np.array(coeff_arr, dtype=np.float64)
        features[f"dwt_{name}_mean"] = float(np.mean(arr))
        features[f"dwt_{name}_var"] = float(np.var(arr))
        features[f"dwt_{name}_std"] = float(np.std(arr))
        features[f"dwt_{name}_energy"] = float(np.sum(arr ** 2))
        features[f"dwt_{name}_entropy"] = float(shannon_entropy(arr))

    return features


def compute_raw_stats(window_values):
    """
    Compute raw statistical features directly on the log1p-transformed IAT window.
    Returns: feature_dict
    """
    arr = np.array(window_values, dtype=np.float64)
    n = len(arr)

    # Basic stats
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    median_val = float(np.median(arr))

    # Skewness (Fisher's definition)
    if std_val > 0:
        skew_val = float(np.mean(((arr - mean_val) / std_val) ** 3))
    else:
        skew_val = 0.0

    # Kurtosis (excess kurtosis, Fisher's definition)
    if std_val > 0:
        kurt_val = float(np.mean(((arr - mean_val) / std_val) ** 4) - 3.0)
    else:
        kurt_val = 0.0

    # Percentiles
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    # Energy and entropy
    energy = float(np.sum(arr ** 2))
    entropy = float(shannon_entropy(arr))

    return {
        "raw_mean": mean_val,
        "raw_std": std_val,
        "raw_min": min_val,
        "raw_max": max_val,
        "raw_median": median_val,
        "raw_skew": skew_val,
        "raw_kurtosis": kurt_val,
        "raw_p25": p25,
        "raw_p75": p75,
        "raw_p95": p95,
        "raw_p99": p99,
        "raw_energy": energy,
        "raw_entropy": entropy,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def load_windows(device_name):
    """Load windowed data from Step 4 output."""
    filepath = os.path.join(FEATURES_DIR, f"{device_name}_windows.csv")
    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return []

    windows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iat_values = [float(row[f"iat_{i}"]) for i in range(WINDOW_SIZE)]
            windows.append({
                "device_label": row["device_label"],
                "window_id": int(row["window_id"]),
                "time_block_id": int(row["time_block_id"]),
                "start_timestamp": row["start_timestamp"],
                "end_timestamp": row["end_timestamp"],
                "iat_values": iat_values,
            })
    return windows


def process_device(device_name):
    """Process all windows for a device, extracting both feature sets."""
    windows = load_windows(device_name)
    if not windows:
        return [], []

    dwt_rows = []
    raw_rows = []

    for w in windows:
        iat_vals = w["iat_values"]
        meta = {
            "device_label": w["device_label"],
            "window_id": w["window_id"],
            "time_block_id": w["time_block_id"],
        }

        # DWT features
        dwt_feats = compute_dwt_features(iat_vals)
        dwt_row = {**meta, **dwt_feats}
        dwt_rows.append(dwt_row)

        # Raw statistical features
        raw_feats = compute_raw_stats(iat_vals)
        raw_row = {**meta, **raw_feats}
        raw_rows.append(raw_row)

    return dwt_rows, raw_rows


def save_features(rows, filename, feature_names):
    """Save feature rows to CSV."""
    if not rows:
        return None

    filepath = os.path.join(FEATURES_DIR, filename)
    meta_cols = ["device_label", "window_id", "time_block_id"]
    all_cols = meta_cols + feature_names

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for row in rows:
            # Format floats
            formatted = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted[k] = f"{v:.9f}"
                else:
                    formatted[k] = v
            writer.writerow(formatted)
    return filepath


def main():
    print("=" * 80)
    print("  STEP 5: DWT FEATURE EXTRACTION + RAW STATISTICAL FEATURES")
    print(f"  Wavelet: {WAVELET_FAMILY}, Level: {DECOMP_LEVEL}")
    print(f"  Window size: {WINDOW_SIZE}")
    print("=" * 80)

    # Verify DWT decomposition structure
    dummy = np.zeros(WINDOW_SIZE)
    coeffs = pywt.wavedec(dummy, WAVELET_FAMILY, level=DECOMP_LEVEL)
    level_names = [f"cA{DECOMP_LEVEL}"]
    for i in range(DECOMP_LEVEL, 0, -1):
        level_names.append(f"cD{i}")

    print(f"\n  DWT decomposition structure for {WINDOW_SIZE}-length window:")
    for name, c in zip(level_names, coeffs):
        print(f"    {name}: {len(c)} coefficients")
    print(f"  Total DWT coefficients: {sum(len(c) for c in coeffs)}")

    # Define feature column names
    dwt_feature_names = []
    for name in level_names:
        for stat in ["mean", "var", "std", "energy", "entropy"]:
            dwt_feature_names.append(f"dwt_{name}_{stat}")

    raw_feature_names = [
        "raw_mean", "raw_std", "raw_min", "raw_max", "raw_median",
        "raw_skew", "raw_kurtosis", "raw_p25", "raw_p75", "raw_p95", "raw_p99",
        "raw_energy", "raw_entropy",
    ]

    print(f"\n  DWT feature dimensionality: {len(dwt_feature_names)}")
    print(f"  Raw stat feature dimensionality: {len(raw_feature_names)}")

    # Process all devices
    all_dwt_rows = []
    all_raw_rows = []
    device_counts = {}

    # Store sample windows for comparison
    sample_windows = {}

    for device_name in DEVICE_SUBSET:
        print(f"\n  {'='*60}")
        print(f"  Device: {device_name}")
        print(f"  {'='*60}")

        dwt_rows, raw_rows = process_device(device_name)
        print(f"  Processed {len(dwt_rows):,} windows")

        device_counts[device_name] = len(dwt_rows)
        all_dwt_rows.extend(dwt_rows)
        all_raw_rows.extend(raw_rows)

        # Store first window as sample
        if dwt_rows:
            sample_windows[device_name] = {
                "dwt": dwt_rows[0],
                "raw": raw_rows[0],
            }

    # ── Save combined feature files ────────────────────────────────────
    print(f"\n\n  {'='*60}")
    print(f"  SAVING FEATURE FILES")
    print(f"  {'='*60}")

    dwt_path = save_features(all_dwt_rows, "dwt_features.csv", dwt_feature_names)
    raw_path = save_features(all_raw_rows, "raw_stat_features.csv", raw_feature_names)

    print(f"  DWT features saved:  {dwt_path}")
    print(f"    Rows: {len(all_dwt_rows):,}, Columns: {3 + len(dwt_feature_names)} "
          f"(3 meta + {len(dwt_feature_names)} features)")
    print(f"  Raw features saved:  {raw_path}")
    print(f"    Rows: {len(all_raw_rows):,}, Columns: {3 + len(raw_feature_names)} "
          f"(3 meta + {len(raw_feature_names)} features)")

    # ── Summary Table ──────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  WINDOW & FEATURE COUNT SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Device':<28} {'Windows':>10} {'DWT Feats':>12} {'Raw Feats':>12}")
    print(f"  {'-'*64}")
    for device_name in DEVICE_SUBSET:
        n = device_counts.get(device_name, 0)
        print(f"  {device_name:<28} {n:>10,} {len(dwt_feature_names):>12} "
              f"{len(raw_feature_names):>12}")
    print(f"  {'-'*64}")
    print(f"  {'TOTAL':<28} {len(all_dwt_rows):>10,} {len(dwt_feature_names):>12} "
          f"{len(raw_feature_names):>12}")
    print(f"{'='*80}")

    # ── Side-by-Side Sanity Check ──────────────────────────────────────
    # Compare BelkinWemoMotionSensor vs TPLinkSmartPlug
    dev_a = "BelkinWemoMotionSensor"
    dev_b = "TPLinkSmartPlug"

    if dev_a in sample_windows and dev_b in sample_windows:
        print(f"\n\n{'='*100}")
        print(f"  SANITY CHECK: {dev_a} vs {dev_b} (window_id=0)")
        print(f"{'='*100}")

        # DWT features comparison
        print(f"\n  DWT Features:")
        print(f"  {'Feature':<28} {dev_a:>22} {dev_b:>22}  {'Different?':>12}")
        print(f"  {'-'*88}")
        dwt_a = sample_windows[dev_a]["dwt"]
        dwt_b = sample_windows[dev_b]["dwt"]
        for feat in dwt_feature_names:
            va = dwt_a[feat]
            vb = dwt_b[feat]
            diff = abs(va - vb)
            rel_diff = diff / max(abs(va), abs(vb), 1e-10) * 100
            marker = "YES" if rel_diff > 10 else "similar"
            print(f"  {feat:<28} {va:>22.6f} {vb:>22.6f}  {marker:>12}")

        # Raw features comparison
        print(f"\n  Raw Statistical Features:")
        print(f"  {'Feature':<28} {dev_a:>22} {dev_b:>22}  {'Different?':>12}")
        print(f"  {'-'*88}")
        raw_a = sample_windows[dev_a]["raw"]
        raw_b = sample_windows[dev_b]["raw"]
        for feat in raw_feature_names:
            va = raw_a[feat]
            vb = raw_b[feat]
            diff = abs(va - vb)
            rel_diff = diff / max(abs(va), abs(vb), 1e-10) * 100
            marker = "YES" if rel_diff > 10 else "similar"
            print(f"  {feat:<28} {va:>22.6f} {vb:>22.6f}  {marker:>12}")

    # ── Feature Column Listing ─────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  FEATURE COLUMN LISTING")
    print(f"{'='*60}")
    print(f"\n  DWT Features ({len(dwt_feature_names)}):")
    for i, name in enumerate(dwt_feature_names):
        print(f"    {i:>2}. {name}")
    print(f"\n  Raw Statistical Features ({len(raw_feature_names)}):")
    for i, name in enumerate(raw_feature_names):
        print(f"    {i:>2}. {name}")

    print(f"\n  Step 5 complete. Awaiting review before proceeding to Step 6.")


if __name__ == "__main__":
    main()
