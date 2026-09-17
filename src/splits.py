"""
Step 6 — Leakage-Safe Train/Val/Test Split + Scaling
=====================================================
1. Load DWT and raw feature sets from Step 5
2. Split by time_block_id (contiguous temporal), NOT random shuffle (Hard Rule 2)
3. Ratios: 70/15/15 from config
4. Automated no-overlap check: verify no time_block_id in >1 split
5. Fit StandardScaler on training split only (Hard Rule 3), save to disk
6. Apply scaler to val/test, save all scaled features + split assignments
7. Report per-device, per-split window counts
"""

import os
import csv
import math
import pickle
import yaml
import numpy as np

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "phase1_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE_SUBSET = config["dataset"]["device_subset"]
TRAIN_RATIO = config["splits"]["train_ratio"]
VAL_RATIO = config["splits"]["val_ratio"]
TEST_RATIO = config["splits"]["test_ratio"]
SEED = config["dataset"]["random_seed"]

FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_feature_csv(filename):
    """Load a feature CSV, returning rows as list of dicts and feature column names."""
    filepath = os.path.join(FEATURES_DIR, filename)
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    meta_cols = ["device_label", "window_id", "time_block_id"]
    feature_cols = [c for c in fieldnames if c not in meta_cols]
    return rows, feature_cols


def assign_splits(all_block_ids):
    """
    Assign time blocks to train/val/test using contiguous temporal split.
    This is the strongest guarantee against temporal leakage.
    Returns: dict mapping block_id -> split_name
    """
    sorted_blocks = sorted(all_block_ids)
    n = len(sorted_blocks)

    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)
    # n_test gets the remainder
    n_test = n - n_train - n_val

    # Ensure all splits have at least 1 block
    if n_test < 1:
        n_test = 1
        n_val = min(n_val, n - n_train - 1)
    if n_val < 1:
        n_val = 1
        n_train = n - n_val - n_test

    train_blocks = sorted_blocks[:n_train]
    val_blocks = sorted_blocks[n_train:n_train + n_val]
    test_blocks = sorted_blocks[n_train + n_val:]

    assignment = {}
    for b in train_blocks:
        assignment[b] = "train"
    for b in val_blocks:
        assignment[b] = "val"
    for b in test_blocks:
        assignment[b] = "test"

    return assignment, train_blocks, val_blocks, test_blocks


def verify_no_overlap(train_blocks, val_blocks, test_blocks):
    """Verify no time_block_id appears in more than one split."""
    train_set = set(train_blocks)
    val_set = set(val_blocks)
    test_set = set(test_blocks)

    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    passed = True
    issues = []

    if train_val:
        passed = False
        issues.append(f"Train ∩ Val: {train_val}")
    if train_test:
        passed = False
        issues.append(f"Train ∩ Test: {train_test}")
    if val_test:
        passed = False
        issues.append(f"Val ∩ Test: {val_test}")

    return passed, issues


class StandardScaler:
    """Simple StandardScaler that fits on training data only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.feature_names_ = None

    def fit(self, data, feature_names):
        """Fit scaler on training data. data is a 2D numpy array."""
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        # Replace zero std with 1 to avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        self.feature_names_ = feature_names
        return self

    def transform(self, data):
        """Transform data using fitted parameters."""
        return (data - self.mean_) / self.std_

    def save(self, filepath):
        """Save fitted scaler to disk."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "mean": self.mean_,
                "std": self.std_,
                "feature_names": self.feature_names_,
            }, f)

    @classmethod
    def load(cls, filepath):
        """Load fitted scaler from disk."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        scaler = cls()
        scaler.mean_ = state["mean"]
        scaler.std_ = state["std"]
        scaler.feature_names_ = state["feature_names"]
        return scaler


def split_and_scale(rows, feature_cols, block_assignment, scaler_name):
    """
    Split rows by time block assignment, fit scaler on train, apply to all.
    Returns split data and the fitted scaler.
    """
    # Organize rows by split
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        block_id = int(row["time_block_id"])
        split = block_assignment.get(block_id)
        if split:
            splits[split].append(row)

    # Extract feature matrices
    def rows_to_matrix(row_list):
        return np.array([[float(r[c]) for c in feature_cols] for r in row_list])

    train_X = rows_to_matrix(splits["train"])
    val_X = rows_to_matrix(splits["val"])
    test_X = rows_to_matrix(splits["test"])

    # Fit scaler on training data ONLY
    scaler = StandardScaler()
    scaler.fit(train_X, feature_cols)

    # Transform all splits
    train_X_scaled = scaler.transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)

    # Save scaler
    scaler_path = os.path.join(RESULTS_DIR, f"{scaler_name}_scaler.pkl")
    scaler.save(scaler_path)

    return {
        "train": {"rows": splits["train"], "X": train_X, "X_scaled": train_X_scaled},
        "val": {"rows": splits["val"], "X": val_X, "X_scaled": val_X_scaled},
        "test": {"rows": splits["test"], "X": test_X, "X_scaled": test_X_scaled},
    }, scaler, scaler_path


def save_split_data(split_data, feature_cols, prefix):
    """Save scaled feature data per split."""
    meta_cols = ["device_label", "window_id", "time_block_id", "split"]
    all_cols = meta_cols + feature_cols

    for split_name in ["train", "val", "test"]:
        rows = split_data[split_name]["rows"]
        X_scaled = split_data[split_name]["X_scaled"]

        filepath = os.path.join(FEATURES_DIR, f"{prefix}_{split_name}.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(all_cols)
            for i, row in enumerate(rows):
                meta = [row["device_label"], row["window_id"],
                        row["time_block_id"], split_name]
                feats = [f"{v:.9f}" for v in X_scaled[i]]
                writer.writerow(meta + feats)


def save_split_assignment(block_assignment, train_blocks, val_blocks, test_blocks):
    """Save the split assignment mapping to disk."""
    filepath = os.path.join(RESULTS_DIR, "split_assignment.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_block_id", "split"])
        for block_id in sorted(block_assignment.keys()):
            writer.writerow([block_id, block_assignment[block_id]])

    # Also save as pickle for easy reuse
    pkl_path = os.path.join(RESULTS_DIR, "split_assignment.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "assignment": block_assignment,
            "train_blocks": train_blocks,
            "val_blocks": val_blocks,
            "test_blocks": test_blocks,
        }, f)

    return filepath, pkl_path


def main():
    print("=" * 80)
    print("  STEP 6: LEAKAGE-SAFE TRAIN/VAL/TEST SPLIT + SCALING")
    print(f"  Split ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  Split method: contiguous temporal by time_block_id")
    print("=" * 80)

    # ── Load feature sets ──────────────────────────────────────────────
    print("\n  Loading feature sets...")
    dwt_rows, dwt_feature_cols = load_feature_csv("dwt_features.csv")
    raw_rows, raw_feature_cols = load_feature_csv("raw_stat_features.csv")
    print(f"  DWT: {len(dwt_rows):,} rows, {len(dwt_feature_cols)} features")
    print(f"  Raw: {len(raw_rows):,} rows, {len(raw_feature_cols)} features")

    # ── Collect all unique time blocks ─────────────────────────────────
    all_blocks = set()
    for row in dwt_rows:
        all_blocks.add(int(row["time_block_id"]))
    all_blocks_sorted = sorted(all_blocks)
    print(f"\n  Total unique time blocks: {len(all_blocks_sorted)}")
    print(f"  Block range: {all_blocks_sorted[0]} to {all_blocks_sorted[-1]}")

    # ── Assign blocks to splits ────────────────────────────────────────
    block_assignment, train_blocks, val_blocks, test_blocks = assign_splits(all_blocks)

    print(f"\n  Block assignment:")
    print(f"    Train: blocks {min(train_blocks)}-{max(train_blocks)} "
          f"({len(train_blocks)} blocks, {len(train_blocks)/len(all_blocks)*100:.1f}%)")
    print(f"    Val:   blocks {min(val_blocks)}-{max(val_blocks)} "
          f"({len(val_blocks)} blocks, {len(val_blocks)/len(all_blocks)*100:.1f}%)")
    print(f"    Test:  blocks {min(test_blocks)}-{max(test_blocks)} "
          f"({len(test_blocks)} blocks, {len(test_blocks)/len(all_blocks)*100:.1f}%)")

    # ── NO-OVERLAP VERIFICATION ────────────────────────────────────────
    print(f"\n  {'='*60}")
    print("  NO-OVERLAP VERIFICATION")
    print(f"  {'='*60}")
    passed, issues = verify_no_overlap(train_blocks, val_blocks, test_blocks)
    if passed:
        print("  [PASS] No time_block_id appears in more than one split.")
        print(f"         Train blocks: {set(train_blocks) & set(val_blocks) | set(train_blocks) & set(test_blocks)} "
              f"overlap with Val/Test = EMPTY SET")
    else:
        print("  [FAIL] Overlap detected!")
        for issue in issues:
            print(f"    {issue}")
        return

    # Additional verification: check actual row assignments
    train_block_set = set(train_blocks)
    val_block_set = set(val_blocks)
    test_block_set = set(test_blocks)

    # Verify contiguity (train < val < test)
    if max(train_blocks) < min(val_blocks) and max(val_blocks) < min(test_blocks):
        print("  [PASS] Splits are temporally contiguous (train < val < test).")
    else:
        print("  [WARN] Splits are not temporally contiguous.")

    # ── Save split assignment ──────────────────────────────────────────
    csv_path, pkl_path = save_split_assignment(
        block_assignment, train_blocks, val_blocks, test_blocks
    )
    print(f"\n  Split assignment saved:")
    print(f"    CSV: {csv_path}")
    print(f"    PKL: {pkl_path}")

    # ── Split and scale DWT features ───────────────────────────────────
    print(f"\n  {'='*60}")
    print("  SCALING: DWT FEATURES")
    print(f"  {'='*60}")
    dwt_split, dwt_scaler, dwt_scaler_path = split_and_scale(
        dwt_rows, dwt_feature_cols, block_assignment, "dwt"
    )
    print(f"  Scaler fit on training data ({len(dwt_split['train']['rows']):,} rows)")
    print(f"  Scaler saved: {dwt_scaler_path}")
    save_split_data(dwt_split, dwt_feature_cols, "dwt_scaled")

    # Show scaler parameters
    print(f"\n  Scaler parameters (mean / std) for first 5 features:")
    for i in range(min(5, len(dwt_feature_cols))):
        print(f"    {dwt_feature_cols[i]:<28} mean={dwt_scaler.mean_[i]:>12.6f}  "
              f"std={dwt_scaler.std_[i]:>12.6f}")

    # ── Split and scale Raw features ───────────────────────────────────
    print(f"\n  {'='*60}")
    print("  SCALING: RAW STATISTICAL FEATURES")
    print(f"  {'='*60}")
    raw_split, raw_scaler, raw_scaler_path = split_and_scale(
        raw_rows, raw_feature_cols, block_assignment, "raw_stat"
    )
    print(f"  Scaler fit on training data ({len(raw_split['train']['rows']):,} rows)")
    print(f"  Scaler saved: {raw_scaler_path}")
    save_split_data(raw_split, raw_feature_cols, "raw_scaled")

    # Show scaler parameters
    print(f"\n  Scaler parameters (mean / std) for first 5 features:")
    for i in range(min(5, len(raw_feature_cols))):
        print(f"    {raw_feature_cols[i]:<28} mean={raw_scaler.mean_[i]:>12.6f}  "
              f"std={raw_scaler.std_[i]:>12.6f}")

    # ══════════════════════════════════════════════════════════════════
    # PER-DEVICE, PER-SPLIT DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════
    print(f"\n\n{'='*100}")
    print("  PER-DEVICE WINDOW DISTRIBUTION ACROSS SPLITS")
    print(f"{'='*100}")

    # Count windows per device per split
    device_split_counts = {}
    for split_name in ["train", "val", "test"]:
        for row in dwt_split[split_name]["rows"]:
            device = row["device_label"]
            key = (device, split_name)
            device_split_counts[key] = device_split_counts.get(key, 0) + 1

    print(f"\n  {'Device':<28} {'Train':>10} {'Val':>10} {'Test':>10} "
          f"{'Total':>10} {'Train%':>8} {'Val%':>8} {'Test%':>8}")
    print(f"  {'-'*96}")

    tplink_warning = False
    for device in DEVICE_SUBSET:
        train_n = device_split_counts.get((device, "train"), 0)
        val_n = device_split_counts.get((device, "val"), 0)
        test_n = device_split_counts.get((device, "test"), 0)
        total_n = train_n + val_n + test_n
        t_pct = train_n / total_n * 100 if total_n > 0 else 0
        v_pct = val_n / total_n * 100 if total_n > 0 else 0
        te_pct = test_n / total_n * 100 if total_n > 0 else 0

        flag = ""
        if device == "TPLinkSmartPlug":
            if any(n == 0 for n in [train_n, val_n, test_n]):
                flag = "  *** EMPTY SPLIT ***"
                tplink_warning = True
            elif any(n < 5 for n in [train_n, val_n, test_n]):
                flag = "  *** LOW COUNT ***"
                tplink_warning = True

        print(f"  {device:<28} {train_n:>10,} {val_n:>10,} {test_n:>10,} "
              f"{total_n:>10,} {t_pct:>7.1f}% {v_pct:>7.1f}% {te_pct:>7.1f}%{flag}")

    # Totals
    total_train = sum(device_split_counts.get((d, "train"), 0) for d in DEVICE_SUBSET)
    total_val = sum(device_split_counts.get((d, "val"), 0) for d in DEVICE_SUBSET)
    total_test = sum(device_split_counts.get((d, "test"), 0) for d in DEVICE_SUBSET)
    total_all = total_train + total_val + total_test
    print(f"  {'-'*96}")
    print(f"  {'TOTAL':<28} {total_train:>10,} {total_val:>10,} {total_test:>10,} "
          f"{total_all:>10,} {total_train/total_all*100:>7.1f}% "
          f"{total_val/total_all*100:>7.1f}% {total_test/total_all*100:>7.1f}%")
    print(f"{'='*100}")

    # ── TPLinkSmartPlug specific check ─────────────────────────────────
    print(f"\n  TPLINK SMART PLUG CHECK:")
    tp_train = device_split_counts.get(("TPLinkSmartPlug", "train"), 0)
    tp_val = device_split_counts.get(("TPLinkSmartPlug", "val"), 0)
    tp_test = device_split_counts.get(("TPLinkSmartPlug", "test"), 0)
    print(f"    Train: {tp_train}, Val: {tp_val}, Test: {tp_test}")
    if all(n > 0 for n in [tp_train, tp_val, tp_test]):
        print(f"    [PASS] Non-zero count in all three splits.")
    else:
        print(f"    [FAIL] One or more splits have zero TPLinkSmartPlug windows!")

    if tplink_warning:
        print(f"\n  WARNING: TPLinkSmartPlug has very few windows in some splits.")
        print(f"  This may affect classifier performance for this device class.")

    # ── Verify scaled data has zero mean and unit variance on train ────
    print(f"\n  SCALING VERIFICATION (train split should have ~mean=0, ~std=1):")
    train_X_dwt = dwt_split["train"]["X_scaled"]
    train_X_raw = raw_split["train"]["X_scaled"]

    dwt_train_mean = np.mean(train_X_dwt, axis=0)
    dwt_train_std = np.std(train_X_dwt, axis=0)
    raw_train_mean = np.mean(train_X_raw, axis=0)
    raw_train_std = np.std(train_X_raw, axis=0)

    print(f"    DWT train mean range:  [{dwt_train_mean.min():.6f}, {dwt_train_mean.max():.6f}]")
    print(f"    DWT train std range:   [{dwt_train_std.min():.6f}, {dwt_train_std.max():.6f}]")
    print(f"    Raw train mean range:  [{raw_train_mean.min():.6f}, {raw_train_mean.max():.6f}]")
    print(f"    Raw train std range:   [{raw_train_std.min():.6f}, {raw_train_std.max():.6f}]")

    if (abs(dwt_train_mean).max() < 1e-6 and abs(dwt_train_std - 1.0).max() < 1e-6 and
            abs(raw_train_mean).max() < 1e-6 and abs(raw_train_std - 1.0).max() < 1e-6):
        print(f"    [PASS] Train split has zero mean and unit variance for both feature sets.")
    else:
        print(f"    [WARN] Check scaling — values not exactly 0/1 (may be floating point).")

    # ── Output files listing ───────────────────────────────────────────
    print(f"\n  OUTPUT FILES:")
    for f in sorted(os.listdir(FEATURES_DIR)):
        if "scaled" in f:
            fpath = os.path.join(FEATURES_DIR, f)
            size = os.path.getsize(fpath)
            print(f"    {f} ({size / 1024:.1f} KB)")
    for f in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(fpath)
        print(f"    {f} ({size / 1024:.1f} KB)")

    print(f"\n  Step 6 complete. Awaiting review before proceeding to Step 7.")


if __name__ == "__main__":
    main()
