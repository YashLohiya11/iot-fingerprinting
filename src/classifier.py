"""
Step 7 — Baseline Classifier + Raw-vs-DWT Comparison
======================================================
1. Train Random Forest on DWT features (train split), hyperparams from config
2. Train identical RF on raw-stat features for direct comparison
3. Tune only against validation split; touch test split exactly once at end
4. Report: accuracy, macro F1, per-device P/R/F1, confusion matrix
5. Produce the raw-vs-DWT comparison table
6. Flag TPLinkSmartPlug instability given 9-sample test set
7. Save models, config snapshot, and report to timestamped results/ subfolder
"""

import os
import csv
import pickle
import json
import datetime
import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "phase1_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SEED = config["dataset"]["random_seed"]
DEVICE_SUBSET = config["dataset"]["device_subset"]
N_ESTIMATORS = config["classifier"]["n_estimators"]
MAX_DEPTH = config["classifier"]["max_depth"]  # None = unlimited
MIN_SAMPLES_LEAF = config["classifier"]["min_samples_leaf"]

FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")

# Create timestamped results subfolder
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_DIR, f"step7_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)


def load_split(prefix, split_name):
    """Load a scaled feature split CSV, returning X (features) and y (labels)."""
    filepath = os.path.join(FEATURES_DIR, f"{prefix}_{split_name}.csv")
    X = []
    y = []
    meta = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        feature_cols = [c for c in reader.fieldnames
                        if c not in ("device_label", "window_id", "time_block_id", "split")]
        for row in reader:
            X.append([float(row[c]) for c in feature_cols])
            y.append(row["device_label"])
            meta.append({
                "window_id": row["window_id"],
                "time_block_id": row["time_block_id"],
            })
    return np.array(X), np.array(y), feature_cols, meta


def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, label):
    """Train RF and evaluate on val and test."""
    print(f"\n  Training Random Forest ({label})...")
    print(f"    n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
          f"min_samples_leaf={MIN_SAMPLES_LEAF}, seed={SEED}")
    print(f"    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH if MAX_DEPTH is not None else None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Validation metrics ─────────────────────────────────────────────
    y_val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1_macro = f1_score(y_val, y_val_pred, average="macro")

    print(f"\n  --- Validation Results ({label}) ---")
    print(f"  Accuracy:  {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"  Macro F1:  {val_f1_macro:.4f}")
    print(f"\n  Classification Report (Val):")
    print(classification_report(y_val, y_val_pred, digits=4, zero_division=0))

    val_cm = confusion_matrix(y_val, y_val_pred, labels=DEVICE_SUBSET)

    # ── Test metrics (TOUCH EXACTLY ONCE) ──────────────────────────────
    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_macro = f1_score(y_test, y_test_pred, average="macro")

    print(f"\n  --- Test Results ({label}) ---")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Macro F1:  {test_f1_macro:.4f}")
    print(f"\n  Classification Report (Test):")
    print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

    test_cm = confusion_matrix(y_test, y_test_pred, labels=DEVICE_SUBSET)

    # Per-class metrics for test
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, labels=DEVICE_SUBSET, zero_division=0
    )

    per_class = {}
    for i, device in enumerate(DEVICE_SUBSET):
        per_class[device] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return clf, {
        "val_accuracy": val_acc,
        "val_f1_macro": val_f1_macro,
        "val_cm": val_cm,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1_macro,
        "test_cm": test_cm,
        "per_class": per_class,
    }


def plot_confusion_matrix(cm, labels, title, filepath):
    """Save confusion matrix as a plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    short_labels = [l[:12] for l in labels]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=short_labels,
           yticklabels=short_labels,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {filepath}")


def print_comparison_table(dwt_results, raw_results):
    """Print the raw-vs-DWT comparison table."""
    print(f"\n\n{'='*100}")
    print("  RAW-vs-DWT FEATURE COMPARISON (Test Set)")
    print(f"{'='*100}")

    print(f"\n  {'Metric':<30} {'DWT Features':>18} {'Raw Stats':>18} {'Winner':>12}")
    print(f"  {'-'*80}")
    print(f"  {'Accuracy':<30} {dwt_results['test_accuracy']:>17.4f} "
          f"{raw_results['test_accuracy']:>17.4f} "
          f"{'DWT' if dwt_results['test_accuracy'] >= raw_results['test_accuracy'] else 'Raw':>12}")
    print(f"  {'Macro F1':<30} {dwt_results['test_f1_macro']:>17.4f} "
          f"{raw_results['test_f1_macro']:>17.4f} "
          f"{'DWT' if dwt_results['test_f1_macro'] >= raw_results['test_f1_macro'] else 'Raw':>12}")

    print(f"\n  Per-Device F1 (Test):")
    print(f"  {'Device':<28} {'DWT F1':>10} {'Raw F1':>10} {'Support':>10} {'Winner':>10} {'Note':>20}")
    print(f"  {'-'*92}")

    for device in DEVICE_SUBSET:
        dwt_f1 = dwt_results["per_class"][device]["f1"]
        raw_f1 = raw_results["per_class"][device]["f1"]
        support = dwt_results["per_class"][device]["support"]
        winner = "DWT" if dwt_f1 >= raw_f1 else "Raw"

        note = ""
        if device == "TPLinkSmartPlug":
            note = "* small test set"

        print(f"  {device:<28} {dwt_f1:>10.4f} {raw_f1:>10.4f} {support:>10} "
              f"{winner:>10} {note:>20}")

    print(f"{'='*100}")

    # TPLinkSmartPlug warning
    tp_support = dwt_results["per_class"]["TPLinkSmartPlug"]["support"]
    if tp_support < 20:
        print(f"\n  WARNING: TPLinkSmartPlug test set has only {tp_support} samples.")
        print(f"    Per-class metrics for this device are UNSTABLE and should not be")
        print(f"    interpreted with the same confidence as the other devices.")
        print(f"    A single misclassification changes F1 by ~{1.0/tp_support*100:.0f}%+ for this class.")


def main():
    print("=" * 80)
    print("  STEP 7: BASELINE CLASSIFIER + RAW-vs-DWT COMPARISON")
    print(f"  Config: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
          f"min_samples_leaf={MIN_SAMPLES_LEAF}")
    print(f"  Results dir: {RUN_DIR}")
    print("=" * 80)

    # ── Load all splits ────────────────────────────────────────────────
    print("\n  Loading scaled features...")

    dwt_X_train, dwt_y_train, dwt_cols, _ = load_split("dwt_scaled", "train")
    dwt_X_val, dwt_y_val, _, _ = load_split("dwt_scaled", "val")
    dwt_X_test, dwt_y_test, _, _ = load_split("dwt_scaled", "test")

    raw_X_train, raw_y_train, raw_cols, _ = load_split("raw_scaled", "train")
    raw_X_val, raw_y_val, _, _ = load_split("raw_scaled", "val")
    raw_X_test, raw_y_test, _, _ = load_split("raw_scaled", "test")

    print(f"  DWT features: {len(dwt_cols)} cols, "
          f"Train={dwt_X_train.shape[0]}, Val={dwt_X_val.shape[0]}, Test={dwt_X_test.shape[0]}")
    print(f"  Raw features: {len(raw_cols)} cols, "
          f"Train={raw_X_train.shape[0]}, Val={raw_X_val.shape[0]}, Test={raw_X_test.shape[0]}")

    # ── Train & Evaluate: DWT ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  MODEL A: Random Forest on DWT Features")
    print(f"{'='*70}")
    dwt_clf, dwt_results = train_and_evaluate(
        dwt_X_train, dwt_y_train,
        dwt_X_val, dwt_y_val,
        dwt_X_test, dwt_y_test,
        "DWT"
    )

    # ── Train & Evaluate: Raw ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  MODEL B: Random Forest on Raw Statistical Features")
    print(f"{'='*70}")
    raw_clf, raw_results = train_and_evaluate(
        raw_X_train, raw_y_train,
        raw_X_val, raw_y_val,
        raw_X_test, raw_y_test,
        "Raw Stats"
    )

    # ── Feature Importances ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  FEATURE IMPORTANCES (DWT model)")
    print(f"{'='*70}")
    importances = dwt_clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx[:10]):
        print(f"    {rank+1:>2}. {dwt_cols[idx]:<28} {importances[idx]:.4f}")

    print(f"\n  FEATURE IMPORTANCES (Raw Stats model)")
    importances_raw = raw_clf.feature_importances_
    sorted_idx_raw = np.argsort(importances_raw)[::-1]
    for rank, idx in enumerate(sorted_idx_raw[:10]):
        print(f"    {rank+1:>2}. {raw_cols[idx]:<28} {importances_raw[idx]:.4f}")

    # ── Comparison Table ───────────────────────────────────────────────
    print_comparison_table(dwt_results, raw_results)

    # ── Save Confusion Matrices ────────────────────────────────────────
    print(f"\n  Saving confusion matrices...")
    plot_confusion_matrix(
        dwt_results["test_cm"], DEVICE_SUBSET,
        "DWT Features — Test Confusion Matrix",
        os.path.join(RUN_DIR, "cm_dwt_test.png")
    )
    plot_confusion_matrix(
        raw_results["test_cm"], DEVICE_SUBSET,
        "Raw Stats — Test Confusion Matrix",
        os.path.join(RUN_DIR, "cm_raw_test.png")
    )
    plot_confusion_matrix(
        dwt_results["val_cm"], DEVICE_SUBSET,
        "DWT Features — Validation Confusion Matrix",
        os.path.join(RUN_DIR, "cm_dwt_val.png")
    )
    plot_confusion_matrix(
        raw_results["val_cm"], DEVICE_SUBSET,
        "Raw Stats — Validation Confusion Matrix",
        os.path.join(RUN_DIR, "cm_raw_val.png")
    )

    # ── Save Models ────────────────────────────────────────────────────
    dwt_model_path = os.path.join(RUN_DIR, "rf_dwt_model.pkl")
    raw_model_path = os.path.join(RUN_DIR, "rf_raw_model.pkl")
    with open(dwt_model_path, "wb") as f:
        pickle.dump(dwt_clf, f)
    with open(raw_model_path, "wb") as f:
        pickle.dump(raw_clf, f)
    print(f"\n  Models saved:")
    print(f"    DWT: {dwt_model_path}")
    print(f"    Raw: {raw_model_path}")

    # ── Save Config Snapshot ───────────────────────────────────────────
    config_snapshot_path = os.path.join(RUN_DIR, "config_snapshot.yaml")
    with open(config_snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Config snapshot: {config_snapshot_path}")

    # ── Save Results Summary ───────────────────────────────────────────
    summary = {
        "timestamp": TIMESTAMP,
        "dwt": {
            "val_accuracy": dwt_results["val_accuracy"],
            "val_f1_macro": dwt_results["val_f1_macro"],
            "test_accuracy": dwt_results["test_accuracy"],
            "test_f1_macro": dwt_results["test_f1_macro"],
            "per_class": dwt_results["per_class"],
        },
        "raw": {
            "val_accuracy": raw_results["val_accuracy"],
            "val_f1_macro": raw_results["val_f1_macro"],
            "test_accuracy": raw_results["test_accuracy"],
            "test_f1_macro": raw_results["test_f1_macro"],
            "per_class": raw_results["per_class"],
        },
        "config": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "seed": SEED,
        },
    }
    summary_path = os.path.join(RUN_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Results summary: {summary_path}")

    print(f"\n  Step 7 complete. Awaiting review before proceeding to Step 8.")


if __name__ == "__main__":
    main()
