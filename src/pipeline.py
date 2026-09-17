"""
Step 9 — Final Production Pipeline (Open-Set Routing)
======================================================
1. Classifier: Train on 4 known devices (exclude TPLinkSmartPlug).
2. VAE: Train on 4 known devices (latent_dim=16), threshold on val split.
3. Routing: Test split passed through VAE.
   - If recon_error > threshold -> BLOCK
   - Else -> VERIFY (predict label with Classifier)
4. Report routing decision breakdown.
5. Report full known-set accuracy to check for VAE selection bias.
6. Save models, config, and results to timestamped folder.
"""

import os
import csv
import json
import datetime
import numpy as np
import yaml
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "phase1_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SEED = config["dataset"]["random_seed"]
THRESHOLD_PERCENTILE = config["vae"]["threshold_percentile"]
LATENT_DIM = 16
HELD_OUT_DEVICE = "TPLinkSmartPlug"

# Classifier params
N_ESTIMATORS = config["classifier"]["n_estimators"]
MAX_DEPTH = config["classifier"]["max_depth"]
MIN_SAMPLES_LEAF = config["classifier"]["min_samples_leaf"]

FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_DIR, f"step9_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_split_dict(split_name):
    """Load DWT features, group by device_label."""
    filepath = os.path.join(FEATURES_DIR, f"dwt_scaled_{split_name}.csv")
    data = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        feature_cols = [c for c in reader.fieldnames
                        if c not in ("device_label", "window_id", "time_block_id", "split")]
        for row in reader:
            label = row["device_label"]
            feats = [float(row[c]) for c in feature_cols]
            if label not in data:
                data[label] = []
            data[label].append(feats)
    return {k: np.array(v) for k, v in data.items()}, feature_cols

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        hidden_dim = max(input_dim, latent_dim)
        
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = torch.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h = torch.relu(self.dec_fc1(z))
        return self.dec_out(h)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def get_reconstruction_errors(model, X_tensor, batch_size=128):
    model.eval()
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0]
            recon_x, _, _ = model(x)
            err = torch.mean((recon_x - x) ** 2, dim=1).numpy()
            errors.extend(err)
    return np.array(errors)

def main():
    print("=" * 80)
    print("  STEP 9: FINAL PRODUCTION PIPELINE (ROUTING ENGINE)")
    print(f"  Known Devices (4): AmazonEcho, Belkin, Philips, Samsung")
    print(f"  Unknown Device (1): {HELD_OUT_DEVICE}")
    print(f"  Results saved to: {RUN_DIR}")
    print("=" * 80)

    # ── 1. Load Data ───────────────────────────────────────────────────
    train_dict, feat_cols = load_split_dict("train")
    val_dict, _ = load_split_dict("val")
    test_dict, _ = load_split_dict("test")
    input_dim = len(feat_cols)
    
    known_devices = [d for d in train_dict.keys() if d != HELD_OUT_DEVICE]
    
    # ── 2. Train VAE ───────────────────────────────────────────────────
    print("\n  [VAE] Training on 4 known devices (Train Split)...")
    X_train_known = np.vstack([train_dict[d] for d in known_devices])
    train_tensor = torch.tensor(X_train_known, dtype=torch.float32)
    
    model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)
    
    model.train()
    epochs = 50
    for epoch in range(epochs):
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            
    # Threshold on Val
    X_val_known = np.vstack([val_dict[d] for d in known_devices if d in val_dict])
    val_tensor = torch.tensor(X_val_known, dtype=torch.float32)
    val_errors = get_reconstruction_errors(model, val_tensor)
    threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))
    print(f"  [VAE] Frozen Threshold (99th %ile of Val): {threshold:.4f}")

    # ── 3. Train Classifier ────────────────────────────────────────────
    print("\n  [Classifier] Training Random Forest on 4 known devices...")
    y_train_known = np.hstack([[d]*len(train_dict[d]) for d in known_devices])
    
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH if MAX_DEPTH is not None else None,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train_known, y_train_known)
    print(f"  [Classifier] Trained on {len(X_train_known)} samples.")

    # ── 4. Routing Engine (Test Split) & Full Accuracy ─────────────────
    print("\n" + "="*80)
    print("  ROUTING RESULTS (TEST SPLIT)")
    print("="*80)
    print(f"  {'Device':<28} | {'Total':<6} | {'BLOCK':<8} | {'VERIFY':<8} | {'Classifier Acc (on VERIFY)':<25}")
    print("-" * 80)
    
    total_known = 0
    total_known_block = 0
    total_known_verify = 0
    
    # Track overall full test set for known devices to measure selection bias
    full_y_true = []
    full_y_pred = []
    full_y_verify_true = []
    full_y_verify_pred = []
    
    for device in list(known_devices) + [HELD_OUT_DEVICE]:
        if device not in test_dict:
            continue
            
        X_test = test_dict[device]
        n_samples = len(X_test)
        
        # VAE gating
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        errors = get_reconstruction_errors(model, test_tensor)
        
        is_blocked = (errors > threshold)
        n_block = np.sum(is_blocked)
        n_verify = n_samples - n_block
        
        # Classifier full prediction (including BLOCKED)
        y_pred_all = clf.predict(X_test)
        
        if device in known_devices:
            full_y_true.extend([device] * n_samples)
            full_y_pred.extend(y_pred_all)
            full_y_verify_true.extend([device] * n_verify)
            full_y_verify_pred.extend(y_pred_all[~is_blocked])
        
        # Verified prediction accuracy
        if n_verify > 0:
            y_pred_verify = y_pred_all[~is_blocked]
            if device in known_devices:
                acc = accuracy_score([device]*n_verify, y_pred_verify)
                acc_str = f"{acc*100:.1f}% ({np.sum(y_pred_verify == device)}/{n_verify})"
            else:
                acc_str = "N/A (Unknown Device)"
        else:
            acc_str = "N/A (0 verified)"
            
        if device in known_devices:
            total_known += n_samples
            total_known_block += n_block
            total_known_verify += n_verify
            
        print(f"  {device:<28} | {n_samples:<6} | {n_block:<8} | {n_verify:<8} | {acc_str:<25}")

    print("-" * 80)
    print(f"  {'ALL KNOWN DEVICES':<28} | {total_known:<6} | {total_known_block:<8} | {total_known_verify:<8} | {total_known_block/total_known*100:.1f}% False Positive Rate")
    
    if HELD_OUT_DEVICE in test_dict:
        tp_rate = test_dict[HELD_OUT_DEVICE].shape[0]
        n_block_unknown = np.sum(get_reconstruction_errors(model, torch.tensor(test_dict[HELD_OUT_DEVICE], dtype=torch.float32)) > threshold)
        print(f"  {'UNKNOWN DEVICE':<28} | {tp_rate:<6} | {n_block_unknown:<8} | {tp_rate-n_block_unknown:<8} | {n_block_unknown/tp_rate*100:.1f}% True Positive Rate")
    print("=" * 80)

    # 5. Selection Bias Check
    full_accuracy = accuracy_score(full_y_true, full_y_pred)
    verify_accuracy = accuracy_score(full_y_verify_true, full_y_verify_pred)
    print("\n" + "="*80)
    print("  SELECTION BIAS CHECK (KNOWN DEVICES TEST SET)")
    print("="*80)
    print(f"  Classifier Accuracy on FULL known test set (1058 samples): {full_accuracy*100:.2f}%")
    print(f"  Classifier Accuracy on VERIFY subset ({total_known_verify} samples):     {verify_accuracy*100:.2f}%")
    print(f"  Delta (VERIFY - FULL):                                    {(verify_accuracy - full_accuracy)*100:+.2f} pp")

    # 6. Save Artifacts
    torch.save(model.state_dict(), os.path.join(RUN_DIR, "vae_model.pt"))
    with open(os.path.join(RUN_DIR, "rf_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(RUN_DIR, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    summary = {
        "timestamp": TIMESTAMP,
        "vae_threshold": threshold,
        "total_known": total_known,
        "total_known_blocked": int(total_known_block),
        "total_known_verified": int(total_known_verify),
        "full_accuracy": full_accuracy,
        "verify_accuracy": verify_accuracy,
    }
    with open(os.path.join(RUN_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Pipeline components and results saved to {RUN_DIR}")
    print("  Phase 1 Complete.")

if __name__ == "__main__":
    main()
