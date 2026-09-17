"""
Step 8 — VAE Open-Set / Anomaly Scoring
======================================================
1. Train VAE on DWT training features, holding out BelkinWemoMotionSensor.
2. Select anomaly threshold from validation split (known devices only) using config percentile.
3. Evaluate anomaly detection on test split (known devices).
4. Evaluate anomaly detection on held-out device (BelkinWemoMotionSensor).
"""

import os
import csv
import json
import yaml
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "phase1_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SEED = config["dataset"]["random_seed"]
LATENT_DIM = config["vae"]["latent_dim"]
THRESHOLD_PERCENTILE = config["vae"]["threshold_percentile"]

HELD_OUT_DEVICE = "BelkinWemoMotionSensor"

FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_DIR, f"step8_{TIMESTAMP}")
os.makedirs(RUN_DIR, exist_ok=True)

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_split(split_name):
    """Load scaled DWT features, separate into known vs held-out."""
    filepath = os.path.join(FEATURES_DIR, f"dwt_scaled_{split_name}.csv")
    X_known = []
    X_heldout = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        feature_cols = [c for c in reader.fieldnames
                        if c not in ("device_label", "window_id", "time_block_id", "split")]
        for row in reader:
            feats = [float(row[c]) for c in feature_cols]
            if row["device_label"] == HELD_OUT_DEVICE:
                X_heldout.append(feats)
            else:
                X_known.append(feats)
                
    return np.array(X_known), np.array(X_heldout), feature_cols

# ── VAE Architecture ───────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        hidden_dim = max(input_dim, latent_dim)
        
        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
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
        return self.dec_out(h) # Linear output since input is StandardScaler scaled
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    # MSE for reconstruction (since inputs are standard-scaled, no sigmoid cross-entropy)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence
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
            # MSE per sample
            err = torch.mean((recon_x - x) ** 2, dim=1).numpy()
            errors.extend(err)
    return np.array(errors)

def plot_histogram(errors, threshold, title, filepath):
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.title(title)
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def main():
    print("=" * 80)
    print("  STEP 8: VAE OPEN-SET / ANOMALY SCORING")
    print(f"  Latent dim = {LATENT_DIM}, Threshold % = {THRESHOLD_PERCENTILE}")
    print(f"  Held-out device (Unknown) = {HELD_OUT_DEVICE}")
    print("=" * 80)

    # 1. Load Data
    X_train_known, X_train_heldout, feature_cols = load_split("train")
    X_val_known, X_val_heldout, _ = load_split("val")
    X_test_known, X_test_heldout, _ = load_split("test")
    
    input_dim = len(feature_cols)
    print(f"  [Data] Train (Known): {X_train_known.shape}")
    print(f"  [Data] Val (Known):   {X_val_known.shape}")
    print(f"  [Data] Test (Known):  {X_test_known.shape}")
    
    # Combine held-out data to evaluate the unknown class thoroughly
    X_all_heldout = np.vstack([X_train_heldout, X_val_heldout, X_test_heldout])
    print(f"  [Data] Held-out (All): {X_all_heldout.shape}")
    
    # 2. Train VAE
    print("\n  [Training] VAE on Train (Known)...")
    model = VAE(input_dim=input_dim, latent_dim=LATENT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_tensor = torch.tensor(X_train_known, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)
    
    epochs = 50
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss_function(recon_x, x, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
    print(f"    -> Finished {epochs} epochs. Final loss: {train_loss / len(train_tensor):.4f}")
    
    # 3. Select Threshold (Validation Derived)
    print("\n  [Validation-Derived] Selecting Anomaly Threshold...")
    val_tensor = torch.tensor(X_val_known, dtype=torch.float32)
    val_errors = get_reconstruction_errors(model, val_tensor)
    
    threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)
    print(f"    -> {THRESHOLD_PERCENTILE}th percentile of Validation errors: {threshold:.4f}")
    
    # 4. Evaluate on Test (Test Derived)
    print("\n  [Test-Derived] Evaluating Known Devices (False Positives)...")
    test_tensor = torch.tensor(X_test_known, dtype=torch.float32)
    test_errors = get_reconstruction_errors(model, test_tensor)
    
    test_flagged = np.sum(test_errors > threshold)
    test_total = len(test_errors)
    test_fp_rate = test_flagged / test_total
    print(f"    -> Flagged as Anomaly: {test_flagged} / {test_total} ({test_fp_rate*100:.2f}%)")
    
    # 5. Evaluate on Held-Out (Test Derived / Evaluation Derived)
    print("\n  [Test-Derived] Evaluating Held-Out Device (True Positives)...")
    heldout_tensor = torch.tensor(X_all_heldout, dtype=torch.float32)
    heldout_errors = get_reconstruction_errors(model, heldout_tensor)
    
    heldout_flagged = np.sum(heldout_errors > threshold)
    heldout_total = len(heldout_errors)
    heldout_tp_rate = heldout_flagged / heldout_total
    print(f"    -> Flagged as Anomaly: {heldout_flagged} / {heldout_total} ({heldout_tp_rate*100:.2f}%)")
    
    # 6. Save Artifacts
    plot_histogram(val_errors, threshold, "Validation Reconstruction Errors (Known)", 
                   os.path.join(RUN_DIR, "val_errors.png"))
    plot_histogram(test_errors, threshold, "Test Reconstruction Errors (Known)", 
                   os.path.join(RUN_DIR, "test_errors.png"))
    plot_histogram(heldout_errors, threshold, f"Held-Out Errors ({HELD_OUT_DEVICE})", 
                   os.path.join(RUN_DIR, "heldout_errors.png"))
                   
    torch.save(model.state_dict(), os.path.join(RUN_DIR, "vae_model.pt"))
    
    summary = {
        "timestamp": TIMESTAMP,
        "vae_config": {
            "latent_dim": LATENT_DIM,
            "threshold_percentile": THRESHOLD_PERCENTILE
        },
        "validation_derived": {
            "threshold": float(threshold),
            "val_errors_mean": float(np.mean(val_errors)),
            "val_errors_std": float(np.std(val_errors))
        },
        "test_derived": {
            "known_total": int(test_total),
            "known_flagged": int(test_flagged),
            "known_fp_rate": float(test_fp_rate),
            "heldout_total": int(heldout_total),
            "heldout_flagged": int(heldout_flagged),
            "heldout_tp_rate": float(heldout_tp_rate)
        }
    }
    with open(os.path.join(RUN_DIR, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\n  Results saved to: {RUN_DIR}")
    print("  Step 8 complete.")

if __name__ == "__main__":
    main()
