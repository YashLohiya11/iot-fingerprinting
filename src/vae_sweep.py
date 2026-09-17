"""
Step 8 (Sweep) — VAE Open-Set / Anomaly Scoring Ablation
======================================================
Sweep across latent_dim in {2, 4, 8, 16}.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

# ── Load config ────────────────────────────────────────────────────────
PROJ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CONFIG_PATH = os.path.join(PROJ_ROOT, "configs", "phase1_config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

SEED = config["dataset"]["random_seed"]
THRESHOLD_PERCENTILE = config["vae"]["threshold_percentile"]
HELD_OUT_DEVICE = "BelkinWemoMotionSensor"
FEATURES_DIR = os.path.join(PROJ_ROOT, "data", "features")

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
        return self.dec_out(h)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    # Standard beta=1 VAE loss
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
            # MSE per sample
            err = torch.mean((recon_x - x) ** 2, dim=1).numpy()
            errors.extend(err)
    return np.array(errors)

def main():
    X_train_known, X_train_heldout, feature_cols = load_split("train")
    X_val_known, X_val_heldout, _ = load_split("val")
    X_test_known, X_test_heldout, _ = load_split("test")
    
    input_dim = len(feature_cols)
    X_all_heldout = np.vstack([X_train_heldout, X_val_heldout, X_test_heldout])
    
    train_tensor = torch.tensor(X_train_known, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_known, dtype=torch.float32)
    test_tensor = torch.tensor(X_test_known, dtype=torch.float32)
    heldout_tensor = torch.tensor(X_all_heldout, dtype=torch.float32)
    
    latent_dims_to_test = [2, 4, 8, 16]
    epochs = 50
    
    results = []
    
    print("Starting Sweep...")
    
    for ld in latent_dims_to_test:
        torch.manual_seed(SEED)
        
        model = VAE(input_dim=input_dim, latent_dim=ld)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                x = batch[0]
                optimizer.zero_grad()
                recon_x, mu, logvar = model(x)
                loss = vae_loss_function(recon_x, x, mu, logvar)
                loss.backward()
                optimizer.step()
                
        # Val
        val_errors = get_reconstruction_errors(model, val_tensor)
        threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)
        avg_val_error = np.mean(val_errors)
        
        # Test (Known)
        test_errors = get_reconstruction_errors(model, test_tensor)
        test_flagged = np.sum(test_errors > threshold)
        test_fp_rate = test_flagged / len(test_errors)
        
        # Held-Out
        heldout_errors = get_reconstruction_errors(model, heldout_tensor)
        heldout_flagged = np.sum(heldout_errors > threshold)
        heldout_tp_rate = heldout_flagged / len(heldout_errors)
        
        results.append({
            "latent_dim": ld,
            "threshold": threshold,
            "avg_val_error": avg_val_error,
            "test_fp_rate": test_fp_rate,
            "heldout_tp_rate": heldout_tp_rate
        })
        
        print(f"Done ld={ld}")
        
    print("\n" + "="*80)
    print(f"{'Latent':<8} | {'Avg Val Err':<12} | {'Threshold':<10} | {'Test FP':<10} | {'Held-Out TP':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['latent_dim']:<8} | {r['avg_val_error']:<12.4f} | {r['threshold']:<10.4f} | {r['test_fp_rate']*100:>6.2f}%    | {r['heldout_tp_rate']*100:>8.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()
