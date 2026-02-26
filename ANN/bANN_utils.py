import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# ========== LABELS ==========
label_names = ["p_teff", "s_teff", "p_logg", "s_logg", "p_radius", "s_radius"]

# ========== WAVELENGTH MASK ==========
def build_wavelength_mask(wl):
    return (
        (wl <= 0.6860) |
        ((wl >= 0.6880) & (wl <= 0.7600)) |
        ((wl >= 0.7660) & (wl <= 0.8210)) |
        (wl >= 0.8240)
    )

# ========== NOISE ==========
def apply_additive_noise(X, snr=30):
    sigma = 1.0 / snr
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise

# ========== DATASET ==========
class StellarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========== MODEL ==========
class StellarANN(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_output)
        )

    def forward(self, x):
        return self.net(x)


def plot_pred_vs_true(y_true, y_pred, label_names, output_dir):

    units = {
        "teff": r" [K]",
        "logg": r" [dex]",
        "radius": r" [$R_{\odot}$]"
    }

    for i, name in enumerate(label_names):

        unit = ""
        for key in units:
            if key in name:
                unit = units[key]

        plt.figure(figsize=(6,6))
        plt.scatter(y_true[:, i], y_pred[:, i], s=8, alpha=0.5)

        plt.plot(
            [y_true[:, i].min(), y_true[:, i].max()],
            [y_true[:, i].min(), y_true[:, i].max()],
            'r--'
        )

        plt.xlabel(f"True {name}{unit}", fontsize=20)
        plt.ylabel(f"Predicted {name}{unit}", fontsize=20)
        plt.title(f"Predicted vs True: {name}", fontsize=18)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pred_vs_true_{name}.png"))
        plt.close()


def plot_mask_check(df, wavelength, build_wavelength_mask, output_dir):

    flux_full = df[[c for c in df.columns if c.startswith("flux_")]].iloc[0].values
    mask_full = build_wavelength_mask(wavelength)

    flux_masked = flux_full.copy()
    flux_masked[~mask_full] = np.nan

    plt.figure(figsize=(12,5))
    plt.plot(wavelength, flux_full, alpha=0.3, label="Original")
    plt.plot(wavelength, flux_masked, lw=1, label="Masked (with gaps)")

    plt.legend(fontsize=14)
    plt.xlabel("Wavelength (microns)", fontsize=20)
    plt.ylabel("Normalized Flux", fontsize=20)
    plt.title("Wavelength Mask Check (True Gaps)", fontsize=18)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "masking_check.png"))
    plt.close()

# ========== INFERENCE ==========
def predict_stellar_params_from_spectrum(
    filename,
    model,
    x_scaler,
    y_scaler,
    device="cpu"
):

    data = np.loadtxt(filename)
    wl = data[:, 0]
    flux = data[:, 1]

    mask = build_wavelength_mask(wl)
    flux = flux[mask]

    flux = flux / np.median(flux)
    flux = flux.reshape(1, -1)

    flux_scaled = x_scaler.transform(flux)

    with torch.no_grad():
        pred_scaled = model(
            torch.tensor(flux_scaled, dtype=torch.float32).to(device)
        ).cpu().numpy()

    pred = y_scaler.inverse_transform(pred_scaled)[0]

    return dict(zip(label_names, pred))

def plot_pred_vs_estimated(y_true, y_pred, label_names, output_dir):

    units = {
        "teff": " [K]",
        "logg": " [dex]",
        "radius": " [R_sun]"
    }

    os.makedirs(output_dir, exist_ok=True)

    for i, name in enumerate(label_names):

        unit = ""
        for key in units:
            if key in name:
                unit = units[key]

        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        # Symmetric limits
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())

        plt.figure(figsize=(6,6))
        plt.scatter(true_vals, pred_vals, s=10, alpha=0.6)

        # Perfect prediction line
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlabel(f"True {name}{unit}", fontsize=16)
        plt.ylabel(f"Predicted {name}{unit}", fontsize=16)
        plt.title(f"{name}: Predicted vs True", fontsize=16)

        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f"pred_vs_true_{name}.png"))
        plt.close()