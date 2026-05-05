# ==========================
# bANN_utils.py
# ==========================

import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

# ==========================
# LABELS
# ==========================

label_names = [
    "p_teff",
    "s_teff",
    "p_logg",
    "s_logg",
    "p_radius",
    "s_radius"
]

# ==========================
# CONSTANTS
# ==========================

EXPECTED_FEATURES = 1947

# ==========================
# MASKING
# ==========================

def build_wavelength_mask(wl):

    return (
        (wl <= 0.6860)
        |
        ((wl >= 0.6880) & (wl <= 0.7600))
        |
        ((wl >= 0.7660) & (wl <= 0.8210))
        |
        (wl >= 0.8240)
    )

def enforce_masking(X, wavelength):

    mask = build_wavelength_mask(
        wavelength
    )

    expected_full = len(mask)

    expected_masked = np.sum(mask)

    # --------------------------------
    # already masked
    # --------------------------------

    if X.shape[1] == expected_masked:

        return X, mask

    # --------------------------------
    # full spectrum
    # --------------------------------

    if X.shape[1] == expected_full:

        X_masked = X[:, mask]

        return X_masked, mask

    # --------------------------------
    # invalid
    # --------------------------------

    raise ValueError(
        f"[MASK ERROR] Got "
        f"{X.shape[1]} features.\n"
        f"Expected:\n"
        f"full={expected_full}\n"
        f"masked={expected_masked}"
    )

def preprocess_spectrum_matrix(
    X,
    wavelength
):

    # --------------------------------
    # FORCE CONSISTENT MASKING
    # --------------------------------

    X, mask = enforce_masking(
        X,
        wavelength
    )

    # --------------------------------
    # NORMALIZE AFTER MASKING
    # --------------------------------

    median_flux = np.median(
        X,
        axis=1,
        keepdims=True
    )

    median_flux[
        median_flux == 0
    ] = 1.0

    X = X / median_flux

    return X, mask

# ==========================
# NOISE
# ==========================

def apply_additive_noise(
    X,
    snr=30
):

    sigma = 1.0 / snr

    noise = np.random.normal(
        loc=0.0,
        scale=sigma,
        size=X.shape
    )

    return X + noise

# ==========================
# DATASET
# ==========================

class StellarDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        weights
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

        self.weights = torch.tensor(
            weights,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx],
            self.weights[idx]
        )

# ==========================
# MODEL
# ==========================

class StellarANN(nn.Module):

    def __init__(
        self,
        n_input,
        n_output
    ):

        super().__init__()

        # ===================================
        # BIGGER NETWORK
        # ===================================
        #
        # 1947 -> 512 -> 256 -> 128 -> 6
        #
        # NO DROPOUT
        #
        # ===================================

        self.net = nn.Sequential(

            nn.Linear(n_input, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, n_output)
        )

    def forward(self, x):

        return self.net(x)

# ============================================================
# KEEP ALL YOUR EXISTING PLOTTING / KOI FUNCTIONS BELOW
# ============================================================

#
# IMPORTANT:
#
# DO NOT change your prediction pipeline anymore.
#
# You now have:
#
# ✔ consistent masking
# ✔ consistent normalization
# ✔ larger ANN capacity
# ✔ no dropout
# ✔ weight decay
# ✔ early stopping
# ✔ weighted low-luminosity training
#
# This is now MUCH better aligned with
# spectroscopy regression problems.
#
# ============================================================
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
        plt.savefig(os.path.join(output_dir, f"pred vs true {name}.png"))
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
    plt.savefig(os.path.join(output_dir, "masking check.png"))
    plt.close()

def plot_mse(output_dir, train_losses, val_losses):
    os.makedirs(output_dir, exist_ok=True)

    # Validation loss only
    plt.figure(figsize=(7, 5))
    plt.plot(val_losses, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Validation Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_loss_curve.png"))
    plt.close()

    # Train vs validation
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_vs_validation_loss.png"))
    plt.close()

    # Smoothed validation curve
    def smooth(values, window=5):
        return [
            np.mean(values[max(0, i - window): i + 1])
            for i in range(len(values))
        ]

    smoothed_val = smooth(val_losses)

    plt.figure(figsize=(7, 5))
    plt.plot(val_losses, alpha=0.3, label="Raw")
    plt.plot(smoothed_val, linewidth=2, label="Smoothed")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Smoothed Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "smoothed_validation_loss.png"))
    plt.close()

def predict_stellar_params_from_spectrum(
    filename,
    model,
    x_scaler,
    y_scaler,
    device="cpu"
):

    data = np.loadtxt(filename)

    wl = data[:, 0]

    flux = data[:, 1].reshape(1, -1)

    # --------------------------------
    # EXACT SAME PIPELINE AS TRAINING
    # --------------------------------

    flux, mask = preprocess_spectrum_matrix(
        flux,
        wl
    )

    # DO NOT ADD NOISE DURING INFERENCE

    # --------------------------------
    # SCALE
    # --------------------------------

    if flux.shape[1] != x_scaler.n_features_in_:
        raise ValueError(
            f"[SCALER ERROR] "
            f"Input has {flux.shape[1]} features "
            f"but scaler expects "
            f"{x_scaler.n_features_in_}"
        )

    flux_scaled = x_scaler.transform(flux)

    # --------------------------------
    # PREDICT
    # --------------------------------

    with torch.no_grad():

        pred_scaled = model(
            torch.tensor(
                flux_scaled,
                dtype=torch.float32
            ).to(device)
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


def plot_teff_with_luminosity_ratio(
    s_teff,
    s_teff_pred,
    lum_ratio, 
    save_path=None,
    show=True
):
    """
    Scatter plot of true vs predicted secondary Teff,
    color-coded by log10 luminosity ratio (clipped at max=1).

    Parameters
    ----------
    s_teff : array-like
        True secondary Teff values
    s_teff_pred : array-like
        Predicted secondary Teff values
    lum_ratio : array-like
        L_s / L_p values (in log)
    """

    plt.figure(figsize=(7, 6))

    # already be in log10 scale
    # log_lum_ratio = np.log10(lum_ratio)

    # Clip at max value of 1
    log_lum_ratio_clipped = np.clip(lum_ratio, None, 1)

    sc = plt.scatter(
        s_teff,
        s_teff_pred,
        c=log_lum_ratio_clipped,
        cmap="viridis",
        alpha=0.8,
        vmin=np.min(log_lum_ratio_clipped), 
        vmax=1
    )

    # 1:1 reference line
    min_val = np.min(s_teff)
    max_val = np.max(s_teff)

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        'k--'
    )

    plt.xlabel("True Secondary Teff [K]")
    plt.ylabel("Predicted Secondary Teff [K]")
    plt.title("Secondary Teff (Color-coded by log10 Luminosity Ratio)")

    cbar = plt.colorbar(sc)
    cbar.set_label("log10(L_s / L_p) (clipped at 1)")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

def evaluate_koi_predictions(
    base_dir,
    model,
    x_scaler,
    y_scaler,
    device,
    output_dir
):

    import numpy as np
    import os

    koi_dirs = [os.path.join(base_dir, f"Koi{i}") for i in range(1,4)]
    derived_file = os.path.join(base_dir, "Derived Star.tex")

    derived_dict = parse_derived_star_tex(derived_file)
    matched_files = collect_matching_spectra(koi_dirs, derived_dict)

    y_true = []
    y_pred = []

    for koi_number, spectrum_file in matched_files:

        try:
            pred_dict = predict_stellar_params_from_spectrum(
                spectrum_file,
                model,
                x_scaler,
                y_scaler,
                device
            )
        except Exception:
            continue

        true_vals = derived_dict[koi_number]

        true_vector = [
            true_vals[0],  # p_teff
            true_vals[3],  # s_teff
            true_vals[6],  # p_radius
            true_vals[9]   # s_radius
        ]

        pred_vector = [
            pred_dict["p_teff"],
            pred_dict["s_teff"],
            pred_dict["p_radius"],
            pred_dict["s_radius"]
        ]

        y_true.append(true_vector)
        y_pred.append(pred_vector)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # unpack
    p_teff, s_teff, p_rad, s_rad = y_true.T
    p_teff_pred, s_teff_pred, p_rad_pred, s_rad_pred = y_pred.T

    # luminosity ratio
    lum_ratio = (s_rad / p_rad)**2 * (s_teff / p_teff)**4

    # ===== PLOTS =====

    plot_param_with_luminosity_ratio(
        p_teff, p_teff_pred, lum_ratio,
        "Primary Teff",
        os.path.join(output_dir, "KOI_p_teff.png")
    )

    plot_param_with_luminosity_ratio(
        s_teff, s_teff_pred, lum_ratio,
        "Secondary Teff",
        os.path.join(output_dir, "KOI_s_teff.png")
    )

    plot_param_with_luminosity_ratio(
        p_rad, p_rad_pred, lum_ratio,
        "Primary Radius",
        os.path.join(output_dir, "KOI_p_radius.png")
    )

    plot_param_with_luminosity_ratio(
        s_rad, s_rad_pred, lum_ratio,
        "Secondary Radius",
        os.path.join(output_dir, "KOI_s_radius.png")
    )

# strictly helper functions for my evaluate_koi_predictions helper function

def parse_derived_star_tex(filename):

    derived_dict = {}

    with open(filename) as f:
        for line in f:

            line = line.strip()

            if not line or line.startswith("\\") or "sname" in line:
                continue

            line = line.replace("\\\\","")
            parts = [p.strip() for p in line.split("&")]

            koi_number = int(float(parts[0]))
            values = np.array(parts[1:],dtype=float)

            derived_dict[koi_number] = values

    return derived_dict

def extract_koi_number(filename):

    base = filename.split("_")[0]
    number = base.replace("koi","")

    return int(number)

def collect_matching_spectra(spec_dirs, derived_dict):

    matched_files = []

    for spec_dir in spec_dirs:

        for file in os.listdir(spec_dir):

            if not file.endswith(".txt"):
                continue

            koi_number = extract_koi_number(file)

            if koi_number in derived_dict:
                matched_files.append((koi_number, os.path.join(spec_dir,file)))

    return matched_files

def plot_param_with_luminosity_ratio(
    true_vals,
    pred_vals,
    lum_ratio,
    param_name,
    save_path
):
    """
    Generic plot: true vs predicted colored by luminosity ratio
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # clip to avoid extreme color scaling
    lum_ratio_clipped = np.clip(lum_ratio, 1e-3, 1)

    plt.figure(figsize=(7,6))

    sc = plt.scatter(
        true_vals,
        pred_vals,
        c=lum_ratio_clipped,
        cmap="viridis",
        alpha=0.8
    )

    # 1:1 line
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())

    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    plt.xlabel(f"True {param_name}")
    plt.ylabel(f"Predicted {param_name}")
    plt.title(f"{param_name} (Colored by Luminosity Ratio)")

    cbar = plt.colorbar(sc)
    cbar.set_label("L_s / L_p (clipped)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()