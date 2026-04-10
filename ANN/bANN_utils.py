import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt







# ========== LABELS ==========
label_names = ["p_teff", "s_teff", "p_logg", "s_logg", "p_radius", "s_radius"]

def build_wavelength_mask(wl):
    return (
        (wl <= 0.6860) |
        ((wl >= 0.6880) & (wl <= 0.7600)) |
        ((wl >= 0.7660) & (wl <= 0.8210)) |
        (wl >= 0.8240)
    )

def apply_additive_noise(X, snr=30):
    sigma = 1.0 / snr
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise

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
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, n_output)
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

def plot_mse(output_dir, val_losses, train_losses, y_true, y_pred):
    #  plotting mse vs # of iterations for testing loss ---> like the only one we actually gaf about tbh
    plt.figure()
    plt.plot(val_losses, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Validation MSE Loss vs Epoch")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "★ Validation Loss Curve ★.png"))
    plt.close()


    # we're also plotting train + validation but i actually dont really gaf about this as much
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Train vs Validation Loss.png"))
    plt.close()

    def smooth_curve(values, window=5):
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window)
            smoothed.append(np.mean(values[start:i+1]))
        return smoothed
    
    smoothed_val = smooth_curve(val_losses)

    plt.figure()
    plt.plot(val_losses, alpha=0.3, label="Raw")
    plt.plot(smoothed_val, linewidth=2, label="Smoothed")
    plt.legend()
    plt.title("Smoothed Train vs Validation Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "SMOOTH Train vs Validation Loss.png"))
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
    output_dir, 
    dataset = "clean_stellar_dataset.csv"
):

    base_dir = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN"
    koi_dirs = [os.path.join(base_dir, f"Koi{i}") for i in range(1,4)]

    saved_model_path = os.path.join(base_dir, "stellar_ann_model.pt")
    x_scaler_path = os.path.join(base_dir, "x_scaler.save")
    y_scaler_path = os.path.join(base_dir, "y_scaler.save")

    derived_file = "Derived Star.tex"
    synthetic_csv = os.path.join(base_dir, dataset)
    wavelength_file = os.path.join(base_dir, "Koi1422_HET.txt")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    model = StellarANN(len(x_scaler.mean_), len(label_names)).to(device)
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model.eval()

    derived_dict = parse_derived_star_tex(derived_file)
    derived_dict = {k:v for k,v in derived_dict.items() if v[0] <= 6200 and v[3] <= 6200}
    matched_files = collect_matching_spectra(koi_dirs, derived_dict)

    # okay cool actual predictions now lol

    y_true = []
    y_pred = []
    primary_mass_list = []

    for koi_number, spectrum_file in matched_files:

        try:
            predicted_labels = predict_stellar_params_from_spectrum(
                filename=spectrum_file,
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                device=device
            )
        except ValueError:
            print(f"Skipping {spectrum_file} due to shape mismatch")
            continue

        true_values = derived_dict[koi_number]

        true_vector = [
            true_values[0],  # p_teff
            true_values[3],  # s_teff
            true_values[6],  # p_radius
            true_values[9]   # s_radius
        ]

        pred_vector = [
            predicted_labels["p_teff"],
            predicted_labels["s_teff"],
            predicted_labels["p_radius"],
            predicted_labels["s_radius"]
        ]

        primary_mass = true_values[1]
        primary_mass_list.append(primary_mass)

        y_true.append(true_vector)
        y_pred.append(pred_vector)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    primary_mass_array = np.array(primary_mass_list)

    p_teff = y_true[:,0]
    s_teff = y_true[:,1]
    p_radius = y_true[:,2]
    s_radius = y_true[:,3]
    s_teff_pred = y_pred[:,1]
    residual = s_teff_pred - s_teff
    log_lum_ratio = np.log10((s_radius/p_radius)**2 * (s_teff/p_teff)**4)


    df_syn = pd.read_csv(synthetic_csv)
    flux_cols = [c for c in df_syn.columns if c.startswith("flux_")]
    wavelength = np.loadtxt(wavelength_file)[:,0]
    mask = build_wavelength_mask(wavelength)
    X_syn = df_syn[flux_cols].values[:,mask]
    X_syn = X_syn / np.median(X_syn,axis=1,keepdims=True)
    X_syn_scaled = x_scaler.transform(X_syn)

    df_syn = pd.read_csv(synthetic_csv)
    flux_cols = [c for c in df_syn.columns if c.startswith("flux_")]

    wavelength = np.loadtxt(wavelength_file)[:,0]
    mask = build_wavelength_mask(wavelength)

    X_syn = df_syn[flux_cols].values[:,mask]
    X_syn = X_syn / np.median(X_syn,axis=1,keepdims=True)

    X_syn_scaled = x_scaler.transform(X_syn)

    with torch.no_grad():

        y_pred_syn_scaled = model(
            torch.tensor(X_syn_scaled,dtype=torch.float32).to(device)
        ).cpu().numpy()

    y_pred_syn = y_scaler.inverse_transform(y_pred_syn_scaled)
    y_true_syn = df_syn[label_names].values

    p_teff = y_true[:, 0]
    s_teff = y_true[:, 1]
    p_radius = y_true[:, 2]
    s_radius = y_true[:, 3]

    s_teff_pred = y_pred[:, 1]

    lum_ratio = (s_radius / p_radius)**2 * (s_teff / p_teff)**4



    print(len(s_teff), len(s_teff_pred), len(log_lum_ratio))

    lum_ratio_clipped = np.clip(lum_ratio, 1e-3, 1)

    
    plt.figure(figsize=(7,6))

    sc = plt.scatter(
        s_teff,
        s_teff_pred,
        c=lum_ratio_clipped,
        cmap="viridis",
        alpha=0.8
    )
 
    plt.plot([s_teff.min(), s_teff.max()],
            [s_teff.min(), s_teff.max()],
            'k--')

    plt.xlabel("True Secondary Teff [K]")
    plt.ylabel("Predicted Secondary Teff [K]")
    plt.title("KOI's Secondary Teff (Color-coded by Luminosity Ratio)")

    cbar = plt.colorbar(sc)
    cbar.set_label("L_s / L_p")

    plt.tight_layout()
    plt.savefig("filtered results/KOI's Secondary Teff (Color-coded by Luminosity Ratio).png")
    plt.show()


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
