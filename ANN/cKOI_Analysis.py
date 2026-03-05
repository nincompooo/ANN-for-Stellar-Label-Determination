import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

from bANN_utils import (
    StellarANN,
    label_names,
    predict_stellar_params_from_spectrum,
    plot_pred_vs_estimated
)

# ==========================
# PATHS
# ==========================

base_dir = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN"
koi1 = os.path.join(base_dir, "Koi1")
koi2 = os.path.join(base_dir, "Koi2")
koi3 = os.path.join(base_dir, "Koi3")

saved_model_path = os.path.join(base_dir, "stellar_ann_model.pt")
x_scaler_path = os.path.join(base_dir, "x_scaler.save")
y_scaler_path = os.path.join(base_dir, "y_scaler.save")

derived_file = "Derived Star.tex"
spec_dirs = [koi1, koi2, koi3]

comparison_labels = ["p_teff", "s_teff", "p_radius", "s_radius"]

device = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================
# LOAD MODEL
# ==========================

x_scaler = joblib.load(x_scaler_path)
y_scaler = joblib.load(y_scaler_path)

n_input = len(x_scaler.mean_)
n_output = len(label_names)

model = StellarANN(n_input, n_output).to(device)
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.eval()


# ==========================
# HELPER FUNCTIONS
# ==========================

def parse_derived_star_tex(filename):
    derived_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("\\") or "sname" in line:
                continue

            line = line.replace("\\\\", "")
            parts = [p.strip() for p in line.split("&")]

            koi_number = int(float(parts[0]))
            values = np.array(parts[1:], dtype=float)

            derived_dict[koi_number] = values

    return derived_dict


def extract_koi_number(filename):
    base = filename.split("_")[0]
    number = base.replace("koi", "")
    return int(number)


def collect_matching_spectra(spec_dirs, derived_dict):
    matched_files = []

    for spec_dir in spec_dirs:
        for file in os.listdir(spec_dir):

            if not file.endswith(".txt"):
                continue

            koi_number = extract_koi_number(file)

            if koi_number in derived_dict:
                full_path = os.path.join(spec_dir, file)
                matched_files.append((koi_number, full_path))

    return matched_files


# ==========================
# LOAD TRUE VALUES
# ==========================

derived_dict = parse_derived_star_tex(derived_file)

# Remove hot systems
filtered_dict = {}

for koi_number, values in derived_dict.items():

    p_teff = values[0]
    s_teff = values[3]

    if p_teff <= 6200 and s_teff <= 6200:
        filtered_dict[koi_number] = values

print(f"Removed {len(derived_dict) - len(filtered_dict)} hot stars (>6200 K)")
derived_dict = filtered_dict

matched_files = collect_matching_spectra(spec_dirs, derived_dict)
print(f"Matched {len(matched_files)} stars")


# ==========================
# PREDICTIONS
# ==========================

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

    # ---- Store primary mass (adjust index if needed) ----
    primary_mass = true_values[1]   # <-- CHANGE INDEX IF DIFFERENT
    primary_mass_list.append(primary_mass)

    y_true.append(true_vector)
    y_pred.append(pred_vector)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
primary_mass_array = np.array(primary_mass_list)


# ==========================
# STANDARD PLOT
# ==========================

plot_pred_vs_estimated(
    y_true=y_true,
    y_pred=y_pred,
    label_names=comparison_labels,
    output_dir="Prediction_Plots"
)


# ==========================
# LUMINOSITY RATIO
# ==========================

p_teff = y_true[:, 0]
s_teff = y_true[:, 1]
p_radius = y_true[:, 2]
s_radius = y_true[:, 3]

s_teff_pred = y_pred[:, 1]

lum_ratio = (s_radius / p_radius)**2 * (s_teff / p_teff)**4


# ==========================
# PLOT 1 — Colored by Luminosity Ratio
# ==========================

plt.figure(figsize=(7,6))

sc = plt.scatter(
    s_teff,
    s_teff_pred,
    c=lum_ratio,
    cmap="viridis",
    alpha=0.8
)

plt.plot([s_teff.min(), s_teff.max()],
         [s_teff.min(), s_teff.max()],
         'k--')

plt.xlabel("True Secondary Teff [K]")
plt.ylabel("Predicted Secondary Teff [K]")
plt.title("Secondary Teff (Color-coded by Luminosity Ratio)")

cbar = plt.colorbar(sc)
cbar.set_label("L_s / L_p")

plt.tight_layout()
plt.savefig("Prediction_Plots/s_teff_luminosity_colored.png")
plt.show()


# ==========================
# PLOT 2 — Colored by Primary Mass
# ==========================

plt.figure(figsize=(7,6))

sc = plt.scatter(
    s_teff,
    s_teff_pred,
    c=primary_mass_array,
    cmap="plasma",
    alpha=0.8
)

plt.plot([s_teff.min(), s_teff.max()],
         [s_teff.min(), s_teff.max()],
         'k--')

plt.xlabel("True Secondary Teff [K]")
plt.ylabel("Predicted Secondary Teff [K]")
plt.title("Secondary Teff (Color-coded by Primary Mass)")

cbar = plt.colorbar(sc)
cbar.set_label("Primary Mass [Msun]")

plt.tight_layout()
plt.savefig("Prediction_Plots/s_teff_primary_mass_colored.png")
plt.show()


# ==========================
# PLOT 3 — Residual vs Luminosity Ratio
# ==========================

residual = s_teff_pred - s_teff

plt.figure(figsize=(7,6))

plt.scatter(lum_ratio, residual, alpha=0.8)

plt.axhline(0, linestyle='--')
plt.xlabel("L_s / L_p")
plt.ylabel("Teff Residual (Pred - True)")
plt.title("Secondary Teff Residual vs Luminosity Ratio")

plt.tight_layout()
plt.savefig("Prediction_Plots/s_teff_residual_vs_luminosity.png")
plt.show()


print(f"Plotted {len(y_true)} stars")