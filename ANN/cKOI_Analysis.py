import os
import torch
import joblib
import numpy as np

from bANN_utils import StellarANN, label_names, predict_stellar_params_from_spectrum, plot_pred_vs_estimated

eval_labels = ["p_teff", "s_teff", "p_radius", "s_radius"]

base_dir = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN"
koi1 = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN/Koi1"
koi2 = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN/Koi2"
koi3 = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN/Koi3"

saved_model_path = os.path.join(base_dir, "stellar_ann_model.pt")
x_scaler_path = os.path.join(base_dir, "x_scaler.save")
y_scaler_path = os.path.join(base_dir, "y_scaler.save")

device = "cuda" if torch.cuda.is_available() else "cpu"

x_scaler = joblib.load(x_scaler_path)
y_scaler = joblib.load(y_scaler_path)

n_input = len(x_scaler.mean_)      # must match training input size
n_output = len(label_names)

model = StellarANN(n_input, n_output).to(device)
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.eval()

comparison_labels = ["p_teff", "s_teff", "p_radius", "s_radius"]

# ===== PREDICTION =====
new_spectrum_file = "Koi1422_HET.txt"  

predicted_labels = predict_stellar_params_from_spectrum(
    filename=new_spectrum_file,
    model=model,
    x_scaler=x_scaler,
    y_scaler=y_scaler,
    device=device
)

print("\nPredicted stellar parameters:")
for k, v in predicted_labels.items():
    print(f"{k:10s}: {v:.4f}")

def save_prediction_tex(star_name, predicted_labels, true_values, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f"{star_name}_predicted.tex")

    with open(outfile, "w") as f:
        f.write("Parameter    Predicted_Value    True_Value\n")
        f.write("------------------------------------------------\n")

        # Order must match comparison_labels
        param_order = ["p_teff", "s_teff", "p_radius", "s_radius"]

        for i, param in enumerate(param_order):
            pred_val = predicted_labels[param]
            true_val = true_values[i]
            f.write(f"{param:<12} {pred_val:<18.6f} {true_val:<18.6f}\n")

def parse_derived_star_tex(filename):
    derived_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("\\") or "sname" in line:
                continue

            line = line.replace("\\\\", "")
            parts = [p.strip() for p in line.split("&")]

            # Convert sname like "42.0" → 42
            koi_number = int(float(parts[0]))

            values = np.array(parts[1:], dtype=float)

            derived_dict[koi_number] = values

    return derived_dict

def extract_koi_number(filename):
    # koi0005_HET.txt → 5
    base = filename.split("_")[0]     # koi0005
    number = base.replace("koi", "")  # 0005
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

derived_file = "Derived Star.tex"
spec_dirs = [koi1, koi2, koi3]

derived_dict = parse_derived_star_tex(derived_file)
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

y_true = []
y_pred = []


label_names = comparison_labels

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

    # ----- Get true values FIRST -----
    true_values = derived_dict[koi_number]

    true_vector = [
        true_values[0],  # p_teff
        true_values[3],  # s_teff
        true_values[6],  # p_radius
        true_values[9]   # s_radius
    ]

    # ----- Save file -----
    save_prediction_tex(
        star_name=f"koi{koi_number:04d}",
        predicted_labels=predicted_labels,
        true_values=true_vector,
        output_dir="Predicted_Tables"
    )

    # ----- Store predictions -----
    pred_vector = [
        predicted_labels["p_teff"],
        predicted_labels["s_teff"],
        predicted_labels["p_radius"],
        predicted_labels["s_radius"]
    ]

    y_pred.append(pred_vector)
    y_true.append(true_vector)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

plot_pred_vs_estimated(
    y_true=y_true,
    y_pred=y_pred,
    label_names=comparison_labels,
    output_dir="Prediction_Plots"
)