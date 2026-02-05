import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# ========== PATHS ==========
base_dir = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN"
output_dir = os.path.join(base_dir, "results")
os.makedirs(output_dir, exist_ok=True)

csv_path = "noisy_stellar_dataset.csv"
wavelength_file = "Koi1422_HET.txt"  # for wavelength grid reference, optional

# ========== LABELS ==========
# We'll transform radius columns to log space for training
label_names = ["p_teff", "s_teff", "p_logg", "s_logg", "p_radius_log", "s_radius_log"]

# ========== 1. LOAD DATA ==========
df = pd.read_csv(csv_path)
flux_cols = [c for c in df.columns if c.startswith("flux_")]

epsilon = 1e-5
# Create log radius columns to stabilize training
df["p_radius_log"] = np.log(df["p_radius"] + epsilon)
df["s_radius_log"] = np.log(df["s_radius"] + epsilon)

def undo_log_radii(arr, label_names, epsilon):
    arr = arr.copy()
    pr = label_names.index("p_radius_log")
    sr = label_names.index("s_radius_log")
    arr[:, pr] = np.exp(arr[:, pr]) - epsilon
    arr[:, sr] = np.exp(arr[:, sr]) - epsilon
    return arr

# Load wavelength for info (not used in training)
wavelength = np.loadtxt(wavelength_file)[:, 0]
assert len(flux_cols) == len(wavelength), "Flux columns != wavelength bins"

# ========== 2. BUILD INPUTS & TARGETS ==========
X = df[flux_cols].values
y = df[label_names].values

# ========== 3. NORMALIZATION FUNCTION ==========
def normalize_synthetic(flux_primary, flux_secondary):
    """
    Normalize both primary and secondary flux by median of primary flux
    This keeps relative brightness ratios intact.
    """
    norm = np.median(flux_primary)
    return flux_primary / norm, flux_secondary / norm

# Normalize each combined flux row by its median (simulate primary star normalization)
X = X / np.median(X, axis=1, keepdims=True)

# ========== 4. SCALE DATA ==========
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# ========== 5. TRAIN/TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ========== 6. DATASET CLASS ==========
class StellarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(StellarDataset(X_train, y_train), batch_size=128, shuffle=True)
test_loader = DataLoader(StellarDataset(X_test, y_test), batch_size=128)

# ========== 7. MODEL ==========
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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = StellarANN(len(flux_cols), len(label_names)).to(device)

# ========== 8. TRAINING SETUP ==========
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ========== 9. TRAINING LOOP WITH EARLY STOPPING ==========
best_val = float("inf")
patience = 15
counter = 0
max_epochs = 20

for epoch in range(max_epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item()
    val_loss /= len(test_loader)

    print(f"Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), os.path.join(base_dir, "stellar_ann_model.pt"))
        joblib.dump(x_scaler, os.path.join(base_dir, "x_scaler.save"))
        joblib.dump(y_scaler, os.path.join(base_dir, "y_scaler.save"))
        print("New best model saved")
    else:
        counter += 1
        print(f"No improvement ({counter}/{patience})")
        if counter >= patience:
            print("Early stopping triggered")
            break

# ========== 10. LOAD BEST MODEL FOR PREDICTION ==========
model.load_state_dict(torch.load(os.path.join(base_dir, "stellar_ann_model.pt")))
model.eval()
x_scaler = joblib.load(os.path.join(base_dir, "x_scaler.save"))
y_scaler = joblib.load(os.path.join(base_dir, "y_scaler.save"))

# ========== 11. PREDICTIONS ON TEST SET ==========
with torch.no_grad():
    y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test)

# Optionally convert radius logs back to linear for metrics or keep log-space
y_pred_lin = undo_log_radii(y_pred, label_names, epsilon)
y_true_lin = undo_log_radii(y_true, label_names, epsilon)

# ========== 12. METRICS ==========
errors = y_pred - y_true
abs_errors = np.abs(errors)

print("\nMean Absolute Error per parameter (log-radius):")
for name, mae in zip(label_names, abs_errors.mean(axis=0)):
    print(f"{name:10s}: {mae:.4f}")

rel_errors = abs_errors / np.abs(y_true)
print("\nMedian Relative Errors (log-radius):")
for name, re in zip(label_names, np.median(rel_errors, axis=0)):
    print(f"{name:10s}: {re*100:.2f}%")

# If you want metrics in linear radius units, do this:
errors_lin = y_pred_lin - y_true_lin
abs_errors_lin = np.abs(errors_lin)

print("\nMean Absolute Error per parameter (linear radius):")
for name, mae in zip(label_names, abs_errors_lin.mean(axis=0)):
    print(f"{name:10s}: {mae:.4f}")

rel_errors_lin = abs_errors_lin / np.abs(y_true_lin)
print("\nMedian Relative Errors (linear radius):")
for name, re in zip(label_names, np.median(rel_errors_lin, axis=0)):
    print(f"{name:10s}: {re*100:.2f}%")

# ========== 13. SAVE PREDICTIONS TABLE ==========
pred_df = pd.DataFrame(y_pred, columns=[f"pred_{n}" for n in label_names])
true_df = pd.DataFrame(y_true, columns=[f"true_{n}" for n in label_names])
err_df  = pd.DataFrame(errors, columns=[f"err_{n}" for n in label_names])

results_df = pd.concat([true_df, pred_df, err_df], axis=1)
results_df.to_csv(os.path.join(base_dir, "test_predictions.csv"), index=False)

# ========== 14. PLOTTING ==========
for i, name in enumerate(label_names):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true[:, i], y_pred[:, i], s=5, alpha=0.5)
    plt.plot([y_true[:, i].min(), y_true[:, i].max()],
             [y_true[:, i].min(), y_true[:, i].max()], 'r--')
    unit = " (R☉)" if "radius" in name else ""
    plt.xlabel(f"True {name}{unit}")
    plt.ylabel(f"Predicted {name}{unit}")
    plt.title(f"Predicted vs True: {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pred_vs_true_{name}.png"))
    plt.close()

# ========== 15. INFERENCE ON NEW REAL SPECTRUM ==========
def predict_stellar_params_from_spectrum(filename, model, x_scaler, y_scaler, epsilon=1e-5):
    data = np.loadtxt(filename)
    flux = data[:, 1]

    flux = flux / np.median(flux)
    flux = flux.reshape(1, -1)

    flux_scaled = x_scaler.transform(flux)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(flux_scaled, dtype=torch.float32)).numpy()

    pred = y_scaler.inverse_transform(pred_scaled)[0]

    # Indexes for log radius
    p_radius_log_idx = label_names.index("p_radius_log")
    s_radius_log_idx = label_names.index("s_radius_log")

    print("Log p_radius (model output):", pred[p_radius_log_idx])
    print("Log s_radius (model output):", pred[s_radius_log_idx])

    # Exponentiate to linear radius
    p_radius_lin = np.exp(pred[p_radius_log_idx]) - epsilon
    s_radius_lin = np.exp(pred[s_radius_log_idx]) - epsilon

    # Prepare clean output dict with renamed keys
    clean_labels = ["p_teff", "s_teff", "p_logg", "s_logg", "p_radius", "s_radius"]

    pred_clean = pred.copy()
    pred_clean[p_radius_log_idx] = p_radius_lin
    pred_clean[s_radius_log_idx] = s_radius_lin

    result = dict(zip(clean_labels, pred_clean))

    return result

# Example usage:
new_spectrum_file = "Koi1422_HET.txt"
predicted_labels = predict_stellar_params_from_spectrum(new_spectrum_file, model, x_scaler, y_scaler)
print("\nPredicted stellar parameters for new spectrum:")
for k, v in predicted_labels.items():
    print(f"{k:10s}: {v:.4f}")
