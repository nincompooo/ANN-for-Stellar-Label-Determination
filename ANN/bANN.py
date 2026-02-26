# run_ann.py

import os
import numpy as np
import pandas as pd
import torch
import joblib
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from bANN_utils import (
    label_names,
    build_wavelength_mask,
    apply_additive_noise,
    StellarDataset,
    StellarANN,
    predict_stellar_params_from_spectrum,
    plot_pred_vs_true,
    plot_mask_check
)

# ========== PATHS ==========
base_dir = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN"
output_dir = os.path.join(base_dir, "results")
os.makedirs(output_dir, exist_ok=True)

csv_path = "clean_stellar_dataset.csv"
wavelength_file = "Koi1422_HET.txt"

def train_and_evaluate():

    # LOAD DATA
    df = pd.read_csv(csv_path)
    flux_cols = [c for c in df.columns if c.startswith("flux_")]

    R_SUN = 6.957e10
    df["p_radius"] /= R_SUN
    df["s_radius"] /= R_SUN

    wavelength = np.loadtxt(wavelength_file)[:, 0]
    mask = build_wavelength_mask(wavelength)

    X_full = df[flux_cols].values
    X = X_full[:, mask]
    y = df[label_names].values

    X = X / np.median(X, axis=1, keepdims=True)
    X = apply_additive_noise(X, snr=30)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(StellarDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(StellarDataset(X_test, y_test), batch_size=128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StellarANN(X_train.shape[1], len(label_names)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    patience = 15
    counter = 0

    for epoch in range(20):

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
            if counter >= patience:
                print("Early stopping triggered")
                break

    # Load best
    model.load_state_dict(torch.load(os.path.join(base_dir, "stellar_ann_model.pt")))
    model.eval()

    x_scaler = joblib.load(os.path.join(base_dir, "x_scaler.save"))
    y_scaler = joblib.load(os.path.join(base_dir, "y_scaler.save"))

    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    print("\nMean Absolute Error per parameter:")
    for name, mae in zip(label_names, abs_errors.mean(axis=0)):
        print(f"{name:10s}: {mae:.4f}")

    rel_errors = abs_errors / np.abs(y_true)

    print("\nMedian Relative Errors per parameter:")
    for name, re in zip(label_names, np.median(rel_errors, axis=0)):
        print(f"{name:10s}: {re*100:.2f}%")

    plot_pred_vs_true(y_true, y_pred, label_names, output_dir)
    plot_mask_check(df, wavelength, build_wavelength_mask, output_dir)


if __name__ == "__main__":
    train_and_evaluate()