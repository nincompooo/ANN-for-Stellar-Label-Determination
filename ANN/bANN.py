# ==========================
# bANN.py
# ==========================

import os
import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from bANN_utils import (
    label_names,
    apply_additive_noise,
    StellarDataset,
    StellarANN,
    plot_pred_vs_true,
    plot_mse,
    evaluate_koi_predictions,
    preprocess_spectrum_matrix,
)

EXPECTED_FEATURES = 1947

# ==========================
# PATHS
# ==========================

base_dir = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN"
output_dir = os.path.join(base_dir, "filtered results")
os.makedirs(output_dir, exist_ok=True)

csv_path = "clean_stellar_dataset_Tdiff1000.csv"

model_path = os.path.join(base_dir, "stellar_ann_model.pt")
x_scaler_path = os.path.join(base_dir, "x_scaler.save")
y_scaler_path = os.path.join(base_dir, "y_scaler.save")

# ==========================
# TRAIN
# ==========================

def train_and_evaluate():

    # ----------------------
    # LOAD DATA
    # ----------------------

    df = pd.read_csv(csv_path)

    flux_cols = [c for c in df.columns if c.startswith("flux_")]

    # Convert radii
    R_SUN = 6.957e10
    df["p_radius"] /= R_SUN
    df["s_radius"] /= R_SUN

    X_full = df[flux_cols].values
    y = df[label_names].values

    # ----------------------
    # LOAD WAVELENGTH GRID
    # ----------------------

    wavelength = np.loadtxt("Koi1422_HET.txt")[:, 0]

    # ----------------------
    # CONSISTENT PREPROCESSING
    # ----------------------
    # ALWAYS:
    # 1) enforce mask
    # 2) normalize AFTER masking
    # 3) add noise AFTER normalization
    #
    # this guarantees:
    # - train spectra are masked
    # - validation spectra are masked
    # - test spectra are masked
    # - KOI spectra are masked
    # - scalers only ever see masked spectra
    # ----------------------

    X, mask = preprocess_spectrum_matrix(X_full, wavelength)

    # additive noise AFTER masking + normalization
    X = apply_additive_noise(X, snr=30)

    print(f"\nMasked feature count: {X.shape[1]}")
    print(f"Expected masked count: {EXPECTED_FEATURES}")

    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"[FATAL] Incorrect masked feature count: "
            f"{X.shape[1]} != {EXPECTED_FEATURES}"
        )

    # ----------------------
    # SCALE
    # ----------------------

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # ----------------------
    # SPLIT
    # ----------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled,
        y_scaled,
        test_size=0.3,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42
    )

    # ----------------------
    # DATALOADERS
    # ----------------------

    train_loader = DataLoader(
        StellarDataset(X_train, y_train),
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        StellarDataset(X_val, y_val),
        batch_size=128
    )

    # ----------------------
    # DEVICE
    # ----------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------
    # MODEL
    # ----------------------

    model = StellarANN(
        X.shape[1],
        len(label_names)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-5
    )

    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    patience = 20
    counter = 0

    train_losses = []
    val_losses = []

    # ======================
    # TRAIN LOOP
    # ======================

    for epoch in range(100):

        # ------------------
        # TRAIN
        # ------------------

        model.train()

        train_loss = 0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            loss = criterion(pred, yb)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        train_losses.append(train_loss)

        # ------------------
        # VALIDATION
        # ------------------

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)

                loss = criterion(pred, yb)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:3d} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f}"
        )

        # ------------------
        # EARLY STOPPING
        # ------------------

        if val_loss < best_val:

            best_val = val_loss
            counter = 0

            torch.save(model.state_dict(), model_path)

            joblib.dump(x_scaler, x_scaler_path)
            joblib.dump(y_scaler, y_scaler_path)

        else:

            counter += 1

            if counter >= patience:
                print("\nEarly stopping triggered")
                break

    # ======================
    # TEST EVALUATION
    # ======================

    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():

        y_pred_scaled = model(
            torch.tensor(
                X_test,
                dtype=torch.float32
            ).to(device)
        ).cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    y_true = y_scaler.inverse_transform(y_test)

    # ----------------------
    # METRICS
    # ----------------------

    mse = ((y_pred - y_true) ** 2).mean()

    print(f"\nFinal Test MSE: {mse:.4f}")

    print("\nMSE per parameter:")

    for name, val in zip(
        label_names,
        ((y_pred - y_true) ** 2).mean(axis=0)
    ):
        print(f"{name:10s}: {val:.4f}")

    # ----------------------
    # PLOTS
    # ----------------------

    plot_mse(
        output_dir,
        train_losses,
        val_losses
    )

    plot_pred_vs_true(
        y_true,
        y_pred,
        label_names,
        output_dir
    )

    # ======================
    # KOI EVALUATION
    # ======================

    print("\nRunning KOI evaluation...")

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    evaluate_koi_predictions(
        base_dir=base_dir,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        output_dir=output_dir
    )

# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    train_and_evaluate()