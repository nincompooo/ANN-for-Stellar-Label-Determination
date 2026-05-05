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
    StellarDataset,
    StellarANN,
    apply_additive_noise,
    preprocess_spectrum_matrix,
    plot_pred_vs_true,
    plot_mse,
    evaluate_koi_predictions,
)

# ==========================
# CONSTANTS
# ==========================

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

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(csv_path)

    flux_cols = [c for c in df.columns if c.startswith("flux_")]

    # -----------------------------------
    # CONVERT RADII TO SOLAR UNITS
    # -----------------------------------

    R_SUN = 6.957e10

    df["p_radius"] /= R_SUN
    df["s_radius"] /= R_SUN

    # -----------------------------------
    # INPUTS / LABELS
    # -----------------------------------

    X_full = df[flux_cols].values

    y = df[label_names].values

    # -----------------------------------
    # LOAD WAVELENGTH GRID
    # -----------------------------------

    wavelength = np.loadtxt("Koi1422_HET.txt")[:, 0]

    # -----------------------------------
    # CONSISTENT MASKING + NORMALIZATION
    # -----------------------------------

    X, mask = preprocess_spectrum_matrix(
        X_full,
        wavelength
    )

    # -----------------------------------
    # ADDITIVE NOISE
    # -----------------------------------

    X = apply_additive_noise(
        X,
        snr=30
    )

    print(f"\nMasked feature count: {X.shape[1]}")

    if X.shape[1] != EXPECTED_FEATURES:

        raise ValueError(
            f"[FATAL] Expected {EXPECTED_FEATURES} features "
            f"but got {X.shape[1]}"
        )

    # ===================================
    # LUMINOSITY RATIO WEIGHTS
    # ===================================
    #
    # Ls/Lp = (Rs/Rp)^2 * (Ts/Tp)^4
    #
    # Hard low-luminosity systems
    # receive larger weights.
    #
    # This prevents training from being
    # dominated by easy bright secondaries.
    #
    # ===================================

    p_teff = df["p_teff"].values
    s_teff = df["s_teff"].values

    p_radius = df["p_radius"].values
    s_radius = df["s_radius"].values

    lum_ratio = (
        (s_radius / p_radius) ** 2
        *
        (s_teff / p_teff) ** 4
    )

    # -----------------------------------
    # INVERSE WEIGHTING
    # -----------------------------------

    sample_weights = 1.0 / (lum_ratio + 0.05)

    # avoid huge exploding weights
    sample_weights = np.clip(
        sample_weights,
        1.0,
        20.0
    )

    # normalize mean weight to 1
    sample_weights = (
        sample_weights /
        np.mean(sample_weights)
    )

    print("\nSample weight stats:")
    print(f"min = {sample_weights.min():.3f}")
    print(f"max = {sample_weights.max():.3f}")
    print(f"mean = {sample_weights.mean():.3f}")

    # -----------------------------------
    # SCALE
    # -----------------------------------

    x_scaler = StandardScaler()

    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)

    y_scaled = y_scaler.fit_transform(y)

    # -----------------------------------
    # SPLIT
    # -----------------------------------

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
        w_train,
        w_temp
    ) = train_test_split(
        X_scaled,
        y_scaled,
        sample_weights,
        test_size=0.30,
        random_state=42
    )

    (
        X_val,
        X_test,
        y_val,
        y_test,
        w_val,
        w_test
    ) = train_test_split(
        X_temp,
        y_temp,
        w_temp,
        test_size=0.50,
        random_state=42
    )

    # -----------------------------------
    # DATASETS
    # -----------------------------------

    train_dataset = StellarDataset(
        X_train,
        y_train,
        w_train
    )

    val_dataset = StellarDataset(
        X_val,
        y_val,
        w_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False
    )

    # -----------------------------------
    # DEVICE
    # -----------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}")

    # ===================================
    # BIGGER ANN
    # ===================================
    #
    # 1947 -> 512 -> 256 -> 128 -> 6
    #
    # NO DROPOUT
    #
    # ===================================

    model = StellarANN(
        n_input=X.shape[1],
        n_output=len(label_names)
    ).to(device)

    # -----------------------------------
    # OPTIMIZER
    # -----------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-5
    )

    # -----------------------------------
    # LOSS
    # -----------------------------------

    criterion = torch.nn.MSELoss(
        reduction="none"
    )

    # -----------------------------------
    # EARLY STOPPING
    # -----------------------------------

    best_val = float("inf")

    patience = 20

    counter = 0

    train_losses = []

    val_losses = []

    # ===================================
    # TRAIN LOOP
    # ===================================

    print("\nBeginning training...\n")

    for epoch in range(100):

        # ------------------------------
        # TRAIN
        # ------------------------------

        model.train()

        train_loss = 0

        for xb, yb, wb in train_loader:

            xb = xb.to(device)

            yb = yb.to(device)

            wb = wb.to(device)

            pred = model(xb)

            # --------------------------
            # WEIGHTED MSE
            # --------------------------

            loss_per_sample = criterion(
                pred,
                yb
            ).mean(dim=1)

            weighted_loss = (
                loss_per_sample * wb
            ).mean()

            optimizer.zero_grad()

            weighted_loss.backward()

            optimizer.step()

            train_loss += weighted_loss.item()

        train_loss /= len(train_loader)

        train_losses.append(train_loss)

        # ------------------------------
        # VALIDATION
        # ------------------------------

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for xb, yb, wb in val_loader:

                xb = xb.to(device)

                yb = yb.to(device)

                wb = wb.to(device)

                pred = model(xb)

                loss_per_sample = criterion(
                    pred,
                    yb
                ).mean(dim=1)

                weighted_loss = (
                    loss_per_sample * wb
                ).mean()

                val_loss += weighted_loss.item()

        val_loss /= len(val_loader)

        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:3d} | "
            f"Train {train_loss:.5f} | "
            f"Val {val_loss:.5f}"
        )

        # ------------------------------
        # SAVE BEST MODEL
        # ------------------------------

        if val_loss < best_val:

            best_val = val_loss

            counter = 0

            torch.save(
                model.state_dict(),
                model_path
            )

            joblib.dump(
                x_scaler,
                x_scaler_path
            )

            joblib.dump(
                y_scaler,
                y_scaler_path
            )

        else:

            counter += 1

            if counter >= patience:

                print("\nEarly stopping triggered.")

                break

    # ===================================
    # TEST EVALUATION
    # ===================================

    print("\nLoading best model...")

    model.load_state_dict(
        torch.load(model_path)
    )

    model.eval()

    with torch.no_grad():

        y_pred_scaled = model(
            torch.tensor(
                X_test,
                dtype=torch.float32
            ).to(device)
        ).cpu().numpy()

    y_pred = y_scaler.inverse_transform(
        y_pred_scaled
    )

    y_true = y_scaler.inverse_transform(
        y_test
    )

    # -----------------------------------
    # METRICS
    # -----------------------------------

    mse = np.mean(
        (y_pred - y_true) ** 2
    )

    print(f"\nFinal Test MSE: {mse:.6f}")

    print("\nPer-parameter MSE:")

    for name, val in zip(
        label_names,
        ((y_pred - y_true) ** 2).mean(axis=0)
    ):
        print(f"{name:12s}: {val:.6f}")

    # -----------------------------------
    # PLOTS
    # -----------------------------------

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

    # ===================================
    # KOI EVALUATION
    # ===================================

    print("\nRunning KOI evaluation...")

    x_scaler = joblib.load(
        x_scaler_path
    )

    y_scaler = joblib.load(
        y_scaler_path
    )

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