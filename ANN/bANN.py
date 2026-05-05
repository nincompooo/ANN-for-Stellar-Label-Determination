import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
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
    plot_pred_vs_true,
    plot_mask_check,
    plot_teff_with_luminosity_ratio,
    evaluate_koi_predictions,
    plot_mse,
    heteroscedastic_loss
)

# ==========================
# PATHS
# ==========================

base_dir = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN"
output_dir = os.path.join(base_dir, "filtered results")
os.makedirs(output_dir, exist_ok=True)

# csv_path = "clean_stellar_dataset.csv"
csv_path = "clean_stellar_dataset_Tdiff1000.csv"
wavelength_file = "Koi1422_HET.txt"


# ==========================
# TRAINING FUNCTION
# ==========================

def train_and_evaluate():

    # ======================
    # LOAD DATA
    # ======================

    df = pd.read_csv(csv_path)
    flux_cols = [c for c in df.columns if c.startswith("flux_")]

    # Convert radii to solar units
    R_SUN = 6.957e10
    df["p_radius"] /= R_SUN
    df["s_radius"] /= R_SUN

    wavelength = np.loadtxt(wavelength_file)[:, 0]
    mask = build_wavelength_mask(wavelength)

    X_full = df[flux_cols].values
    X = X_full[:, mask]
    y = df[label_names].values

    # Normalize spectra
    X = X / np.median(X, axis=1, keepdims=True)
    X = apply_additive_noise(X, snr=30)

    # ======================
    # SCALING
    # ======================

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        StellarDataset(X_train, y_train),
        batch_size=128,
        shuffle=True
    )

    test_loader = DataLoader(
        StellarDataset(X_test, y_test),
        batch_size=128
    )

    # ======================
    # MODEL SETUP
    # ======================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = StellarANN(X_train.shape[1], len(label_names)).to(device)

    # Stronger secondary weighting (bro idek what this is anymore lmaoo)
    weights = torch.tensor([1.0, 10.0, 1.0, 5.0, 1.0, 5.0], device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = float("inf")
    patience = 20
    counter = 0

    # ======================
    # TRAINING LOOP
    # ======================

    n_epochs = 100 

    testing_loss = []
    val_losses = []
    train_losses = []


    for epoch in range(n_epochs):

        train_loss = 0

        # ----- Training -----
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            # pred = model(xb)

            # Proper weighted MSE
            # loss_per_param = (pred - yb) ** 2
            # weighted_loss = loss_per_param * weights
            # loss = weighted_loss.mean()

            #####################

            # criterion = nn.MSELoss()
            # pred = model(xb)
            # loss = criterion(pred, yb)


            pred = model(xb)

            # compute luminosity ratio from TRUE labels (UNSCALED)
            yb_unscaled = torch.tensor(
                y_scaler.inverse_transform(yb.cpu().numpy()),
                dtype=torch.float32,
                device=device
            )

            p_teff = yb_unscaled[:, label_names.index("p_teff")]
            s_teff = yb_unscaled[:, label_names.index("s_teff")]
            p_rad  = yb_unscaled[:, label_names.index("p_radius")]
            s_rad  = yb_unscaled[:, label_names.index("s_radius")]

            lum_ratio = (s_rad / p_rad)**2 * (s_teff / p_teff)**4

            loss = heteroscedastic_loss(pred, yb, lum_ratio, label_names)
            val_loss += heteroscedastic_loss(pred, yb, lum_ratio, label_names).item()

            #####################

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ----- Validation -----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)

                pred = model(xb)
                val_loss += criterion(pred, yb).item()

                # pred = model(xb)
                # loss_per_param = (pred - yb) ** 2
                # weighted_loss = loss_per_param * weights
                # val_loss += weighted_loss.mean().item()

        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        # ----- Early Stopping -----
        if val_loss < best_val:
            best_val = val_loss
            counter = 0

            torch.save(model.state_dict(),
                       os.path.join(base_dir, "stellar_ann_model.pt"))

            joblib.dump(x_scaler,
                        os.path.join(base_dir, "x_scaler.save"))

            joblib.dump(y_scaler,
                        os.path.join(base_dir, "y_scaler.save"))

            # print("New best model saved")

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    # ======================
    # EVALUATION
    # ======================



    model.load_state_dict(
        torch.load(os.path.join(base_dir, "stellar_ann_model.pt"))
    )
    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(
            torch.tensor(X_test, dtype=torch.float32).to(device)
        ).cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test)

    # ======================
    # TEMPORARY EXTRACT PARAMETERS
    # ======================

    idx_p_teff = label_names.index("p_teff")
    idx_s_teff = label_names.index("s_teff")
    idx_p_rad  = label_names.index("p_radius")
    idx_s_rad  = label_names.index("s_radius")

    # True vs predicted Teff
    p_teff = y_true[:, idx_p_teff]
    p_teff_pred = y_pred[:, idx_p_teff]

    s_teff = y_true[:, idx_s_teff]
    s_teff_pred = y_pred[:, idx_s_teff]

    # Radii (needed for luminosity ratio)
    p_rad = y_true[:, idx_p_rad]
    s_rad = y_true[:, idx_s_rad]

    log_lum_ratio = np.log10(
        (s_rad/p_rad)**2 * 
        (s_teff/p_teff)**4
    )

    # ======================

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    # okay i was stupid, it's MSE not MAE and MRE

    from sklearn.metrics import mean_squared_error
    # overall MSE, kinda useless idfk
    mse_sklearn = mean_squared_error(y_true, y_pred)
    print(f"MSE: {mse_sklearn:.4f}")

    # by parameter heh

    print("\nMean Squared Error per parameter:")
    mse_per_param = ((y_pred - y_true) ** 2).mean(axis=0)

    for name, mse in zip(label_names, mse_per_param):
        print(f"{name:10s}: {mse:.4f}")

    # print("\nMean Absolute Error per parameter:")
    # for name, mae in zip(label_names, abs_errors.mean(axis=0)):
    #     print(f"{name:10s}: {mae:.4f}")

    # rel_errors = abs_errors / np.abs(y_true)

    # print("\nMedian Relative Errors per parameter:")
    # for name, re in zip(label_names, np.median(rel_errors, axis=0)):
    #     print(f"{name:10s}: {re*100:.2f}%")

    # ======================
    # PLOTS
    # ======================

    #  plotting mse vs # of iterations for testing loss ---> like the only one we actually gaf about tbh
    plot_mse(output_dir, val_losses, train_losses, y_true, y_pred)

    plot_pred_vs_true(y_true, y_pred, label_names, output_dir)
    plot_mask_check(df, wavelength, build_wavelength_mask, output_dir)

    # Secondary Teff plot
    plot_teff_with_luminosity_ratio(
        s_teff,
        s_teff_pred,
        log_lum_ratio,
        save_path=os.path.join(output_dir, "Secondary Teff Luminosity Ratio.png")
    )

    # Primary Teff plot
    plot_teff_with_luminosity_ratio(
        p_teff,
        p_teff_pred,
        log_lum_ratio,
        save_path=os.path.join(output_dir, "primary teff luminosity ratio.png")
    )

    print("\nNow we're doing that real KOI data bullshit")

    evaluate_koi_predictions(output_dir)



# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    train_and_evaluate()