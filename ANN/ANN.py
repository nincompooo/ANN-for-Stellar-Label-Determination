import os
import pandas as pd
import numpy as np

def create_dataset_from_dir(data_dir):
    rows = []
    for fname in os.listdir(data_dir):
        if not fname.startswith("spec_") or not fname.endswith(".txt"):
            continue

        # spec_{p_teff}_{s_teff}_{p_logg}_{s_logg}_{p_radius}_{s_radius}.txt
        parts = fname.replace(".txt", "").split("_")
        if len(parts) != 7:
            print(f"Skipping malformed filename: {fname}")
            continue

        _, p_teff, s_teff, p_logg, s_logg, p_radius, s_radius = parts
        params = {
            "p_teff": float(p_teff),
            "s_teff": float(s_teff),
            "p_logg": float(p_logg),
            "s_logg": float(s_logg),
            "p_radius": float(p_radius),
            "s_radius": float(s_radius),
        }

        filepath = os.path.join(data_dir, fname)

        df = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")
        df.columns = ["wv", "flux_p", "flux_s", "flux_combined"]

        flux_combined = df["flux_combined"].values
        row = {**params}

        for i, flux in enumerate(flux_combined):
            row[f"flux_{i}"] = flux

        rows.append(row)

    return pd.DataFrame(rows)


# Paths for clean and noisy data
clean_dir = "/data/niaycarr/clean1_spectrum" 
noisy_dir = "/data/niaycarr/noisy1_spectrum" 
# clean_data_dir = "/data/niaycarr/data/spectrum"
# noisy_data_dir = "/data/niaycarr/data/noisy_spectrum"

# Create datasets
clean_dataset = create_dataset_from_dir(clean_dir)
noisy_dataset = create_dataset_from_dir(noisy_dir)

# Save datasets to CSV
clean_dataset.to_csv("clean_stellar_dataset.csv", index=False)
noisy_dataset.to_csv("noisy_stellar_dataset.csv", index=False)

print(f"Saved clean dataset with shape: {clean_dataset.shape}")
print(clean_dataset.head(1).T)

print(f"Saved noisy dataset with shape: {noisy_dataset.shape}")
print(noisy_dataset.head(1).T)


