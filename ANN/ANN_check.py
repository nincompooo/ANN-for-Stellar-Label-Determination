import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load your trained model and scalers
model = load_model("stellar_model.keras", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Load KOI data: columns = wavelength, primary_flux, secondary_flux
koi = np.loadtxt("Koi 1816 HET.txt")

flux_combined = koi[:, 1]
error = koi[:, 2]

# Normalize fluxes by median primary flux
med_primary_flux = np.median(flux_combined)
if med_primary_flux == 0:
    raise ValueError("Median flux of primary is zero; cannot normalize.")

normalize = flux_combined / med_primary_flux

print(np.median(normalize))


# add 0.1 to every value in the array
# normalize += 0.7


# Scale input features with training scaler
X_scaled = scaler_X.transform(normalize.reshape(1, -1))

# Predict scaled outputs and invert scaling
y_pred_scaled = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

param_names = ["p_teff", "s_teff", "p_logg", "s_logg", "p_radius", "s_radius"]

print("\n===== KOI Prediction Results =====")
for name, value in zip(param_names, y_pred):
    print(f"{name:10s} = {value:.4f}")


