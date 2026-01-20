import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# SETUP
# =========================
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# =========================
# LOAD DATA
# =========================
dataset = pd.read_csv("noisy_stellar_dataset.csv")

R_sun = 6.957e10  # cm
dataset["p_radius"] /= R_sun
dataset["s_radius"] /= R_sun

dataset.to_csv("stellar_dataset_scaled.csv", index=False)

# =========================
# FEATURES / TARGETS
# =========================
X = dataset.filter(like="flux_").values

cols = ["p_teff", "s_teff", "p_logg", "s_logg", "p_radius", "s_radius"]
y = dataset[cols].values

# =========================
# SCALING
# =========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

# =========================
# TRAIN / TEST SPLIT
# =========================
train_idx, test_idx = train_test_split(
    np.arange(len(dataset)), test_size=0.2, random_state=42
)

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

# =========================
# WEIGHTED + PHYSICS LOSS
# =========================
def weighted_physics_loss(weights):

    IDX = {
        "p_teff": 0,
        "s_teff": 1,
        "p_logg": 2,
        "s_logg": 3,
        "p_radius": 4,
        "s_radius": 5,
    }

    def loss(y_true, y_pred):

        # Weighted MSE
        sq_err = tf.square(y_true - y_pred)
        weighted_mse = tf.reduce_mean(
            tf.reduce_sum(sq_err * weights, axis=1)
        )

        penalty = 0.0

        # --- Temperature bounds ---
        penalty += tf.reduce_mean(tf.nn.relu(3000 - y_pred[:, IDX["p_teff"]]))
        penalty += tf.reduce_mean(tf.nn.relu(y_pred[:, IDX["p_teff"]] - 7000))

        penalty += tf.reduce_mean(tf.nn.relu(3000 - y_pred[:, IDX["s_teff"]]))
        penalty += tf.reduce_mean(
            tf.nn.relu(y_pred[:, IDX["s_teff"]] - y_pred[:, IDX["p_teff"]])
        )

        # --- log g bounds ---
        for key in ["p_logg", "s_logg"]:
            penalty += tf.reduce_mean(tf.nn.relu(4.0 - y_pred[:, IDX[key]]))
            penalty += tf.reduce_mean(tf.nn.relu(y_pred[:, IDX[key]] - 5.5))

        # --- radius positivity ---
        penalty += tf.reduce_mean(tf.nn.relu(-y_pred[:, IDX["p_radius"]]))
        penalty += tf.reduce_mean(tf.nn.relu(-y_pred[:, IDX["s_radius"]]))

        return weighted_mse + 10.0 * penalty

    return loss

# Secondary-emphasized weights
loss_weights = tf.constant(
    [1.0, 5.0,   # p_teff, s_teff
     1.0, 5.0,   # p_logg, s_logg
     1.0, 8.0],  # p_radius, s_radius
    dtype=tf.float32
)

# =========================
# MODEL
# =========================
def build_model(n_features, n_outputs):
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(n_outputs)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=weighted_physics_loss(loss_weights),
        metrics=["mae"]
    )
    return model

model = build_model(X_train.shape[1], y_train.shape[1])

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    verbose=1
)

# =========================
# PREDICTION + SAFETY CLIP
# =========================
def clip_to_physical(y):
    y[:, 0] = np.clip(y[:, 0], 3000, 7000)
    y[:, 1] = np.clip(y[:, 1], 3000, y[:, 0])
    y[:, 2] = np.clip(y[:, 2], 4.0, 5.5)
    y[:, 3] = np.clip(y[:, 3], 4.0, 5.5)
    y[:, 4] = np.maximum(y[:, 4], 0.0)
    y[:, 5] = np.maximum(y[:, 5], 0.0)
    return y

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_pred = clip_to_physical(y_pred)

y_true = scaler_y.inverse_transform(y_test)

# =========================
# METRICS
# =========================
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\nOverall Metrics:")
print(f"MAE  = {mae:.3f}")
print(f"RMSE = {rmse:.3f}")
print(f"R²   = {r2:.3f}")

# =========================
# SAVE MODEL
# =========================
model.save("stellar_model.keras")

# =========================
# CROSS-VALIDATION
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores, r2_scores = [], []

for fold, (tr, va) in enumerate(kf.split(X_scaled)):
    model_cv = build_model(X_train.shape[1], y_train.shape[1])
    model_cv.fit(X_scaled[tr], y_scaled[tr], epochs=20, batch_size=32, verbose=0)

    y_cv = scaler_y.inverse_transform(model_cv.predict(X_scaled[va]))
    y_cv = clip_to_physical(y_cv)
    y_true_cv = scaler_y.inverse_transform(y_scaled[va])

    mae_scores.append(mean_absolute_error(y_true_cv, y_cv))
    r2_scores.append(r2_score(y_true_cv, y_cv))

print("\nCross-Validation:")
print(f"MAE = {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
print(f"R²  = {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

pd.DataFrame({
    "Fold": range(1, 6),
    "MAE": mae_scores,
    "R2": r2_scores
}).to_csv("cross_validation_results.csv", index=False)

