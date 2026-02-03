import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# SETUP
# =========================
base_dir = "/data/niaycarr/ANN-for-Stellar-Label-Determination/ANN"
os.makedirs(base_dir, exist_ok=True)
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

R_sun = 6.957e10  # cm

# =========================
# LOAD DATA
# =========================
dataset = pd.read_csv("noisy_stellar_dataset.csv")

X = dataset.filter(like="flux_").values

cols = ["p_teff", "s_teff", "p_logg", "s_logg", "p_radius", "s_radius"]
y = dataset[cols].values.astype(np.float32)

# ---- convert radii to solar units ----
y[:, 4:6] /= R_sun 

# =========================
# SCALE INPUTS ONLY
# =========================
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
joblib.dump(scaler_X, os.path.join(base_dir, "scaler_X.pkl"))

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# PHYSICS LOSS
# =========================

def variance_penalty(y_pred, y_true):
    pred_std = tf.math.reduce_std(y_pred, axis=0)
    true_std = tf.math.reduce_std(y_true, axis=0)
    return tf.reduce_mean(tf.square(pred_std - true_std))


def physics_loss(y_true, y_pred):

    p_teff, s_teff = y_pred[:, 0], y_pred[:, 1]
    p_logg, s_logg = y_pred[:, 2], y_pred[:, 3]
    p_rad, s_rad   = y_pred[:, 4], y_pred[:, 5]

    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    penalty = 0.0

    penalty += tf.reduce_mean(tf.nn.relu(3000.0 - p_teff))
    penalty += tf.reduce_mean(tf.nn.relu(p_teff - 7000.0))

    penalty += tf.reduce_mean(tf.nn.relu(3000.0 - s_teff))
    penalty += tf.reduce_mean(tf.nn.relu(s_teff - p_teff))

    for g in [p_logg, s_logg]:
        penalty += tf.reduce_mean(tf.nn.relu(3.5 - g))  # only prevent nonsense


    penalty += tf.reduce_mean(tf.nn.relu(-p_rad))
    penalty += tf.reduce_mean(tf.nn.relu(-s_rad))

    return mse + 1.0*penalty + 0.3*variance_penalty(y_pred, y_true)


# =========================
# MODEL
# =========================
def build_model(n_features, n_outputs):
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(128, activation="relu"),

        layers.Dense(n_outputs)
    ])

    model.compile(
        optimizer=Adam(1e-4),
        loss=physics_loss,
        metrics=["mae"]
    )
    return model

model = build_model(X_train.shape[1], y_train.shape[1])

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=60,
    batch_size=32,
    verbose=1
)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)

# convert radii back to cm for output
# y_pred[:, 4:6] *= R_sun
# y_test[:, 4:6] *= R_sun

label_names = cols

results_df = pd.DataFrame(y_pred, columns=[f"pred_{n}" for n in label_names])
results_df.to_csv(os.path.join(base_dir, "test_predictions.csv"), index=False)

model.save(os.path.join(base_dir, "stellar_model.keras"))


# =========================
# METRICS
# =========================
#2/2/26
label_names = ["p_teff", "s_teff", "p_logg", "s_logg", "p_radius", "s_radius"]

y_true = y_test.copy()   # ground truth
errors = y_pred - y_true
abs_errors = np.abs(errors)

pred_df = pd.DataFrame(y_pred, columns=[f"pred_{n}" for n in label_names])
true_df = pd.DataFrame(y_true, columns=[f"true_{n}" for n in label_names])
err_df  = pd.DataFrame(errors, columns=[f"err_{n}" for n in label_names])

results_df = pd.concat([true_df, pred_df, err_df], axis=1)
results_df.to_csv(os.path.join(base_dir, "test_predictions.csv"), index=False)


# error distributions
for i, name in enumerate(label_names):
    plt.figure(figsize=(6,4))
    plt.hist(errors[:, i], bins=50, alpha=0.7)
    plt.axvline(0, color="k", linestyle="--", alpha=0.7)
    plt.xlabel(f"Prediction Error ({name})")
    plt.ylabel("Count")
    plt.title(f"Error Distribution: {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"error_dist_{name}.png"))
    plt.show()

# MAE labels
mae_per_label = abs_errors.mean(axis=0)

plt.figure(figsize=(7,5))
plt.bar(label_names, mae_per_label)
plt.ylabel("MAE")
plt.title("MAE per Stellar Label")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mae_per_label.png"))
plt.show()

for name, val in zip(label_names, mae_per_label):
    print(f"{name} MAE = {val:.3f}")

# predicted vs true
for i, name in enumerate(label_names):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true[:, i], y_pred[:, i], s=5, alpha=0.5)
    
    minv = min(y_true[:, i].min(), y_pred[:, i].min())
    maxv = max(y_true[:, i].max(), y_pred[:, i].max())
    plt.plot([minv, maxv], [minv, maxv], "r--", lw=2)

    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"Predicted vs True: {name}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pred_vs_true_{name}.png"))
    plt.show()


# mae = mean_absolute_error(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# r2 = r2_score(y_true, y_pred)

# print("\nOverall Metrics:")
# print(f"MAE  = {mae:.3f}")
# print(f"RMSE = {rmse:.3f}")
# print(f"R²   = {r2:.3f}")


mae_per_label = abs_errors.mean(axis=0)
rmse_per_label = np.sqrt((errors**2).mean(axis=0))

metrics_df = pd.DataFrame({
    "Label": label_names,
    "MAE": mae_per_label,
    "RMSE": rmse_per_label
})

metrics_df.to_csv(os.path.join(base_dir, "metrics_per_label.csv"), index=False)

print(metrics_df)


#2/2/26
# =========================
# OVERFITTING DIAGNOSTIC PLOTS
# =========================

train_loss = history.history["loss"]
val_loss   = history.history["val_loss"]

train_mae  = history.history["mae"]
val_mae    = history.history["val_mae"]

epochs = np.arange(1, len(train_loss) + 1)

best_epoch = np.argmin(val_loss) + 1

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ---- Loss panel ----
axs[0].plot(epochs, train_loss, label="Train Loss")
axs[0].plot(epochs, val_loss, label="Validation Loss")
axs[0].axvline(best_epoch, linestyle="--")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("MSE Loss")
axs[0].set_title("Training vs Validation Loss")
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[0].annotate(
    "Best validation loss",
    xy=(best_epoch, val_loss[best_epoch-1]),
    xytext=(best_epoch+3, val_loss[best_epoch-1]*1.1),
    arrowprops=dict(arrowstyle="->")
)

# ---- MAE panel ----
axs[1].plot(epochs, train_mae, label="Train MAE")
axs[1].plot(epochs, val_mae, label="Validation MAE")
axs[1].axvline(best_epoch, linestyle="--")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("MAE")
axs[1].set_title("Training vs Validation MAE")
axs[1].legend()
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "overfitting_diagnostic.png"))
plt.show()


print("Pred std:", y_pred[:,3].std())
print("True std:", y_true[:,3].std())


# =========================
# SAVE MODEL
# =========================
# model.save("stellar_model.keras")

model.save(os.path.join(base_dir, "stellar_model.keras"))
joblib.dump(scaler_X, os.path.join(base_dir, "scaler_X.pkl"))
# joblib.dump(scaler_y, os.path.join(base_dir, "scaler_y.pkl"))

# later we can run:
# model = tf.keras.models.load_model("ann_results/stellar_model.keras",
#                                    custom_objects={"loss": weighted_physics_loss(loss_weights)})


# =========================
# CROSS-VALIDATION
# =========================
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# mae_scores, r2_scores = [], []

# for fold, (tr, va) in enumerate(kf.split(X_scaled)):
#     model_cv = build_model(X_train.shape[1], y_train.shape[1])
#     model_cv.fit(X_scaled[tr], y_scaled[tr], epochs=20, batch_size=32, verbose=0)

#     y_cv = scaler_y.inverse_transform(model_cv.predict(X_scaled[va]))
#     y_cv = clip_to_physical(y_cv)
#     y_true_cv = scaler_y.inverse_transform(y_scaled[va])

#     mae_scores.append(mean_absolute_error(y_true_cv, y_cv))
#     r2_scores.append(r2_score(y_true_cv, y_cv))

# print("\nCross-Validation:")
# print(f"MAE = {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
# print(f"R²  = {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

# pd.DataFrame({
#     "Fold": range(1, 6),
#     "MAE": mae_scores,
#     "R2": r2_scores
# }).to_csv("cross_validation_results.csv", index=False)

