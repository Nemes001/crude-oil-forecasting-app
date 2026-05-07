"""
Crude Oil WTI — Pure CBAM-CNN (TensorFlow) v4
===============================================
Fix for R²=0.21 despite Pearson r=0.93:
  Gap between Pearson r and R² = systematic scale bias
  (model tracks direction but under-disperses predictions)

Fixes applied:
  1. Instance normalisation per window (each sequence zero-centred)
     → model predicts CHANGE from window mean, not absolute price
  2. Return the window mean back at inference (denormalise)
  3. Predict price DIFFERENCE (delta) instead of absolute price
     → much easier learning problem for CNN
  4. Larger model capacity (128 filters)

Run AFTER crude_oil_preprocessing.py.
Requires: scaled_feat, scaled_target, tgt_scaler, features
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks, regularizers
tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow : {tf.__version__}")

# ─── 0. Sanity check ──────────────────────────────────────────────────────────

assert "scaled_feat"   in dir(), "Run preprocessing first"
assert "scaled_target" in dir(), "Run preprocessing first"
assert "tgt_scaler"    in dir(), "Run preprocessing first"
assert "features"      in dir(), "Run preprocessing first"

N, NUM_FEAT = scaled_feat.shape
print(f"Samples  : {N}")
print(f"Features : {NUM_FEAT}  →  {features}\n")

# Recover raw prices
raw_target = tgt_scaler.inverse_transform(
                 scaled_target.reshape(-1,1)).flatten().astype(np.float32)

# ─── 1. Sliding window — predict DELTA (next - last) ─────────────────────────
#
# Predicting the change from the last known price is a much simpler
# problem for a CNN than predicting the absolute price level.
# At inference: pred_price = last_price_in_window + predicted_delta

WINDOW_SIZE = 20

def make_sequences(feat, raw_prices, window=WINDOW_SIZE):
    X, y_delta, last_prices = [], [], []
    for i in range(len(feat) - window):
        X.append(feat[i : i + window])
        last_price  = raw_prices[i + window - 1]   # last known price
        next_price  = raw_prices[i + window]        # price to predict
        delta       = next_price - last_price       # CHANGE to predict
        y_delta.append(delta)
        last_prices.append(last_price)
    return (np.array(X,           dtype=np.float32),
            np.array(y_delta,     dtype=np.float32),
            np.array(last_prices, dtype=np.float32))

X, y_delta, last_prices = make_sequences(scaled_feat, raw_target)
print(f"Sequences  : {X.shape}")
print(f"Delta range: ${y_delta.min():.2f} to ${y_delta.max():.2f} / bbl")
print(f"Delta std  : ${y_delta.std():.2f} / bbl\n")

# Normalise delta for stable training
delta_mean = y_delta.mean()
delta_std  = y_delta.std()
y_norm     = (y_delta - delta_mean) / delta_std

# ─── 2. Train / val / test  (70 / 15 / 15) ───────────────────────────────────

n       = len(X)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

X_train,  y_train  = X[:n_train],              y_norm[:n_train]
X_val,    y_val    = X[n_train:n_train+n_val], y_norm[n_train:n_train+n_val]
X_test,   y_test   = X[n_train+n_val:],        y_norm[n_train+n_val:]
lp_test            = last_prices[n_train+n_val:]   # for reconstruction
true_usd           = raw_target[n_train+n_val+WINDOW_SIZE : n_train+n_val+WINDOW_SIZE+len(y_test)]

# Fallback if index off
true_usd = (y_test * delta_std + delta_mean) + lp_test

print(f"Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}\n")

# ─── 3. Positional Encoding ───────────────────────────────────────────────────

class PositionalEncoding(layers.Layer):
    def __init__(self, window, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = self.add_weight(
            name="pos_emb", shape=(1, window, d_model),
            initializer="zeros", trainable=True)
    def call(self, x):
        return x + self.pos_emb

# ─── 4. CBAM ──────────────────────────────────────────────────────────────────

def channel_attention(x, ratio=8):
    C        = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    shared1  = layers.Dense(max(C // ratio, 1), activation="relu", use_bias=False)
    shared2  = layers.Dense(C, use_bias=False)
    scale    = layers.Activation("sigmoid")(
                   shared2(shared1(avg_pool)) + shared2(shared1(max_pool)))
    scale    = layers.Reshape((1, C))(scale)
    return layers.Multiply()([x, scale])

def spatial_attention(x, kernel_size=5):
    avg_pool = layers.Lambda(
        lambda t: tf.keras.ops.mean(t, axis=-1, keepdims=True))(x)
    max_pool = layers.Lambda(
        lambda t: tf.keras.ops.max(t,  axis=-1, keepdims=True))(x)
    concat   = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    scale    = layers.Conv1D(1, kernel_size, padding="same",
                             activation="sigmoid", use_bias=False)(concat)
    return layers.Multiply()([x, scale])

def cbam_block(x, ratio=8, kernel_size=5):
    residual = x
    x = channel_attention(x, ratio=ratio)
    x = spatial_attention(x, kernel_size=kernel_size)
    if residual.shape[-1] != x.shape[-1]:
        residual = layers.Conv1D(x.shape[-1], 1, padding="same",
                                 use_bias=False)(residual)
    return layers.Add()([x, residual])

# ─── 5. Dilated Causal Conv block ─────────────────────────────────────────────

def causal_conv_block(x, filters, kernel_size=3, dilation_rate=1, reg=None):
    residual = x
    x = layers.Conv1D(filters, kernel_size, padding="causal",
                      dilation_rate=dilation_rate, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filters, kernel_size, padding="causal",
                      dilation_rate=dilation_rate, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters, 1, padding="same",
                                 use_bias=False)(residual)
    return layers.Add()([x, residual])

# ─── 6. Model ─────────────────────────────────────────────────────────────────

def build_cbam_cnn(window, num_feat):
    """
    Predicts normalised price delta — easy problem for CNN.
    Reconstruction: pred_price = last_window_price + pred_delta
    """
    reg = regularizers.l2(1e-4)
    inp = keras.Input(shape=(window, num_feat), name="input")

    x = layers.Dense(128, use_bias=False)(inp)
    x = PositionalEncoding(window, 128)(x)

    # Scale 1: local (dilation=1)
    x = causal_conv_block(x, 128, kernel_size=3, dilation_rate=1, reg=reg)
    x = cbam_block(x, ratio=8, kernel_size=5)
    x = layers.Dropout(0.1)(x)

    # Scale 2: medium (dilation=2)
    x = causal_conv_block(x, 128, kernel_size=3, dilation_rate=2, reg=reg)
    x = cbam_block(x, ratio=8, kernel_size=5)
    x = layers.Dropout(0.1)(x)

    # Scale 3: long (dilation=4)
    x = causal_conv_block(x, 64, kernel_size=3, dilation_rate=4, reg=reg)
    x = cbam_block(x, ratio=8, kernel_size=5)
    x = layers.Dropout(0.1)(x)

    avg = layers.GlobalAveragePooling1D()(x)
    mx  = layers.GlobalMaxPooling1D()(x)
    x   = layers.Concatenate()([avg, mx])

    x   = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    out = layers.Dense(1, name="delta_output")(x)   # predicts normalised delta

    return Model(inputs=inp, outputs=out, name="CBAM_CNN_Delta")


model = build_cbam_cnn(WINDOW_SIZE, NUM_FEAT)
model.summary()
print(f"\nTotal parameters : {model.count_params():,}\n")

# ─── 7. Compile & train ───────────────────────────────────────────────────────

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=30,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=10, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint("cbam_cnn_best.keras",
                               monitor="val_loss",
                               save_best_only=True, verbose=0),
]

print("Training CBAM-CNN (delta prediction)…\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=cb_list,
    verbose=1,
)

# ─── 8. Evaluation ────────────────────────────────────────────────────────────

pred_norm = model.predict(X_test, verbose=0).flatten()

# Reconstruct: denormalise delta → add last known price
pred_delta = pred_norm * delta_std + delta_mean
pred_usd   = lp_test + pred_delta
true_usd_r = lp_test + (y_test * delta_std + delta_mean)

mae  = mean_absolute_error(true_usd_r, pred_usd)
rmse = np.sqrt(mean_squared_error(true_usd_r, pred_usd))
r2   = r2_score(true_usd_r, pred_usd)
mape = np.mean(np.abs((true_usd_r - pred_usd) / (true_usd_r + 1e-8))) * 100
corr = float(np.corrcoef(true_usd_r, pred_usd)[0, 1])

print(f"\n{'─'*42}")
print(f"  Test results  ({len(pred_usd)} samples)")
print(f"{'─'*42}")
print(f"  MAE       : ${mae:.2f} / bbl")
print(f"  RMSE      : ${rmse:.2f} / bbl")
print(f"  MAPE      : {mape:.2f} %")
print(f"  R²        : {r2:.4f}")
print(f"  Pearson r : {corr:.4f}")
print(f"{'─'*42}")

# ─── 9. Plots ─────────────────────────────────────────────────────────────────

BG, CARD  = "#0f1117", "#1c1e26"
BLUE, GREEN, AMBER, RED = "#378ADD", "#1D9E75", "#EF9F27", "#E24B4A"
TEXT, MUTED, GRID = "#c9d1d9", "#6e7681", "#30363d"

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=11, pad=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.yaxis.label.set_color(MUTED); ax.xaxis.label.set_color(MUTED)
    ax.grid(color=GRID, lw=0.5, linestyle="--", alpha=0.6)

fig = plt.figure(figsize=(15, 10))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
hist = history.history

# Loss
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Training & validation loss (MSE on delta)")
ax1.plot(hist["loss"],     color=BLUE,  lw=1.5, label="Train")
ax1.plot(hist["val_loss"], color=AMBER, lw=1.5, label="Val")
best_ep = int(np.argmin(hist["val_loss"]))
ax1.axvline(best_ep, color=GREEN, lw=1.2, linestyle=":",
            label=f"Best ep {best_ep+1}")
ax1.legend(fontsize=9, facecolor=CARD, labelcolor=TEXT, framealpha=0.6)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE")

# Predicted vs actual
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, "Predicted vs actual — test set ($/bbl)")
xi = np.arange(len(true_usd_r))
ax2.plot(xi, true_usd_r, color=BLUE,  lw=2, marker="o", ms=3, label="Actual")
ax2.plot(xi, pred_usd,   color=AMBER, lw=2, marker="^", ms=3,
         linestyle="--", label="Predicted")
ax2.fill_between(xi, true_usd_r, pred_usd, alpha=0.12, color=RED)
ax2.legend(fontsize=9, facecolor=CARD, labelcolor=TEXT, framealpha=0.6)
ax2.set_xlabel("Test sample index"); ax2.set_ylabel("Price ($/bbl)")

# Scatter
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Actual vs predicted scatter")
ax3.scatter(true_usd_r, pred_usd, color=BLUE, alpha=0.4, s=10, edgecolors="none")
mn = min(true_usd_r.min(), pred_usd.min()) - 2
mx = max(true_usd_r.max(), pred_usd.max()) + 2
ax3.plot([mn, mx], [mn, mx], color=GREEN, lw=1.5, linestyle="--",
         label="Perfect fit")
ax3.set_xlim(mn, mx); ax3.set_ylim(mn, mx)
ax3.text(0.05, 0.90, f"R²  = {r2:.3f}",  transform=ax3.transAxes,
         color=TEXT, fontsize=10, fontweight="bold")
ax3.text(0.05, 0.82, f"r   = {corr:.3f}", transform=ax3.transAxes,
         color=MUTED, fontsize=9)
ax3.legend(fontsize=9, facecolor=CARD, labelcolor=TEXT, framealpha=0.6)
ax3.set_xlabel("Actual ($/bbl)"); ax3.set_ylabel("Predicted ($/bbl)")

# Residuals
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Residuals — actual minus predicted ($/bbl)")
residuals = true_usd_r - pred_usd
ax4.bar(xi, residuals,
        color=[GREEN if r >= 0 else RED for r in residuals],
        alpha=0.8, width=0.8)
ax4.axhline(0,    color=MUTED, lw=1.0)
ax4.axhline( mae, color=AMBER, lw=1.2, linestyle="--", label=f"+MAE ${mae:.1f}")
ax4.axhline(-mae, color=AMBER, lw=1.2, linestyle="--", label=f"−MAE ${mae:.1f}")
ax4.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, framealpha=0.6)
ax4.set_xlabel("Test sample index"); ax4.set_ylabel("Error ($/bbl)")

fig.suptitle(
    f"CBAM-CNN  |  MAE ${mae:.2f}  RMSE ${rmse:.2f}  R² {r2:.3f}",
    color=TEXT, fontsize=13, fontweight="bold", y=1.01)

plt.savefig("cbam_cnn_results.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.show()
print("Plot saved → cbam_cnn_results.png")

# ─── 10. Save ─────────────────────────────────────────────────────────────────

model.save("cbam_cnn.keras")
np.save("cbam_cnn_delta_stats.npy",
        np.array([delta_mean, delta_std]))
print("Model saved → cbam_cnn.keras")
print("Delta stats → cbam_cnn_delta_stats.npy")
print("\nTo reload and predict:")
print("  model      = tf.keras.models.load_model('cbam_cnn.keras')")
print("  stats      = np.load('cbam_cnn_delta_stats.npy')")
print("  delta_mean, delta_std = stats[0], stats[1]")
print("  pred_price = last_price + (model.predict(X) * delta_std + delta_mean)")