"""
Crude Oil WTI — Multi-Channel CNN (TensorFlow)
===============================================
Idea: different CNN branches look at the same window
      through different "lenses" (kernel sizes), then
      their outputs are fused — like an Inception block
      applied to time-series.

Channels (parallel branches):
  Branch 1 — kernel=3  : captures very short-term moves (1-3 days)
  Branch 2 — kernel=7  : captures weekly patterns
  Branch 3 — kernel=14 : captures bi-weekly trends
  Branch 4 — kernel=21 : captures monthly momentum

Each branch is a dilated causal CNN so time ordering is respected.
All branch outputs are concatenated → shared dense head.

Target: price DELTA (same trick that fixed R² in CBAM-CNN)

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

raw_target = tgt_scaler.inverse_transform(
                 scaled_target.reshape(-1,1)).flatten().astype(np.float32)

# ─── 1. Sliding window — delta prediction ─────────────────────────────────────

WINDOW_SIZE = 30   # larger window so all kernel sizes fit

def make_sequences(feat, raw_prices, window=WINDOW_SIZE):
    X, y_delta, last_prices = [], [], []
    for i in range(len(feat) - window):
        X.append(feat[i : i + window])
        last_price = raw_prices[i + window - 1]
        next_price = raw_prices[i + window]
        y_delta.append(next_price - last_price)
        last_prices.append(last_price)
    return (np.array(X,           dtype=np.float32),
            np.array(y_delta,     dtype=np.float32),
            np.array(last_prices, dtype=np.float32))

X, y_delta, last_prices = make_sequences(scaled_feat, raw_target)
print(f"Sequences  : {X.shape}")
print(f"Delta range: ${y_delta.min():.2f} → ${y_delta.max():.2f} / bbl")
print(f"Delta std  : ${y_delta.std():.2f} / bbl\n")

delta_mean = float(y_delta.mean())
delta_std  = float(y_delta.std())
y_norm     = (y_delta - delta_mean) / delta_std

# ─── 2. Train / val / test  (70 / 15 / 15) ───────────────────────────────────

n       = len(X)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

X_train, y_train = X[:n_train],              y_norm[:n_train]
X_val,   y_val   = X[n_train:n_train+n_val], y_norm[n_train:n_train+n_val]
X_test,  y_test  = X[n_train+n_val:],        y_norm[n_train+n_val:]
lp_test          = last_prices[n_train+n_val:]

print(f"Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}\n")

# ─── 3. Building blocks ───────────────────────────────────────────────────────

def causal_conv_block(x, filters, kernel_size, dilation_rate=1, reg=None):
    """Causal dilated residual conv block."""
    residual = x
    x = layers.Conv1D(filters, kernel_size, padding="causal",
                      dilation_rate=dilation_rate,
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filters, kernel_size, padding="causal",
                      dilation_rate=dilation_rate,
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters, 1, padding="same",
                                 use_bias=False)(residual)
    return layers.Add()([x, residual])


def channel_branch(inp, filters, kernel_size, reg, name):
    """
    One branch of the multi-channel CNN.
    Three dilated causal conv blocks at increasing dilation rates.
    Each branch focuses on a different temporal scale via kernel_size.
    """
    x = layers.Conv1D(filters, 1, padding="same",
                      use_bias=False, name=f"{name}_proj")(inp)   # input proj

    x = causal_conv_block(x, filters, kernel_size, dilation_rate=1, reg=reg)
    x = causal_conv_block(x, filters, kernel_size, dilation_rate=2, reg=reg)
    x = causal_conv_block(x, filters, kernel_size, dilation_rate=4, reg=reg)

    # Dual pooling per branch
    avg = layers.GlobalAveragePooling1D(name=f"{name}_avg")(x)
    mx  = layers.GlobalMaxPooling1D(name=f"{name}_max")(x)
    return layers.Concatenate(name=f"{name}_pool")([avg, mx])      # (B, 2*filters)

# ─── 4. Multi-channel CNN model ───────────────────────────────────────────────

def build_mc_cnn(window, num_feat):
    """
    Architecture
    ────────────
    Input (window, num_feat)
      │
      ├─ Branch 1 (kernel=3,  filters=64)  ← short-term  1-3 day patterns
      ├─ Branch 2 (kernel=7,  filters=64)  ← weekly patterns
      ├─ Branch 3 (kernel=14, filters=64)  ← bi-weekly patterns
      └─ Branch 4 (kernel=21, filters=64)  ← monthly momentum
      │
      Concatenate all branch outputs  →  (B, 8 * 64 = 512)
      │
      Dense(256) → ReLU → Dropout(0.3)
      Dense(128) → ReLU → Dropout(0.2)
      Dense(64)  → ReLU
      Dense(1)   → normalised price delta
    """
    reg     = regularizers.l2(1e-4)
    FILTERS = 64
    inp     = keras.Input(shape=(window, num_feat), name="input")

    # Four parallel branches — different kernel sizes = different time scales
    b1 = channel_branch(inp, FILTERS, kernel_size=3,  reg=reg, name="short")
    b2 = channel_branch(inp, FILTERS, kernel_size=7,  reg=reg, name="weekly")
    b3 = channel_branch(inp, FILTERS, kernel_size=14, reg=reg, name="biweekly")
    b4 = channel_branch(inp, FILTERS, kernel_size=21, reg=reg, name="monthly")

    # Fuse all branches
    x = layers.Concatenate(name="fusion")([b1, b2, b3, b4])   # (B, 512)

    # Shared dense head
    x   = layers.Dense(256, activation="relu",
                        kernel_regularizer=reg, name="dense1")(x)
    x   = layers.Dropout(0.3)(x)
    x   = layers.Dense(128, activation="relu",
                        kernel_regularizer=reg, name="dense2")(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64,  activation="relu",
                        kernel_regularizer=reg, name="dense3")(x)
    out = layers.Dense(1, name="delta_output")(x)

    return Model(inputs=inp, outputs=out, name="MultiChannel_CNN")


model = build_mc_cnn(WINDOW_SIZE, NUM_FEAT)
model.summary()
print(f"\nTotal parameters : {model.count_params():,}\n")

# ─── 5. Compile & train ───────────────────────────────────────────────────────

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
    callbacks.ModelCheckpoint("mc_cnn_best.keras",
                               monitor="val_loss",
                               save_best_only=True, verbose=0),
]

print("Training Multi-Channel CNN…\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=cb_list,
    verbose=1,
)

# ─── 6. Evaluation ────────────────────────────────────────────────────────────

pred_norm  = model.predict(X_test, verbose=0).flatten()
pred_delta = pred_norm * delta_std + delta_mean
pred_usd   = lp_test + pred_delta
true_usd   = lp_test + (y_test * delta_std + delta_mean)

mae  = mean_absolute_error(true_usd, pred_usd)
rmse = np.sqrt(mean_squared_error(true_usd, pred_usd))
r2   = r2_score(true_usd, pred_usd)
mape = np.mean(np.abs((true_usd - pred_usd) / (true_usd + 1e-8))) * 100
corr = float(np.corrcoef(true_usd, pred_usd)[0, 1])

print(f"\n{'─'*42}")
print(f"  Test results  ({len(pred_usd)} samples)")
print(f"{'─'*42}")
print(f"  MAE       : ${mae:.2f} / bbl")
print(f"  RMSE      : ${rmse:.2f} / bbl")
print(f"  MAPE      : {mape:.2f} %")
print(f"  R²        : {r2:.4f}")
print(f"  Pearson r : {corr:.4f}")
print(f"{'─'*42}")

# ─── 7. Branch contribution analysis ──────────────────────────────────────────
# Show which time scale each branch focuses on most

print("\nBranch receptive fields:")
print(f"  Short   (k=3 ) : up to  {3  * 4}  days effective context")
print(f"  Weekly  (k=7 ) : up to  {7  * 4}  days effective context")
print(f"  Biweekly(k=14) : up to  {14 * 4}  days effective context")
print(f"  Monthly (k=21) : up to  {21 * 4}  days effective context")

# ─── 8. Plots ─────────────────────────────────────────────────────────────────

BG, CARD  = "#0f1117", "#1c1e26"
BLUE, GREEN, AMBER, RED = "#378ADD", "#1D9E75", "#EF9F27", "#E24B4A"
PURPLE    = "#7F77DD"
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

# Loss curves
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, "Training & validation loss")
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
xi = np.arange(len(true_usd))
ax2.plot(xi, true_usd, color=BLUE,  lw=2, marker="o", ms=3, label="Actual")
ax2.plot(xi, pred_usd, color=AMBER, lw=2, marker="^", ms=3,
         linestyle="--", label="Predicted")
ax2.fill_between(xi, true_usd, pred_usd, alpha=0.12, color=RED)
ax2.legend(fontsize=9, facecolor=CARD, labelcolor=TEXT, framealpha=0.6)
ax2.set_xlabel("Test sample index"); ax2.set_ylabel("Price ($/bbl)")

# Scatter
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Actual vs predicted scatter")
ax3.scatter(true_usd, pred_usd, color=BLUE, alpha=0.4, s=10, edgecolors="none")
mn = min(true_usd.min(), pred_usd.min()) - 2
mx = max(true_usd.max(), pred_usd.max()) + 2
ax3.plot([mn, mx], [mn, mx], color=GREEN, lw=1.5, linestyle="--",
         label="Perfect fit")
ax3.set_xlim(mn, mx); ax3.set_ylim(mn, mx)
ax3.text(0.05, 0.90, f"R²  = {r2:.3f}",  transform=ax3.transAxes,
         color=TEXT,  fontsize=10, fontweight="bold")
ax3.text(0.05, 0.82, f"r   = {corr:.3f}", transform=ax3.transAxes,
         color=MUTED, fontsize=9)
ax3.legend(fontsize=9, facecolor=CARD, labelcolor=TEXT, framealpha=0.6)
ax3.set_xlabel("Actual ($/bbl)"); ax3.set_ylabel("Predicted ($/bbl)")

# Residuals
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, "Residuals — actual minus predicted ($/bbl)")
residuals = true_usd - pred_usd
ax4.bar(xi, residuals,
        color=[GREEN if r >= 0 else RED for r in residuals],
        alpha=0.8, width=0.8)
ax4.axhline(0,    color=MUTED, lw=1.0)
ax4.axhline( mae, color=AMBER, lw=1.2, linestyle="--", label=f"+MAE ${mae:.1f}")
ax4.axhline(-mae, color=AMBER, lw=1.2, linestyle="--", label=f"−MAE ${mae:.1f}")
ax4.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, framealpha=0.6)
ax4.set_xlabel("Test sample index"); ax4.set_ylabel("Error ($/bbl)")

fig.suptitle(
    f"Multi-Channel CNN  |  MAE ${mae:.2f}  RMSE ${rmse:.2f}  R² {r2:.3f}",
    color=TEXT, fontsize=13, fontweight="bold", y=1.01)

plt.savefig("mc_cnn_results.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.show()
print("Plot saved → mc_cnn_results.png")

# ─── 9. Save ──────────────────────────────────────────────────────────────────

model.save("mc_cnn.keras")
np.save("mc_cnn_delta_stats.npy", np.array([delta_mean, delta_std]))
print("Model saved → mc_cnn.keras")
print("Delta stats → mc_cnn_delta_stats.npy")
print("\nTo reload and predict:")
print("  model = tf.keras.models.load_model('mc_cnn.keras')")
print("  stats = np.load('mc_cnn_delta_stats.npy')")
print("  delta_mean, delta_std = stats[0], stats[1]")
print("  pred  = last_price + (model.predict(X) * delta_std + delta_mean)")
