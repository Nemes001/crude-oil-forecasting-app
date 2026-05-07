"""
Crude Oil WTI — GCN in TensorFlow (from scratch)
=================================================
Replicates the PyTorch GCN without torch_geometric.
GCNConv math implemented manually:
    H' = σ(D^(-1/2) * A_hat * D^(-1/2) * H * W)
    where A_hat = A + I  (adjacency + self-loops)

Graph structure:
    Nodes     : each day in the sliding window
    Edges     : sequential (each day → next day, bidirectional)
    Node feats: scaled OHLCV + engineered features
    Label     : next day price delta (same as CBAM-CNN and MC-CNN)

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

# ─── 1. Build normalised adjacency matrix ─────────────────────────────────────
# Sequential edges: node i ↔ node i+1 (bidirectional chain)
# A_hat = A + I  →  normalised as D^(-1/2) * A_hat * D^(-1/2)

WINDOW_SIZE = 10

def build_adj(window):
    """
    Returns normalised adjacency matrix A_norm of shape (window, window).
    This is the symmetric normalisation used in Kipf & Welling 2017.
    """
    # Build adjacency with self-loops (A + I)
    A = np.zeros((window, window), dtype=np.float32)
    for i in range(window - 1):
        A[i, i+1] = 1.0
        A[i+1, i] = 1.0
    A += np.eye(window, dtype=np.float32)   # self-loops

    # Degree matrix D
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))

    # Symmetric normalisation: D^(-1/2) * A * D^(-1/2)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm.astype(np.float32)

A_norm = build_adj(WINDOW_SIZE)
print(f"Adjacency matrix shape : {A_norm.shape}")
print(f"Adjacency sample row   : {A_norm[0].round(3)}\n")

# ─── 2. Delta prediction sequences ───────────────────────────────────────────

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
print(f"Sequences  : {X.shape}  →  (samples, nodes, features)")
print(f"Delta range: ${y_delta.min():.2f} → ${y_delta.max():.2f} / bbl\n")

delta_mean = float(y_delta.mean())
delta_std  = float(y_delta.std())
y_norm     = (y_delta - delta_mean) / delta_std

# ─── 3. Train / val / test  (70 / 15 / 15) ───────────────────────────────────

n       = len(X)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

X_train, y_train = X[:n_train],              y_norm[:n_train]
X_val,   y_val   = X[n_train:n_train+n_val], y_norm[n_train:n_train+n_val]
X_test,  y_test  = X[n_train+n_val:],        y_norm[n_train+n_val:]
lp_test          = last_prices[n_train+n_val:]

print(f"Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}\n")

# ─── 4. GCNConv Layer ─────────────────────────────────────────────────────────

class GCNConv(layers.Layer):
    """
    Graph Convolutional Layer (Kipf & Welling, 2017).

    Forward pass:
        H' = σ( A_norm * H * W )
    where:
        A_norm = pre-computed normalised adjacency  (window, window)
        H      = node feature matrix                (batch, window, in_ch)
        W      = trainable weight matrix            (in_ch, out_ch)

    Equivalent to torch_geometric GCNConv.
    """
    def __init__(self, out_channels, use_bias=True,
                 activation=None, reg=None, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.use_bias     = use_bias
        self.activation   = keras.activations.get(activation)
        self.reg          = reg

    def build(self, input_shape):
        in_ch = input_shape[-1]
        self.W = self.add_weight(
            name="W", shape=(in_ch, self.out_channels),
            initializer="glorot_uniform",
            regularizer=self.reg, trainable=True)
        if self.use_bias:
            self.b = self.add_weight(
                name="b", shape=(self.out_channels,),
                initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, H, A_norm):
        """
        H      : (batch, nodes, in_ch)
        A_norm : (nodes, nodes)   — broadcast over batch
        returns: (batch, nodes, out_ch)
        """
        # H * W  →  (batch, nodes, out_ch)
        HW = tf.matmul(H, self.W)
        # A_norm * HW  →  (batch, nodes, out_ch)
        out = tf.matmul(A_norm, HW)
        if self.use_bias:
            out = out + self.b
        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"out_channels": self.out_channels,
                    "use_bias": self.use_bias})
        return cfg

# ─── 5. GCN Model ─────────────────────────────────────────────────────────────

class CrudeOilGCN(Model):
    """
    3-layer GCN → global mean pool → MLP head → price delta.

    Mirrors the PyTorch architecture exactly:
        GCNConv(NUM_FEAT → 64) → BN → ReLU → Dropout
        GCNConv(64 → 64)       → BN → ReLU → Dropout
        GCNConv(64 → 32)       → BN → ReLU
        GlobalMeanPool
        Dense(32) → ReLU → Dropout → Dense(1)
    """
    def __init__(self, in_channels, hidden=64, out=32,
                 dropout=0.2, reg=None, **kwargs):
        super().__init__(**kwargs)
        self.in_channels  = in_channels
        self.hidden       = hidden
        self.out_ch       = out
        self.dropout_rate = dropout

        self.conv1 = GCNConv(hidden, activation=None, reg=reg, name="gcn1")
        self.conv2 = GCNConv(hidden, activation=None, reg=reg, name="gcn2")
        self.conv3 = GCNConv(out,    activation=None, reg=reg, name="gcn3")

        self.bn1 = layers.BatchNormalization(name="bn1")
        self.bn2 = layers.BatchNormalization(name="bn2")
        self.bn3 = layers.BatchNormalization(name="bn3")

        self.drop = layers.Dropout(dropout)

        self.head = keras.Sequential([
            layers.Dense(32, activation="relu",
                         kernel_regularizer=reg, name="dense1"),
            layers.Dropout(dropout),
            layers.Dense(1, name="output"),
        ], name="mlp_head")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "in_channels" : self.in_channels,
            "hidden"      : self.hidden,
            "out"         : self.out_ch,
            "dropout"     : self.dropout_rate,
        })
        return cfg

    def call(self, inputs, training=False):
        """
        inputs : (batch, nodes, features)
        A_norm : fixed adjacency — passed via self.A_norm set before call
        """
        A = self.A_norm   # (nodes, nodes)

        x = self.conv1(inputs, A)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.drop(x, training=training)

        x = self.conv2(x, A)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.drop(x, training=training)

        x = self.conv3(x, A)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        # Global mean pool over node dimension → (batch, out)
        x = tf.reduce_mean(x, axis=1)

        return tf.squeeze(self.head(x, training=training), axis=-1)


reg   = regularizers.l2(1e-4)
model = CrudeOilGCN(in_channels=NUM_FEAT, hidden=64, out=32,
                     dropout=0.2, reg=reg, name="CrudeOilGCN")

# Attach adjacency matrix to model
model.A_norm = tf.constant(A_norm)

# Build by passing a dummy batch
_ = model(X_train[:2], training=False)
model.summary()
print(f"\nTotal parameters : {model.count_params():,}\n")

# ─── 6. Compile & train ───────────────────────────────────────────────────────

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3,
                                     weight_decay=1e-4),
    loss="mse",
    metrics=["mae"]
)

cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=30,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=10, min_lr=1e-6, verbose=1),
    # ModelCheckpoint removed — use save_weights for subclassed models
    # callbacks.ModelCheckpoint not supported for subclassed Model
]

print("Training GCN (TensorFlow)…\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=32,
    callbacks=cb_list,
    verbose=1,
)

# ─── 7. Evaluation ────────────────────────────────────────────────────────────

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

# ─── 8. Plots ─────────────────────────────────────────────────────────────────

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
style_ax(ax1, "Training & validation loss (MSE)")
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
    f"GCN (TensorFlow)  |  MAE ${mae:.2f}  RMSE ${rmse:.2f}  R² {r2:.3f}",
    color=TEXT, fontsize=13, fontweight="bold", y=1.01)

plt.savefig("gcn_tf_results.png", dpi=150,
            bbox_inches="tight", facecolor=BG)
plt.show()
print("Plot saved → gcn_tf_results.png")

# ─── 9. Save ──────────────────────────────────────────────────────────────────

model.save_weights("gcn_tf_weights.weights.h5")
np.save("gcn_tf_delta_stats.npy", np.array([delta_mean, delta_std]))
np.save("gcn_tf_adj.npy", A_norm)
print("Weights saved → gcn_tf_weights.weights.h5")
print("Delta stats  → gcn_tf_delta_stats.npy")
print("Adjacency    → gcn_tf_adj.npy")
print("\nTo reload and predict:")
print("  model        = CrudeOilGCN(in_channels=NUM_FEAT)")
print("  model.A_norm = tf.constant(np.load('gcn_tf_adj.npy'))")
print("  model(X_train[:1])  # build weights")
print("  model.load_weights('gcn_tf_weights.weights.h5')")
print("  stats        = np.load('gcn_tf_delta_stats.npy')")
print("  pred_price   = last_price + (model.predict(X) * stats[1] + stats[0])")
