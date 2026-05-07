"""
WTI Crude Oil — FastAPI Backend for Hugging Face Spaces
========================================================
Hugging Face Spaces runs the app via this file.
Entry point: hf_app.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
tf.random.set_seed(42)
np.random.seed(42)

# ─── Paths ────────────────────────────────────────────────────────────────────

MODELS_DIR  = Path(__file__).parent / "models"
CBAM_PATH   = MODELS_DIR / "cbam_cnn.keras"
CBAM_STATS  = MODELS_DIR / "cbam_cnn_delta_stats.npy"
MC_PATH     = MODELS_DIR / "mc_cnn.keras"
MC_STATS    = MODELS_DIR / "mc_cnn_delta_stats.npy"
GCN_WEIGHTS = MODELS_DIR / "gcn_tf_weights.weights.h5"
GCN_ADJ     = MODELS_DIR / "gcn_tf_adj.npy"
GCN_STATS   = MODELS_DIR / "gcn_tf_delta_stats.npy"

# ─── Constants ────────────────────────────────────────────────────────────────

WINDOW_CBAM = 20
WINDOW_MC   = 30
WINDOW_GCN  = 10
NUM_FEAT    = 11
FEATURES    = ["Close","Open","High","Low","Volume","Change_pct",
               "MA4","MA12","Momentum","Volatility","HL_Spread"]

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="WTI Crude Oil Predictor API",
    description="CBAM-CNN · MC-CNN · GCN",
    version="2.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATE = {
    "cbam_model": None, "mc_model": None, "gcn_model": None,
    "cbam_stats": None, "mc_stats": None, "gcn_stats": None,
    "scaled_feat": None, "raw_prices": None, "loaded_at": None,
}

# ─── Custom layers ────────────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEncoding(layers.Layer):
    def __init__(self, window, d_model, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.d_model = d_model
    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.window, self.d_model),
            initializer="zeros", trainable=True)
        super().build(input_shape)
    def call(self, x): return x + self.pos_embedding
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"window": self.window, "d_model": self.d_model})
        return cfg

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMeanLayer(layers.Layer):
    def call(self, x): return tf.reduce_mean(x, axis=-1, keepdims=True)
    def compute_output_shape(self, input_shape): return input_shape[:-1] + (1,)
    def get_config(self): return super().get_config()

@tf.keras.utils.register_keras_serializable(package="Custom")
class ReduceMaxLayer(layers.Layer):
    def call(self, x): return tf.reduce_max(x, axis=-1, keepdims=True)
    def compute_output_shape(self, input_shape): return input_shape[:-1] + (1,)
    def get_config(self): return super().get_config()

# ─── GCN builder ──────────────────────────────────────────────────────────────

def build_gcn(window, num_feat):
    A   = np.zeros((window, window), dtype=np.float32)
    for i in range(window - 1):
        A[i, i+1] = 1.; A[i+1, i] = 1.
    A  += np.eye(window, dtype=np.float32)
    D_i = np.diag(1. / np.sqrt(A.sum(axis=1)))
    A_n = tf.constant((D_i @ A @ D_i).astype(np.float32))

    class GCNConv(layers.Layer):
        def __init__(self, out_ch, **kw):
            super().__init__(**kw); self.out_ch = out_ch
        def build(self, shape):
            self.W = self.add_weight("W", shape=(shape[-1], self.out_ch),
                                     initializer="glorot_uniform", trainable=True)
            self.b = self.add_weight("b", shape=(self.out_ch,),
                                     initializer="zeros", trainable=True)
        def call(self, H, A): return tf.matmul(A, tf.matmul(H, self.W)) + self.b
        def get_config(self):
            c = super().get_config(); c["out_ch"] = self.out_ch; return c

    reg = regularizers.l2(1e-4)
    inp = keras.Input(shape=(window, num_feat))
    gc1 = GCNConv(64, name="gc1"); gc2 = GCNConv(64, name="gc2")
    gc3 = GCNConv(32, name="gc3")
    bn1 = layers.BatchNormalization(); bn2 = layers.BatchNormalization()
    bn3 = layers.BatchNormalization(); dr  = layers.Dropout(0.2)
    x   = tf.nn.relu(bn1(gc1(inp, A_n))); x = dr(x)
    x   = tf.nn.relu(bn2(gc2(x,   A_n))); x = dr(x)
    x   = tf.nn.relu(bn3(gc3(x,   A_n)))
    x   = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)
    x   = layers.Dense(32, activation="relu", kernel_regularizer=reg)(x)
    x   = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    return Model(inp, out, name="GCN")

# ─── Data pipeline ────────────────────────────────────────────────────────────

def fetch_and_process():
    df = yf.download("CL=F", period="2y", interval="1d", progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open","High","Low","Close","Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("B").ffill()
    df["Change_pct"] = df["Close"].pct_change() * 100
    df["MA4"]        = df["Close"].rolling(4).mean()
    df["MA12"]       = df["Close"].rolling(12).mean()
    df["Momentum"]   = df["Close"].diff()
    df["Volatility"] = df["Close"].rolling(4).std()
    df["HL_Spread"]  = df["High"] - df["Low"]
    df = df.dropna()
    raw_prices  = df["Close"].values.astype(np.float32)
    raw_feat    = df[FEATURES].values.astype(np.float32)
    scaler      = MinMaxScaler()
    scaled_feat = scaler.fit_transform(raw_feat).astype(np.float32)
    return scaled_feat, raw_prices

# ─── Load models ──────────────────────────────────────────────────────────────

def load_all_models():
    keras.config.enable_unsafe_deserialization()

    print("Loading CBAM-CNN...")
    cbam = keras.models.load_model(
        str(CBAM_PATH),
        custom_objects={
            "PositionalEncoding": PositionalEncoding,
            "ReduceMeanLayer":    ReduceMeanLayer,
            "ReduceMaxLayer":     ReduceMaxLayer,
        },
        compile=False,
    )
    cbam_stats = np.load(CBAM_STATS)

    print("Loading MC-CNN...")
    mc = keras.models.load_model(str(MC_PATH), compile=False)
    mc_stats = np.load(MC_STATS)

    print("Loading GCN...")
    gcn = build_gcn(WINDOW_GCN, NUM_FEAT)
    gcn(np.zeros((1, WINDOW_GCN, NUM_FEAT), dtype=np.float32))
    gcn.load_weights(str(GCN_WEIGHTS))
    gcn_stats = np.load(GCN_STATS)

    print("All models loaded.")
    return (cbam, cbam_stats), (mc, mc_stats), (gcn, gcn_stats)


def predict_delta(model, scaled_feat, window):
    X = scaled_feat[-window:][np.newaxis]
    return float(model.predict(X, verbose=0).flatten()[0])


def reconstruct(delta_norm, delta_mean, delta_std, last_price):
    return last_price + (delta_norm * delta_std + delta_mean)

# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print("Fetching latest market data...")
    scaled_feat, raw_prices = fetch_and_process()

    print("Loading pre-trained model weights...")
    (cbam, cbam_stats), (mc, mc_stats), (gcn, gcn_stats) = load_all_models()

    STATE.update({
        "cbam_model": cbam, "mc_model": mc, "gcn_model": gcn,
        "cbam_stats": cbam_stats, "mc_stats": mc_stats, "gcn_stats": gcn_stats,
        "scaled_feat": scaled_feat, "raw_prices": raw_prices,
        "loaded_at": datetime.now().isoformat(),
    })
    print(f"API ready at {STATE['loaded_at']}")

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "WTI Crude Oil Predictor API", "status": "running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_ready": STATE["cbam_model"] is not None,
        "loaded_at": STATE["loaded_at"],
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/predict")
def predict():
    if STATE["cbam_model"] is None:
        raise HTTPException(503, detail="Models not ready")

    scaled_feat = STATE["scaled_feat"]
    last_price  = float(STATE["raw_prices"][-1])
    cbam_dm, cbam_ds = STATE["cbam_stats"]
    mc_dm,   mc_ds   = STATE["mc_stats"]
    gcn_dm,  gcn_ds  = STATE["gcn_stats"]

    cbam_pred = reconstruct(predict_delta(STATE["cbam_model"], scaled_feat, WINDOW_CBAM),
                            cbam_dm, cbam_ds, last_price)
    mc_pred   = reconstruct(predict_delta(STATE["mc_model"],   scaled_feat, WINDOW_MC),
                            mc_dm,   mc_ds,   last_price)
    gcn_pred  = reconstruct(predict_delta(STATE["gcn_model"],  scaled_feat, WINDOW_GCN),
                            gcn_dm,  gcn_ds,  last_price)
    ensemble  = (cbam_pred + mc_pred + gcn_pred) / 3
    delta_ens = ensemble - last_price
    pct       = delta_ens / last_price * 100
    signal    = "BUY" if pct > 1 else "SELL" if pct < -1 else "HOLD"

    return {
        "current_price": round(last_price, 2),
        "predictions":   {
            "CBAM_CNN": round(cbam_pred, 2),
            "MC_CNN":   round(mc_pred,   2),
            "GCN":      round(gcn_pred,  2),
            "ensemble": round(ensemble,  2),
        },
        "deltas": {
            "CBAM_CNN": round(cbam_pred - last_price, 2),
            "MC_CNN":   round(mc_pred   - last_price, 2),
            "GCN":      round(gcn_pred  - last_price, 2),
            "ensemble": round(delta_ens, 2),
        },
        "signal": signal, "signal_pct": round(pct, 2),
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/forecast")
def forecast(days: int = 5):
    if STATE["cbam_model"] is None:
        raise HTTPException(503, detail="Models not ready")
    if not (1 <= days <= 30):
        raise HTTPException(400, detail="days must be 1-30")

    scaled_feat = STATE["scaled_feat"].copy()
    raw_prices  = list(STATE["raw_prices"])
    cbam_dm, cbam_ds = STATE["cbam_stats"]
    mc_dm,   mc_ds   = STATE["mc_stats"]
    gcn_dm,  gcn_ds  = STATE["gcn_stats"]

    result = []
    for i in range(days):
        cbam_p = reconstruct(predict_delta(STATE["cbam_model"], scaled_feat, WINDOW_CBAM),
                             cbam_dm, cbam_ds, raw_prices[-1])
        mc_p   = reconstruct(predict_delta(STATE["mc_model"],   scaled_feat, WINDOW_MC),
                             mc_dm,   mc_ds,   raw_prices[-1])
        gcn_p  = reconstruct(predict_delta(STATE["gcn_model"],  scaled_feat, WINDOW_GCN),
                             gcn_dm,  gcn_ds,  raw_prices[-1])
        ens_p  = (cbam_p + mc_p + gcn_p) / 3
        last_p = raw_prices[-1]
        result.append({
            "day": i+1,
            "CBAM_CNN": round(cbam_p, 2),
            "MC_CNN":   round(mc_p,   2),
            "GCN":      round(gcn_p,  2),
            "ensemble": round(ens_p,  2),
            "pct_change": round((ens_p - last_p) / last_p * 100, 2),
        })
        new_row    = scaled_feat[-1].copy()
        pr         = scaled_feat[:,0].max() - scaled_feat[:,0].min() + 1e-8
        new_row[0] = (ens_p - scaled_feat[:,0].min()) / pr
        scaled_feat = np.vstack([scaled_feat, new_row])
        raw_prices.append(ens_p)

    return {
        "current_price": round(STATE["raw_prices"][-1], 2),
        "forecast": result,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/refresh")
def refresh_data():
    scaled_feat, raw_prices = fetch_and_process()
    STATE["scaled_feat"] = scaled_feat
    STATE["raw_prices"]  = raw_prices
    return {
        "status": "data refreshed",
        "latest_price": round(float(raw_prices[-1]), 2),
        "timestamp": datetime.now().isoformat(),
    }