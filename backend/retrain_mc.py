import os, shutil
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
os.chdir(r"D:\GITDEMO\crude-oil-forecasting-app\backend")

# ── Preprocessing using yfinance ──────────────────────────────────────────────
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

features   = ["Close","Open","High","Low","Volume","Change_pct",
              "MA4","MA12","Momentum","Volatility","HL_Spread"]
raw_prices = df["Close"].values.astype(np.float32)
raw_feat   = df[features].values.astype(np.float32)
feat_scaler  = MinMaxScaler()
tgt_scaler   = MinMaxScaler()
scaled_feat  = feat_scaler.fit_transform(raw_feat).astype(np.float32)
scaled_target = tgt_scaler.fit_transform(raw_prices.reshape(-1,1)).flatten()
print(f"Preprocessing done: {len(df)} samples")

# ── MC-CNN ────────────────────────────────────────────────────────────────────
exec(open("mc_cnn_crude_oil.py", encoding="utf-8").read())

# ── Copy weights ──────────────────────────────────────────────────────────────
shutil.copy("mc_cnn.keras",           r"models\mc_cnn.keras")
shutil.copy("mc_cnn_delta_stats.npy", r"models\mc_cnn_delta_stats.npy")
print("MC-CNN weights copied to models/")