"""
Crude Oil WTI — Data Preprocessing
=====================================
Run this before the GCN model script.
Outputs: scaled_feat, scaled_target, tgt_scaler, features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ─── 1. Load ──────────────────────────────────────────────────────────────────

FILE_PATH = r"C:/Users/amees/Downloads/Crude Oil WTI Futures Historical Data.csv"

df = pd.read_csv(FILE_PATH)
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(df.head())

# ─── 2. Parse dates & sort ────────────────────────────────────────────────────

df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.sort_values("Date").reset_index(drop=True)
print(f"\nDate range: {df['Date'].min().date()} → {df['Date'].max().date()}")

# ─── 3. Clean Volume & Change % ───────────────────────────────────────────────

df["Vol_clean"] = (
    df["Vol."]
    .str.replace("M", "e6", regex=False)
    .str.replace("K", "e3", regex=False)
    .apply(lambda x: float(x))
)

df["Change_clean"] = (
    df["Change %"]
    .str.replace("%", "", regex=False)
    .astype(float)
)

# ─── 4. Feature engineering ───────────────────────────────────────────────────

# Moving averages
df["MA4"]  = df["Price"].rolling(window=4).mean()
df["MA12"] = df["Price"].rolling(window=12).mean()

# Price momentum (week-over-week change)
df["Momentum"] = df["Price"].diff()

# Volatility (rolling std)
df["Volatility"] = df["Price"].rolling(window=4).std()

# High-Low spread
df["HL_Spread"] = df["High"] - df["Low"]

# Drop NaN rows introduced by rolling windows
df = df.dropna().reset_index(drop=True)
print(f"After dropping NaN rows: {len(df)} samples remaining")

# ─── 5. Select features & target ─────────────────────────────────────────────

features = [
    "Price", "Open", "High", "Low",
    "Vol_clean", "Change_clean",
    "MA4", "MA12",
    "Momentum", "Volatility", "HL_Spread"
]
target_col = "Price"

print(f"\nFeatures ({len(features)}): {features}")

raw_feat   = df[features].values.astype(np.float32)
raw_target = df[target_col].values.astype(np.float32)

print(f"Price range: ${raw_target.min():.2f} – ${raw_target.max():.2f}")

# ─── 6. Scale ────────────────────────────────────────────────────────────────

feat_scaler = MinMaxScaler()
tgt_scaler  = MinMaxScaler()

scaled_feat   = feat_scaler.fit_transform(raw_feat)
scaled_target = tgt_scaler.fit_transform(raw_target.reshape(-1, 1)).flatten()

print(f"\nscaled_feat   : {scaled_feat.shape}  (min={scaled_feat.min():.2f}, max={scaled_feat.max():.2f})")
print(f"scaled_target : {scaled_target.shape}  (min={scaled_target.min():.2f}, max={scaled_target.max():.2f})")
print("\nPreprocessing complete. Ready for GCN model.")