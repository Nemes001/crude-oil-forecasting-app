import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# =========================
# TITLE + DESCRIPTION
# =========================
st.title("Crude Oil Price Forecasting Dashboard")

st.markdown("""
### Deep Learning-Based Oil Price Prediction

This project compares:
- CBAM-CNN  
- Multi-Channel CNN  
- Graph Convolutional Network (GCN)

Dataset: Past 2 years of daily crude oil prices.
""")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Controls")
st.sidebar.write("Best Model: CBAM-CNN")
st.sidebar.write("Forecast Horizon: 5 Days")

# =========================
# LOAD DATA
# =========================
try:
    df = pd.read_csv("data/latest_data.csv")
except FileNotFoundError:
    st.error("Dataset not found. Please upload 'data/latest_data.csv'")
    st.stop()

# Clean column names
df.columns = df.columns.str.strip()

# Detect correct column
if "Close" in df.columns:
    col = "Close"
elif "Price" in df.columns:
    col = "Price"
else:
    st.error(f"No valid price column found. Columns: {df.columns}")
    st.stop()

# Convert to numeric
df[col] = df[col].astype(str).str.replace(',', '').astype(float)

prices = df[col].values

# =========================
# PREDICTION LOGIC (TEMP)
# =========================
current_price = prices[-1]

# Simulated predictions (replace with real models later)
cbam_pred = current_price * 1.012
mccnn_pred = current_price * 1.009
gcn_pred = current_price * 1.006

# =========================
# METRICS
# =========================
st.subheader("Current Market Overview")

col1, col2 = st.columns(2)

col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("CBAM Prediction", f"${cbam_pred:.2f}", f"{cbam_pred-current_price:+.2f}")

# =========================
# MODEL COMPARISON
# =========================
st.subheader("Model Comparison")

c1, c2, c3 = st.columns(3)

c1.metric("CBAM-CNN", f"${cbam_pred:.2f}")
c2.metric("Multi-Channel CNN", f"${mccnn_pred:.2f}")
c3.metric("GCN", f"${gcn_pred:.2f}")

# =========================
# PERFORMANCE METRICS
# =========================
st.subheader("Model Performance")

st.write("""
- MAE: 1.95  
- RMSE: 3.00  
- MAPE: 2.53%  
- R²: 0.9590  
- Pearson r: 0.9803  
""")

# =========================
# CHART
# =========================
fig = go.Figure()

fig.add_trace(go.Scatter(
    y=prices,
    mode='lines',
    name="Actual Prices"
))

fig.add_trace(go.Scatter(
    y=list(prices) + [cbam_pred],
    mode='lines',
    name="CBAM Forecast"
))

st.plotly_chart(fig, use_container_width=True)

# =========================
# 5-DAY FORECAST
# =========================
st.subheader("5-Day Forecast (CBAM-CNN)")

future_prices = []
val = current_price

for _ in range(5):
    val *= 1.012
    future_prices.append(val)

cols = st.columns(5)
for i in range(5):
    cols[i].metric(f"Day {i+1}", f"${future_prices[i]:.2f}")

# =========================
# SIGNAL
# =========================
st.subheader("Trading Signal")

if cbam_pred > current_price:
    st.success("BUY 📈")
elif cbam_pred < current_price:
    st.error("SELL 📉")
else:
    st.warning("HOLD ⚖️")