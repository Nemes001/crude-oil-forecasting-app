import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Crude Oil Price Predictor")

# =========================
# LOAD DATA SAFELY
# =========================
try:
    df = pd.read_csv("data/latest_data.csv")
except FileNotFoundError:
    st.error("Dataset not found. Please upload 'data/latest_data.csv'")
    st.stop()

# Clean column names
df.columns = df.columns.str.strip()

# Detect correct price column
if "Close" in df.columns:
    col = "Close"
elif "Price" in df.columns:
    col = "Price"
else:
    st.error(f"No valid price column found. Columns: {df.columns}")
    st.stop()

# Convert to numeric (handles commas like "82,345")
df[col] = df[col].astype(str).str.replace(',', '').astype(float)

prices = df[col].values

# =========================
# BASIC PREDICTION LOGIC
# =========================
current_price = prices[-1]
prediction = current_price * 1.01  # simple +1% prediction

# =========================
# METRICS DISPLAY
# =========================
col1, col2 = st.columns(2)

col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Tomorrow Forecast", f"${prediction:.2f}", f"{prediction-current_price:+.2f}")

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
    y=list(prices) + [prediction],
    mode='lines',
    name="Forecast"
))

st.plotly_chart(fig, use_container_width=True)

# =========================
# 5-DAY FORECAST
# =========================
st.subheader("5-Day Forecast")

future_prices = []
val = current_price

for _ in range(5):
    val *= 1.01
    future_prices.append(val)

cols = st.columns(5)
for i in range(5):
    cols[i].metric(f"Day {i+1}", f"${future_prices[i]:.2f}")

# =========================
# SIGNAL
# =========================
st.subheader("Trading Signal")

if prediction > current_price:
    st.success("BUY 📈")
elif prediction < current_price:
    st.error("SELL 📉")
else:
    st.warning("HOLD ⚖️")