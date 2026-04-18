import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("Crude Oil Price Predictor")

# Load data
df = pd.read_csv("data/latest_data.csv")
prices = df["Close"].values

# Simple prediction logic
current_price = prices[-1]
prediction = current_price * 1.01  # dummy +1%

# Metrics
col1, col2 = st.columns(2)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Tomorrow Forecast", f"${prediction:.2f}")

# Chart
fig = go.Figure()
fig.add_trace(go.Scatter(y=prices, name="Actual"))
fig.add_trace(go.Scatter(y=list(prices) + [prediction], name="Forecast"))

st.plotly_chart(fig, use_container_width=True)

# Signal
if prediction > current_price:
    st.success("BUY 📈")
else:
    st.error("SELL 📉")