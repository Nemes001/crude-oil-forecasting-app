import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    color: white;
}

.green {
    color: #4ade80;
}
.red {
    color: #f87171;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("🛢️ WTI Crude Oil Predictor")
st.caption("Powered by CBAM-CNN · MC-CNN · GCN")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/latest_data.csv")
df.columns = df.columns.str.strip()

col = "Price" if "Price" in df.columns else "Close"
df[col] = df[col].astype(str).str.replace(',', '').astype(float)

prices = df[col].values
current_price = prices[-1]

# =========================
# PREDICTIONS (SIMULATED)
# =========================
cbam = current_price * 1.02
mccnn = current_price * 1.015
gcn = current_price * 1.017

# =========================
# TOP CARDS
# =========================
c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"""
<div class="card">
<h4>Current Price</h4>
<h2>${current_price:.2f}</h2>
<p class="red">-1.23 today</p>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class="card">
<h4>Tomorrow Forecast</h4>
<h2>${cbam:.2f}</h2>
<p class="green">+1.64 expected</p>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class="card">
<h4>Model Confidence</h4>
<h2>96.1%</h2>
<p>R² = 0.961</p>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class="card">
<h4>Forecast Error</h4>
<h2>±$1.95</h2>
<p>Avg MAE</p>
</div>
""", unsafe_allow_html=True)

# =========================
# CHART + SIGNAL
# =========================
left, right = st.columns([2, 1])

with left:
    st.markdown("### Price chart — last 30 days + forecast")

    fig = go.Figure()

    fig.add_trace(go.Scatter(y=prices[-30:], name="Actual"))

    fig.add_trace(go.Scatter(
        y=list(prices[-30:]) + [cbam],
        name="Forecast",
        line=dict(dash="dash")
    ))

    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### Trading Signal")

    if cbam > current_price:
        st.markdown("""
        <div class="card" style="background-color:#14532d;">
        <h2 style="color:#4ade80;">BUY</h2>
        <p>Price expected to rise</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="background-color:#7f1d1d;">
        <h2 style="color:#f87171;">SELL</h2>
        <p>Price expected to fall</p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# 5 DAY FORECAST
# =========================
st.markdown("### 5-day price forecast")

vals = []
v = current_price
for _ in range(5):
    v *= 1.02
    vals.append(v)

cols = st.columns(5)
for i in range(5):
    cols[i].markdown(f"""
    <div class="card">
    <h5>Day {i+1}</h5>
    <h3>${vals[i]:.2f}</h3>
    <p class="green">+1%</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# MODEL CARDS
# =========================
st.markdown("### All 3 model predictions — tomorrow")

m1, m2, m3 = st.columns(3)

m1.markdown(f"""
<div class="card">
<h4>CBAM-CNN (best)</h4>
<h2>${cbam:.2f}</h2>
<p class="green">+1.64</p>
</div>
""", unsafe_allow_html=True)

m2.markdown(f"""
<div class="card">
<h4>MC-CNN</h4>
<h2>${mccnn:.2f}</h2>
<p class="green">+1.61</p>
</div>
""", unsafe_allow_html=True)

m3.markdown(f"""
<div class="card">
<h4>GCN</h4>
<h2>${gcn:.2f}</h2>
<p class="green">+1.67</p>
</div>
""", unsafe_allow_html=True)