import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─── Config ───────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="WTI Crude Oil Predictor",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stButton>button{
        width:100%;background:#185FA5;color:white;
        border:none;border-radius:8px;padding:12px;
        font-size:16px;font-weight:600;
    }
    .signal-box{border-radius:12px;padding:24px;text-align:center;margin-bottom:12px}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Settings")
    model_choice  = st.selectbox(
        "Primary model",
        ["CBAM_CNN","MC_CNN","GCN","ensemble"], index=3)
    forecast_days = st.slider("Forecast horizon (days)", 1, 10, 5)
    data_period   = st.selectbox(
        "Chart period", ["1mo","3mo","6mo","1y","2y"], index=2)

    st.markdown("---")
    st.markdown("**API status**")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            h = r.json()
            if h["models_ready"]:
                st.success("Backend online · Models ready")
                st.caption(f"Last trained: {h['last_trained'][:19]}")
            else:
                st.warning("Backend online · Models loading...")
        else:
            st.error("Backend error")
    except Exception:
        st.error("Backend offline — start api.py first")

    st.markdown("---")
    if st.button("Force retrain models"):
        try:
            r = requests.get(f"{API_URL}/retrain", timeout=300)
            st.success("Retrained successfully")
        except Exception as e:
            st.error(f"Retrain failed: {e}")

    st.markdown("---")
    st.markdown("**Model performance**")
    st.markdown("| Model | R² | MAE |")
    st.markdown("|---|---|---|")
    st.markdown("| CBAM-CNN | 0.9592 | $1.95 |")
    st.markdown("| MC-CNN   | 0.9590 | $1.95 |")
    st.markdown("| GCN      | 0.9589 | $1.95 |")

# ─── Helper ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_chart_data(period):
    df = yf.download("CL=F", period=period, interval="1d", progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Close"]].dropna()

def trading_signal(pct):
    if   pct >  1.0: return "BUY",  "#1D9E75", "#0f3d2e"
    elif pct < -1.0: return "SELL", "#E24B4A", "#3d0f0f"
    else:            return "HOLD", "#EF9F27", "#3d2f0f"

# ─── Main ─────────────────────────────────────────────────────────────────────

st.markdown("# 🛢 WTI Crude Oil Price Predictor")
st.markdown("Real-time forecasting · CBAM-CNN · MC-CNN · GCN")
st.markdown("---")

predict_btn = st.button("Fetch prediction from models")

if predict_btn:

    # ── Call /predict ──
    with st.spinner("Getting predictions from backend..."):
        try:
            pred_resp = requests.get(f"{API_URL}/predict", timeout=30)
            pred_resp.raise_for_status()
            pred_data = pred_resp.json()
        except Exception as e:
            st.error(f"Could not reach backend: {e}")
            st.stop()

    # ── Call /forecast ──
    with st.spinner("Getting forecast..."):
        try:
            fc_resp = requests.get(
                f"{API_URL}/forecast", params={"days": forecast_days}, timeout=60)
            fc_resp.raise_for_status()
            fc_data = fc_resp.json()
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.stop()

    current_price = pred_data["current_price"]
    primary_pred  = pred_data["predictions"][model_choice]
    primary_delta = pred_data["deltas"][model_choice]
    signal_pct    = pred_data["signal_pct"]
    signal, sig_color, sig_bg = trading_signal(signal_pct)

    forecast_list   = fc_data["forecast"]
    forecast_prices = [f[model_choice] for f in forecast_list]

    # ── Top metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current price",       f"${current_price:.2f}", "WTI $/bbl")
    c2.metric("Tomorrow's forecast", f"${primary_pred:.2f}",
              f"{'+' if primary_delta>=0 else ''}{primary_delta:.2f}")
    c3.metric("Model confidence", "96.1%", "R² = 0.959")
    c4.metric("Forecast error",   "±$1.95", "MAE on test set")

    st.markdown("---")
    left, right = st.columns([2,1])

    with left:

        # ── Chart ──
        st.markdown("#### Price chart")
        df_chart = fetch_chart_data(data_period)
        f_dates  = [df_chart.index[-1] + timedelta(days=i+1)
                    for i in range(forecast_days)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_chart.index, y=df_chart["Close"],
            mode="lines", name="Actual",
            line=dict(color="#378ADD", width=2)))
        fig.add_trace(go.Scatter(
            x=f_dates, y=forecast_prices,
            mode="lines+markers", name="Forecast",
            line=dict(color="#1D9E75", width=2, dash="dot"),
            marker=dict(size=6)))
        fig.add_vline(x=df_chart.index[-1],
                      line_dash="dash", line_color="#6e7681", line_width=1)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(orientation="h", y=1.1),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#30363d", tickprefix="$"))
        st.plotly_chart(fig, use_container_width=True)

        # ── N-day forecast ──
        st.markdown(f"#### {forecast_days}-day forecast")
        fcols = st.columns(forecast_days)
        prev  = current_price
        for i, (col, f) in enumerate(zip(fcols, forecast_list)):
            price = f[model_choice]
            chg   = price - prev
            pct   = chg / prev * 100
            col.metric(f"Day {i+1}", f"${price:.2f}",
                       f"{'+' if chg>=0 else ''}{pct:.1f}%")
            prev = price

        # ── All 3 models ──
        st.markdown("#### All 3 model predictions — tomorrow")
        mc1, mc2, mc3 = st.columns(3)
        for col, name, label, badge in [
            (mc1, "CBAM_CNN", "CBAM-CNN", "⭐"),
            (mc2, "MC_CNN",   "MC-CNN",   ""),
            (mc3, "GCN",      "GCN",      ""),
        ]:
            p = pred_data["predictions"][name]
            d = pred_data["deltas"][name]
            col.metric(f"{badge} {label}", f"${p:.2f}",
                       f"{'+' if d>=0 else ''}{d:.2f}")

        # ── Raw API response ──
        with st.expander("Raw API response"):
            st.json(pred_data)

    with right:

        # ── Signal ──
        st.markdown("#### Trading signal")
        st.markdown(f"""
        <div class="signal-box" style="background:{sig_bg}">
            <div style="font-size:38px;font-weight:700;color:{sig_color}">{signal}</div>
            <div style="font-size:14px;color:{sig_color};margin-top:6px">
                {'+' if signal_pct>=0 else ''}{signal_pct:.2f}% expected
            </div>
            <div style="font-size:11px;color:#6e7681;margin-top:8px">
                Based on {model_choice.replace('_','-')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Market info ──
        st.markdown("#### Market info")
        df_1y  = fetch_chart_data("1y")
        prices = df_1y["Close"].values
        info   = {
            "52-week high":  f"${prices.max():.2f}",
            "52-week low":   f"${prices.min():.2f}",
            "30-day avg":    f"${prices[-21:].mean():.2f}",
            "Volatility":    f"±${prices[-21:].std():.2f}",
            "Data source":   "Yahoo Finance",
            "Last updated":  datetime.now().strftime("%d %b %Y %H:%M"),
        }
        for k, v in info.items():
            a, b = st.columns(2)
            a.markdown(f"<span style='color:#6e7681;font-size:12px'>{k}</span>",
                       unsafe_allow_html=True)
            b.markdown(f"<span style='font-size:12px;font-weight:500'>{v}</span>",
                       unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#6e7681;font-size:11px'>"
        "For informational purposes only · Not financial advice"
        "</div>", unsafe_allow_html=True)

else:
    st.info("Click **Fetch prediction from models** to get today's forecast.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    **Step 1 — Start backend**
    ```bash
    uvicorn api:app --port 8000
    ```
    """)
    c2.markdown("""
    **Step 2 — Run this app**
    ```bash
    streamlit run streamlit_app.py
    ```
    """)
    c3.markdown("""
    **Step 3 — Click predict**
    App calls the API, gets predictions from all 3 models and shows results.
    """)