---
title: WTI Crude Oil Predictor API
emoji: 🛢
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🛢 WTI Crude Oil Price Predictor

A deep learning system for WTI crude oil price forecasting using three architectures — CBAM-CNN, Multi-Channel CNN, and Graph Convolutional Network — served via a FastAPI backend and a Streamlit frontend.

> **Motivation:** Escalating geopolitical tensions, OPEC+ supply decisions, and macroeconomic uncertainties have introduced unprecedented volatility into global energy markets, making accurate crude oil price forecasting more critical than ever.

---

## Project structure

```
crude-oil-forecasting-app/
├── backend/
│   ├── api.py                        ← FastAPI server
│   ├── cbam_cnn_pure.py              ← CBAM-CNN training script
│   ├── mc_cnn_crude_oil.py           ← MC-CNN training script
│   ├── gcn_tensorflow.py             ← GCN training script
│   ├── crude_oil_preprocessing.py    ← data pipeline
│   ├── models/
│   │   ├── cbam_cnn.keras
│   │   ├── cbam_cnn_delta_stats.npy
│   │   ├── mc_cnn.keras
│   │   ├── mc_cnn_delta_stats.npy
│   │   ├── gcn_tf_weights.weights.h5
│   │   ├── gcn_tf_adj.npy
│   │   └── gcn_tf_delta_stats.npy
│   └── requirements.txt
│
├── frontend/
│   ├── streamlit_app.py              ← Streamlit dashboard
│   └── requirements.txt
│
├── notebooks/
│   ├── EDA.ipynb                     ← exploratory data analysis
│   └── model_comparison.ipynb        ← model comparison
│
├── hf_app.py                         ← Hugging Face Spaces entry point
├── Dockerfile                        ← HF Docker deployment
├── requirements.txt                  ← HF dependencies
├── .gitignore
└── README.md
```

---

## Models

| Model | Architecture | R² | MAE |
|---|---|---|---|
| CBAM-CNN | Dilated causal CNN + channel & spatial attention | 0.9592 | $1.95/bbl |
| MC-CNN | 4 parallel CNN branches (kernel 3, 7, 14, 21) | 0.9590 | $1.95/bbl |
| GCN | Graph convolutional network (Kipf & Welling) | 0.9589 | $1.95/bbl |

All models use a **price delta prediction strategy** — predicting the next-day price change rather than the absolute price, then reconstructing the final price as:
```
predicted_price = last_known_price + predicted_delta
```

---

## Deployment

### Option 1 — Hugging Face Spaces (backend) + Streamlit Cloud (frontend)

The backend is deployed on Hugging Face Spaces via Docker:
- **API:** `https://your-username-crude-oil-predictor-api.hf.space`
- **Dashboard:** `https://your-app.streamlit.app`

### Option 2 — Run locally

**Backend:**
```bash
conda create -n oilapi python=3.10 -y
conda activate oilapi
pip install tensorflow==2.17.0 fastapi uvicorn yfinance scikit-learn pandas matplotlib
cd backend
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
pip install streamlit plotly requests yfinance
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check API status and model load time |
| `/predict` | GET | Tomorrow's price from all 3 models |
| `/forecast?days=5` | GET | N-day rolling forecast (max 30) |
| `/refresh` | GET | Refresh market data without reloading weights |

### Example response — `/predict`

```json
{
  "current_price": 72.48,
  "predictions": {
    "CBAM_CNN": 74.12,
    "MC_CNN":   74.09,
    "GCN":      74.15,
    "ensemble": 74.12
  },
  "deltas": {
    "CBAM_CNN": 1.64,
    "MC_CNN":   1.61,
    "GCN":      1.67,
    "ensemble": 1.64
  },
  "signal":     "BUY",
  "signal_pct": 2.26,
  "timestamp":  "2026-05-08T09:30:00"
}
```

---

## Data

- **Source:** Yahoo Finance via `yfinance` (ticker: `CL=F`)
- **Period:** 2000 – 2026 (~6,500 daily observations)
- **Features:** Close, Open, High, Low, Volume, Change%, MA4, MA12, Momentum, Volatility, HL Spread

---

## Tech stack

| Layer | Technology |
|---|---|
| Models | TensorFlow 2.17 / Keras |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Deployment | Hugging Face Spaces (Docker) + Streamlit Cloud |
| Data | yfinance + pandas |

---

## Related

- Research paper repo: [crude-oil-research](https://github.com/your-username/crude-oil-research)

---

## Disclaimer

This project is for research and educational purposes only. Predictions are not financial advice.