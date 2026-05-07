# WTI Crude Oil Price Predictor

A deep learning system for WTI crude oil price forecasting using three architectures — CBAM-CNN, Multi-Channel CNN, and Graph Convolutional Network — served via a FastAPI backend and a Streamlit frontend.

---

## Project structure

```
crude-oil-predictor/
├── backend/
│   ├── api.py                        ← FastAPI server
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
│   ├── EDA.ipynb
│   └── model_comparison.ipynb
│
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

All models use a **price delta prediction strategy** — predicting the next-day price change rather than the absolute price, then reconstructing the final price as `predicted_price = last_price + predicted_delta`.

---

## Getting started

### Step 1 — Clone the repo

```bash
git clone https://github.com/your-username/crude-oil-predictor.git
cd crude-oil-predictor
```

### Step 2 — Train models and save weights

Before running the API, train each model locally and save the weights. Run the training scripts from your notebooks or terminal:

```python
# Inside your notebook — run preprocessing first
exec(open("crude_oil_preprocessing.py").read())

# Then run each model script
exec(open("cbam_cnn_pure.py").read())       # saves cbam_cnn.keras
exec(open("mc_cnn_crude_oil.py").read())    # saves mc_cnn.keras
exec(open("gcn_tensorflow.py").read())      # saves gcn_tf_weights.weights.h5
```

Move the generated weight files into `backend/models/`:

```
cbam_cnn.keras
cbam_cnn_delta_stats.npy
mc_cnn.keras
mc_cnn_delta_stats.npy
gcn_tf_weights.weights.h5
gcn_tf_adj.npy
gcn_tf_delta_stats.npy
```

### Step 3 — Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will start, load all model weights, fetch the latest WTI data from Yahoo Finance, and be ready in ~10 seconds.

### Step 4 — Start the frontend

Open a new terminal:

```bash
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open your browser at `http://localhost:8501`.

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
  "timestamp":  "2026-04-19T09:30:00"
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
| Frontend | Streamlit |
| Charts | Plotly |
| Data | yfinance + pandas |

---

## Related

- Research paper repo: [crude-oil-research](https://github.com/your-username/crude-oil-research)

---

## Disclaimer

This project is for research and educational purposes only. Predictions are not financial advice.