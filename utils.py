from tensorflow.keras.models import load_model
import numpy as np

def load_models():
    cbam = load_model("model/cbam_cnn.h5")
    mc = load_model("model/mc_cnn.h5")
    gcn = load_model("model/gcn.h5")
    return cbam, mc, gcn

def preprocess(data, window_size=30):
    return np.array(data[-window_size:]).reshape(1, window_size, 1)

def make_prediction(model, input_data):
    return float(model.predict(input_data)[0][0])

def forecast_future(model, data, days=5):
    predictions = []
    temp = data.copy()

    for _ in range(days):
        pred = model.predict(temp)[0][0]
        predictions.append(pred)
        temp = np.append(temp[:,1:,:], [[[pred]]], axis=1)

    return predictions

def generate_signal(current, predicted):
    if predicted > current:
        return "BUY"
    elif predicted < current:
        return "SELL"
    return "HOLD"