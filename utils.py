def load_model_file():
    return None

def predict(model, input_data):
    return float(input_data[0][-1][0] * 1.01)  # simple +1% prediction

def forecast_future(model, input_data, days=5):
    preds = []
    val = input_data[0][-1][0]

    for _ in range(days):
        val = val * 1.01
        preds.append(val)

    return preds