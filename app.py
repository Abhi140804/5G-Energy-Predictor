import os
import warnings
import threading
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── Globals populated by background thread ────────────────────────────────────
model        = None
FEATURE_COLS = []
ALL_BS       = []
raw          = None
train_r2     = 0.0
test_r2      = 0.0
n_samples    = 0
n_features   = 0
model_ready  = False

def load_and_train():
    global model, FEATURE_COLS, ALL_BS, raw
    global train_r2, test_r2, n_samples, n_features, model_ready

    print("[loading] Reading dataset…")
    CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "5G_energy_consumption_dataset.csv")
    raw = pd.read_csv(CSV_PATH)

    # Time feature engineering
    raw["Time"]  = pd.to_datetime(raw["Time"])
    raw["Year"]  = raw["Time"].dt.year
    raw["Month"] = raw["Time"].dt.month
    raw["Day"]   = raw["Time"].dt.day
    raw["Hour"]  = raw["Time"].dt.hour
    raw.drop("Time", axis=1, inplace=True)
    raw.drop_duplicates(inplace=True)

    ALL_BS = sorted(raw["BS"].unique().tolist())

    # One-hot encode
    data_enc = pd.get_dummies(raw, drop_first=True, dtype=np.int8)
    X = data_enc.drop("Energy", axis=1)
    y = data_enc["Energy"]
    FEATURE_COLS = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    print("[training] Fitting LinearRegression…")
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_r2   = model.score(X_train, y_train)
    test_r2    = model.score(X_test,  y_test)
    n_samples  = len(raw)
    n_features = len(FEATURE_COLS)
    model_ready = True
    print(f"[ready] Train R²={train_r2:.4f}  Test R²={test_r2:.4f}")

# Start training in background so Flask serves immediately
threading.Thread(target=load_and_train, daemon=True).start()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        bs_options=ALL_BS if model_ready else [],
        train_r2=round(train_r2, 4),
        test_r2=round(test_r2, 4),
        n_features=n_features,
        n_samples=n_samples,
        model_ready=model_ready,
    )

@app.route("/status")
def status():
    return jsonify({"ready": model_ready})

@app.route("/predict", methods=["POST"])
def predict():
    if not model_ready:
        return jsonify({"error": "Model is still loading, please wait…"}), 503

    body = request.get_json(force=True)
    try:
        bs_val  = body["bs"]
        load    = float(body["load"])
        esmode  = float(body["esmode"])
        txpower = float(body["txpower"])
        year    = int(body["year"])
        month   = int(body["month"])
        day     = int(body["day"])
        hour    = int(body["hour"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    row = {col: 0 for col in FEATURE_COLS}
    row["load"]    = load
    row["ESMODE"]  = esmode
    row["TXpower"] = txpower
    row["Year"]    = year
    row["Month"]   = month
    row["Day"]     = day
    row["Hour"]    = hour

    bs_col = f"BS_{bs_val}"
    if bs_col in row:
        row[bs_col] = 1

    X_input = pd.DataFrame([row])[FEATURE_COLS]
    pred    = float(model.predict(X_input)[0])
    return jsonify({"prediction": round(pred, 4)})


@app.route("/stats")
def stats():
    if not model_ready:
        return jsonify({"error": "loading"}), 503
    hourly  = raw.groupby("Hour")["Energy"].mean().round(4).to_dict()
    monthly = raw.groupby("Month")["Energy"].mean().round(4).to_dict()
    return jsonify({
        "hourly_avg":  hourly,
        "monthly_avg": monthly,
        "bs_options":  ALL_BS,
        "train_r2":    round(train_r2, 4),
        "test_r2":     round(test_r2,  4),
        "n_samples":   n_samples,
        "n_features":  n_features,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5050, use_reloader=False)
