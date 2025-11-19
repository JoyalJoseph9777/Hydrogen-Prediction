from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and preprocessors
model = tf.keras.models.load_model("pinn_model.h5", compile=False)
preprocessor = joblib.load("preprocessor.pkl")
y_scaler = joblib.load("y_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    additive = ""
    additiveC = ""
    milling_time = ""
    bpr = ""
    speed = ""
    temperature = "375"

    if request.method == "POST":
        additive = request.form["additive"]
        additiveC = request.form["additiveC"]
        milling_time = request.form["milling_time"]
        bpr = request.form["bpr"]
        speed = request.form["speed"]

        # Convert values
        additiveC_f = float(additiveC)
        milling_time_f = float(milling_time)
        bpr_f = int(bpr)
        speed_f = int(speed)
        temperature_f = 375.0   # FIXED

        # Derived features
        milling_energy = speed_f * bpr_f * milling_time_f
        energy_density = temperature_f * bpr_f / (milling_time_f + 1)
        additive_effect = additiveC_f * temperature_f / (speed_f + 1)

        # DataFrame for prediction
        df = pd.DataFrame([{
            "Additive": additive,
            "AdditiveC": additiveC_f,
            "MillingTime": milling_time_f,
            "BPR": bpr_f,
            "Speed": speed_f,
            "Temperature": temperature_f,
            "MillingEnergy": milling_energy,
            "EnergyDensity": energy_density,
            "AdditiveEffect": additive_effect
        }])

        df = df[preprocessor.feature_names_in_]

        X_processed = preprocessor.transform(df)
        pred_scaled = model.predict(X_processed)
        pred = y_scaler.inverse_transform(pred_scaled)[0][0]

        prediction = round(float(pred), 4)

    return render_template(
        "index.html",
        prediction=prediction,
        additive=additive,
        additiveC=additiveC,
        milling_time=milling_time,
        bpr=str(bpr),
        speed=str(speed),
        temperature="375"
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
