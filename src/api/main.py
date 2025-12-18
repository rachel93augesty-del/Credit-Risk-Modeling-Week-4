from fastapi import FastAPI
from src.api.pydantic_models import CreditRiskInput, CreditRiskOutput
import joblib
import os
import numpy as np


# -----------------------------
# 1. Initialize FastAPI app
# -----------------------------
app = FastAPI(title="Credit Risk Prediction API", version="1.0")


# -----------------------------
# 2. Load the best trained model
# -----------------------------
model_path = os.path.join("saved_models", "best_model.pkl")  # adjust if using Random_Forest_best_model.pkl

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

try:
    model = joblib.load(model_path)
    print(f"✅ Loaded model from {model_path} successfully!")
except Exception as e:
    raise RuntimeError(f"❌ Error loading the model: {e}")


# -----------------------------
# 3. Root endpoint
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Credit Risk Prediction API is running."}


# -----------------------------
# 4. Predict endpoint
# -----------------------------
@app.post("/predict", response_model=CreditRiskOutput)
def predict_risk(data: CreditRiskInput):
    # Convert input to numpy array for model
    X_new = np.array(
        [
            [
                data.Amount,
                data.Value,
                data.CurrencyCode,
                data.CountryCode,
                data.ProviderId,
                data.ProductCategory,
                data.ChannelId,
            ]
        ],
        dtype=object,
    )

    # Predict probability (if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        risk_prob = model.predict_proba(X_new)[:, 1][0]  # probability of class 1 (high risk)
    else:
        risk_prob = float(model.predict(X_new)[0])

    # Determine label
    risk_label = int(risk_prob >= 0.5)

    return CreditRiskOutput(risk_label=risk_label, risk_probability=risk_prob)
