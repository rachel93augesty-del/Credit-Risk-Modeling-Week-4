# src/api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import numpy as np

# -------------------------------
# 1. Initialize app
# -------------------------------
app = FastAPI(
    title="Credit Risk Prediction API",
    version="1.0",
    description="Predict high/low credit risk based on input features"
)

# -------------------------------
# 2. Define input schema
# -------------------------------
class CreditRiskInput(BaseModel):
    Amount: float
    Value: float
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductCategory: str
    ChannelId: str

# -------------------------------
# 3. Load the trained model
# -------------------------------
MODEL_PATH = os.path.join("saved_models", "best_model.pkl")  # adjust if needed
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# -------------------------------
# 4. Root endpoint
# -------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to Credit Risk Prediction API"}

# -------------------------------
# 5. Predict endpoint
# -------------------------------
@app.post("/predict")
def predict_credit_risk(input_data: CreditRiskInput):
    if model is None:
        return {"error": "Model not loaded."}

    # Convert input to array for model
    X = np.array([[
        input_data.Amount,
        input_data.Value,
        input_data.CurrencyCode,
        input_data.CountryCode,
        input_data.ProviderId,
        input_data.ProductCategory,
        input_data.ChannelId
    ]])

    # Make prediction
    try:
        prediction = model.predict(X)[0]
        prediction_proba = (
            model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
        )

        result = {"prediction": int(prediction)}
        if prediction_proba is not None:
            result["probability"] = float(prediction_proba)
        return result

    except Exception as e:
        return {"error": f"Prediction failed: {e}"}
