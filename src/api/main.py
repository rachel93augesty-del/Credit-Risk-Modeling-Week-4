# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Credit Risk Prediction API")

# Load your trained model
model_path = "saved_models/best_model.pkl"  # update with your actual path
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define input schema with Pydantic
class CreditData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # add all features your model expects

@app.post("/predict")
def predict(data: CreditData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}
