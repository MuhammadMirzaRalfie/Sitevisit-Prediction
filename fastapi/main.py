from fastapi import FastAPI
from app.schemas import SiteVisitInput
from app.utils import load_assets, encode_input
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="SiteVisit Prediction API", version="1.0")

# Load model dan encoder
model, encoders = load_assets()

@app.get("/")
def home():
    return {"message": "Welcome to SiteVisit Prediction API "}

@app.post("/predict")
def predict_sitevisit(input_data: SiteVisitInput):
    # Ubah input jadi DataFrame
    df = pd.DataFrame([input_data.dict()])

    # Encode data
    encoded_df = encode_input(df, encoders)

    # Prediksi
    pred = model.predict(encoded_df)[0]
    prob = float(model.predict_proba(encoded_df)[0][1])

    return {
        "prediction": int(pred),
        "probability": prob,
        "interpretation": "Needs Site Visit" if pred == 1 else "No Site Visit Needed"
    }
