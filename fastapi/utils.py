import joblib
import pandas as pd

def load_assets():
    model = joblib.load("XGboost_sitevisit.pkl")
    encoders = joblib.load("encoder.pkl")
    return model, encoders

def encode_input(data_df, encoders):
    df = data_df.copy()
    for col in df.columns:
        if col in encoders:
            le = encoders[col]
            classes = set([str(x) for x in le.classes_])
            df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in classes else -1)
    return df.fillna(-1)
