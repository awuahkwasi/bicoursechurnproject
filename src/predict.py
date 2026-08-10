"""
predict.py
================================================================
Shared prediction utilities. Loads the trained model artifacts
produced by train_model.py and exposes simple, reusable functions
that both the AI Agent (agent/) and the Streamlit app (app/) call.

This file is COMPLETE and PROVIDED. Do not modify the ML logic.
================================================================
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "customer_churn_data.csv"


class ChurnPredictor:
    """Wraps the trained model + preprocessing artifacts for easy reuse."""

    def __init__(self):
        self.model = joblib.load(MODELS_DIR / "churn_model.pkl")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        self.label_encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
        with open(MODELS_DIR / "feature_names.json") as f:
            self.feature_names = json.load(f)
        with open(MODELS_DIR / "model_metrics.json") as f:
            self.metrics = json.load(f)

    def _encode_row(self, row: dict) -> pd.DataFrame:
        """Turn a raw customer dict into a model-ready single-row DataFrame."""
        data = {}
        for col in self.feature_names:
            value = row.get(col)
            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Fall back to the most frequent class if an unseen value is passed
                if value not in le.classes_:
                    value = le.classes_[0]
                data[col] = le.transform([str(value)])[0]
            else:
                data[col] = value
        return pd.DataFrame([data], columns=self.feature_names)

    def predict_one(self, row: dict) -> dict:
        """Predict churn probability for a single customer (dict of raw feature values)."""
        X = self._encode_row(row)
        X_scaled = self.scaler.transform(X)
        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        pred = int(proba >= 0.5)
        return {
            "churn_probability": round(proba, 4),
            "churn_prediction": "Yes" if pred == 1 else "No",
            "risk_tier": self._risk_tier(proba),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict churn probability for a DataFrame of raw customers."""
        df = df.copy()
        X = pd.DataFrame(index=df.index)
        for col in self.feature_names:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                col_values = df[col].astype(str)
                col_values = col_values.where(col_values.isin(le.classes_), le.classes_[0])
                X[col] = le.transform(col_values)
            else:
                col_values = df[col]
                if col_values.isna().any():
                    col_values = col_values.fillna(col_values.median())
                X[col] = col_values
        X_scaled = self.scaler.transform(X[self.feature_names])
        proba = self.model.predict_proba(X_scaled)[:, 1]
        df["churn_probability"] = proba.round(4)
        df["churn_prediction"] = np.where(proba >= 0.5, "Yes", "No")
        df["risk_tier"] = [self._risk_tier(p) for p in proba]
        return df

    @staticmethod
    def _risk_tier(proba: float) -> str:
        if proba >= 0.66:
            return "High Risk"
        elif proba >= 0.33:
            return "Medium Risk"
        return "Low Risk"

    def load_full_dataset_scored(self) -> pd.DataFrame:
        """Convenience: load data/customer_churn_data.csv and score every customer."""
        df = pd.read_csv(DATA_PATH)
        return self.predict_batch(df)


if __name__ == "__main__":
    # Quick manual test
    predictor = ChurnPredictor()
    scored = predictor.load_full_dataset_scored()
    print(scored[["customer_id", "churn", "churn_probability", "risk_tier"]].head(10))
    print(f"\nBest model in use: {predictor.metrics['best_model']}")
