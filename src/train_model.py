"""
train_model.py
================================================================
Customer Churn Prediction - Model Training Pipeline
================================================================
This script is COMPLETE and PROVIDED to you. You do not need to
write or modify any machine learning code here. Your job is to:
  1. Run this script once to generate the trained model artifacts
  2. Use those artifacts inside the AI Agent (agent/) and the
     Streamlit dashboard (app/)

Run it with:
    python src/train_model.py

It will create the following files inside models/:
    churn_model.pkl        -> trained classifier (best model)
    scaler.pkl              -> fitted StandardScaler
    label_encoders.pkl      -> fitted LabelEncoders for categorical cols
    feature_names.json      -> ordered list of feature columns
    model_metrics.json      -> evaluation metrics for every model tried
================================================================
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "customer_churn_data.csv"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COL = "churn"
ID_COL = "customer_id"


def load_data() -> pd.DataFrame:
    print("=" * 60)
    print("  STEP 1: DATA LOADING")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Churn distribution:\n{df[TARGET_COL].value_counts()}\n")
    return df


def preprocess(df: pd.DataFrame):
    print("=" * 60)
    print("  STEP 2: PREPROCESSING")
    print("=" * 60)
    df_model = df.copy()

    # Drop identifier column
    if ID_COL in df_model.columns:
        df_model = df_model.drop(columns=[ID_COL])

    # Fill missing values
    num_cols = df_model.select_dtypes(include=np.number).columns
    cat_cols = df_model.select_dtypes(include="object").columns.drop(TARGET_COL, errors="ignore")

    for col in num_cols:
        df_model[col] = df_model[col].fillna(df_model[col].median())
    for col in cat_cols:
        df_model[col] = df_model[col].fillna(df_model[col].mode()[0])

    # Encode target
    df_model[TARGET_COL] = df_model[TARGET_COL].map({"Yes": 1, "No": 0})

    # Encode categorical features and keep encoders for reuse in production
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le

    feature_names = [c for c in df_model.columns if c != TARGET_COL]
    X = df_model[feature_names]
    y = df_model[TARGET_COL]

    print(f"Final feature count: {len(feature_names)}")
    print("Preprocessing complete.\n")
    return X, y, feature_names, label_encoders


def train_and_evaluate(X, y):
    print("=" * 60)
    print("  STEP 3: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42
        ),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        fitted_models[name] = model
        print(
            f"{name:22s} | Acc: {results[name]['accuracy']:.3f} | "
            f"F1: {results[name]['f1']:.3f} | "
            f"ROC-AUC: {results[name]['roc_auc']:.3f}"
        )

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_name]
    print(f"\nBest model selected: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})\n")

    return best_name, best_model, scaler, results


def save_artifacts(best_name, best_model, scaler, label_encoders, feature_names, results):
    print("=" * 60)
    print("  STEP 4: SAVING ARTIFACTS")
    print("=" * 60)

    joblib.dump(best_model, MODELS_DIR / "churn_model.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(label_encoders, MODELS_DIR / "label_encoders.pkl")

    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    metrics_out = {
        "best_model": best_name,
        "all_models": {k: {m: round(v2, 4) for m, v2 in v.items()} for k, v in results.items()},
    }
    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"Artifacts saved to: {MODELS_DIR}")
    print(" - churn_model.pkl")
    print(" - scaler.pkl")
    print(" - label_encoders.pkl")
    print(" - feature_names.json")
    print(" - model_metrics.json\n")


def main():
    df = load_data()
    X, y, feature_names, label_encoders = preprocess(df)
    best_name, best_model, scaler, results = train_and_evaluate(X, y)
    save_artifacts(best_name, best_model, scaler, label_encoders, feature_names, results)
    print("=" * 60)
    print("  DONE. Model ready for the agent and dashboard.")
    print("=" * 60)


if __name__ == "__main__":
    main()
