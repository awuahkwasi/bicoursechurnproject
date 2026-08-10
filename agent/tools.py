"""
tools.py
================================================================
Defines the "tools" (functions) that the AI Agent can call.
Each tool wraps the trained churn model (via src/predict.py) or
the dataset so the agent can answer business questions with real
numbers instead of guessing.

This file is PROVIDED and working. You are encouraged to add your
own tools here (see the "YOUR TASK" section at the bottom) to
extend what the agent can do - that is part of your project work.
================================================================
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from predict import ChurnPredictor  # noqa: E402

_predictor = None
_scored_cache = None


def _get_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor


def _get_scored_data() -> pd.DataFrame:
    global _scored_cache
    if _scored_cache is None:
        _scored_cache = _get_predictor().load_full_dataset_scored()
    return _scored_cache


# ----------------------------------------------------------------
# Tool 1: Predict churn for a single customer by ID
# ----------------------------------------------------------------
def predict_churn_for_customer(customer_id: str) -> dict:
    """Look up a customer by ID and return their churn risk."""
    df = _get_scored_data()
    row = df[df["customer_id"].str.upper() == customer_id.upper()]
    if row.empty:
        return {"error": f"Customer '{customer_id}' not found."}
    r = row.iloc[0]
    return {
        "customer_id": r["customer_id"],
        "actual_churn_label": r["churn"],
        "churn_probability": float(r["churn_probability"]),
        "risk_tier": r["risk_tier"],
        "tenure_months": int(r["tenure_months"]),
        "monthly_charges": float(r["monthly_charges"]),
        "contract_type": r["contract_type"],
        "customer_satisfaction_score": float(r["customer_satisfaction_score"]),
    }


# ----------------------------------------------------------------
# Tool 2: Get aggregate churn risk statistics
# ----------------------------------------------------------------
def get_churn_summary_stats() -> dict:
    """Return overall churn statistics across the whole customer base."""
    df = _get_scored_data()
    return {
        "total_customers": int(len(df)),
        "actual_churn_rate_pct": round((df["churn"] == "Yes").mean() * 100, 2),
        "predicted_high_risk_customers": int((df["risk_tier"] == "High Risk").sum()),
        "predicted_medium_risk_customers": int((df["risk_tier"] == "Medium Risk").sum()),
        "predicted_low_risk_customers": int((df["risk_tier"] == "Low Risk").sum()),
        "avg_monthly_charges_high_risk": round(
            df.loc[df["risk_tier"] == "High Risk", "monthly_charges"].mean(), 2
        ),
        "avg_tenure_months_high_risk": round(
            df.loc[df["risk_tier"] == "High Risk", "tenure_months"].mean(), 2
        ),
    }


# ----------------------------------------------------------------
# Tool 3: List top N highest-risk customers
# ----------------------------------------------------------------
def get_top_risk_customers(top_n: int = 10) -> list:
    """Return the N customers with the highest predicted churn probability."""
    df = _get_scored_data()
    top = df.sort_values("churn_probability", ascending=False).head(top_n)
    cols = ["customer_id", "churn_probability", "risk_tier", "contract_type",
            "monthly_charges", "tenure_months", "customer_service_calls"]
    return top[cols].to_dict(orient="records")


# ----------------------------------------------------------------
# Tool 4: Churn breakdown by a given categorical column
# ----------------------------------------------------------------
def get_churn_rate_by_segment(column: str) -> dict:
    """Return the churn rate (%) grouped by a categorical column,
    e.g. 'contract_type', 'internet_service', 'payment_method'."""
    df = _get_scored_data()
    if column not in df.columns:
        return {"error": f"Column '{column}' not found in dataset."}
    grouped = df.groupby(column)["churn"].apply(lambda x: round((x == "Yes").mean() * 100, 2))
    return grouped.to_dict()


# ----------------------------------------------------------------
# Tool 5: Predict churn for a hypothetical / new customer profile
# ----------------------------------------------------------------
def predict_churn_for_new_profile(profile: dict) -> dict:
    """Predict churn risk for a hypothetical customer described by a
    dict of feature values, e.g.
    {"tenure_months": 3, "contract_type": "Month-to-Month",
     "monthly_charges": 95, "customer_service_calls": 5, ...}
    Any feature not provided will need a reasonable default from the caller.
    """
    predictor = _get_predictor()
    return predictor.predict_one(profile)

# ----------------------------------------------------------------
# Tool 6: Estimate total monthly revenue at risk
# ----------------------------------------------------------------
def estimate_revenue_at_risk() -> dict:
    """
    Estimate the monthly recurring revenue at risk by summing the
    monthly charges of all customers classified as High Risk.
    """
    df = _get_scored_data()

    high_risk = df[df["risk_tier"] == "High Risk"]

    return {
        "high_risk_customers": int(len(high_risk)),
        "total_monthly_revenue_at_risk": round(
            float(high_risk["monthly_charges"].sum()), 2
        ),
        "average_monthly_charge_high_risk": round(
            float(high_risk["monthly_charges"].mean()), 2
        ),
    }

TOOL_REGISTRY = {
    "predict_churn_for_customer": predict_churn_for_customer,
    "get_churn_summary_stats": get_churn_summary_stats,
    "get_top_risk_customers": get_top_risk_customers,
    "get_churn_rate_by_segment": get_churn_rate_by_segment,
    "predict_churn_for_new_profile": predict_churn_for_new_profile,
    "estimate_revenue_at_risk": estimate_revenue_at_risk,
}

# ================================================================
# YOUR TASK (part of the graded project):
# Add at least one new tool of your own. Ideas:
#   - recommend_retention_action(customer_id) -> a rule-based or
#     LLM-generated retention recommendation for a specific customer
#   - compare_segments(column, metric) -> compare two segments side by side
#   - estimate_revenue_at_risk() -> sum(monthly_charges) for High Risk customers
# Remember to register any new tool in TOOL_REGISTRY above, AND add
# its JSON schema in churn_agent.py so Claude knows it exists.
# ================================================================
