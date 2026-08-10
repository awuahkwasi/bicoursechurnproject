# Architecture Overview

```
                     ┌─────────────────────────┐
                     │  customer_churn_data.csv │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   src/train_model.py     │  (provided, run once)
                     │  preprocess → train →    │
                     │  evaluate → select best  │
                     └────────────┬────────────┘
                                  │  saves
                                  ▼
        models/churn_model.pkl, scaler.pkl, label_encoders.pkl,
                 feature_names.json, model_metrics.json
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │   src/predict.py         │  (provided)
                     │   ChurnPredictor class    │
                     └───────┬─────────┬────────┘
                              │         │
              ┌───────────────┘         └────────────────┐
              ▼                                            ▼
 ┌─────────────────────────┐              ┌───────────────────────────┐
 │  agent/tools.py           │              │  app/streamlit_app.py      │
 │  business-logic functions │◄────calls────│  Dashboard tab (charts)    │
 └──────────┬────────────────┘              │  Agent tab (chat)          │
             │ registered as tools           └───────────────┬────────────┘
             ▼                                                │
 ┌─────────────────────────┐                                  │
 │  agent/churn_agent.py     │◄─────────── used by ────────────┘
 │  Claude tool-calling loop │
 │  (Anthropic API)          │
 └────────────────────────────┘

 Separately: powerbi/churn_dashboard_export.csv → Power BI Desktop
             → churn_dashboard.pbix (executive dashboard)

 Deployment: GitHub repo → Streamlit Community Cloud (app/streamlit_app.py)
```

## Data flow summary

1. **Training (offline, once):** `train_model.py` turns raw CSV data
   into a trained model + preprocessing artifacts.
2. **Serving (runtime):** `predict.py` loads those artifacts and
   exposes simple `predict_one()` / `predict_batch()` functions.
3. **Agent layer:** `tools.py` wraps those functions as named tools;
   `churn_agent.py` lets Claude decide when to call which tool based
   on the user's natural-language question.
4. **Presentation layer:** `streamlit_app.py` renders charts directly
   from the scored data, and a chat UI that talks to the agent.
5. **Power BI:** a separate, non-code presentation layer fed by the
   same scored data, aimed at business/executive reporting.
