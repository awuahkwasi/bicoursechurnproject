# Student Guide — Step by Step

This guide walks you through the entire project from a fresh clone to a
deployed, working application. Follow the steps in order.

---

## 0. What you're building

- An **AI Agent** that can answer natural-language questions about
  customer churn using a trained ML model as its "eyes."
- A **Streamlit dashboard** that shows churn KPIs/charts and lets
  people chat with the agent.
- A **Power BI dashboard** for executive-style reporting.
- All of it pushed to **GitHub** and the Streamlit app **deployed
  publicly**.

You will **not** write machine learning code. That part (data
preprocessing, model training, evaluation, model selection) is done
for you in `src/train_model.py`. Your work is on the agent, the
dashboards, and shipping the project.

---

## 1. Accept the GitHub repository

1. Your instructor will share a GitHub Classroom link or an empty
   repository. Click it / accept it.
2. Clone it to your machine:
   ```bash
   git clone <the-repo-url-you-were-given>
   cd <repo-folder>
   ```
3. Copy all the files from this starter kit into that folder (or your
   instructor may have already placed them there).

---

## 2. Set up your Python environment

```bash
python --version        # make sure you have Python 3.10+
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Understand (but do not rewrite) the ML pipeline

Open `src/train_model.py` and read through it. It:

1. Loads `data/customer_churn_data.csv`
2. Cleans and encodes the data
3. Trains 4 candidate models and evaluates them (Accuracy, Precision,
   Recall, F1, ROC-AUC)
4. Picks the best model automatically
5. Saves everything the app needs into `models/`

Run it once:

```bash
python src/train_model.py
```

You should see console output ending in `DONE. Model ready for the
agent and dashboard.` and a new `models/` folder containing 5 files.

**Checkpoint:** open `models/model_metrics.json` and confirm you can
see metrics for all four models plus a `best_model` field.

---

## 4. Get an Anthropic API key

1. Go to https://console.anthropic.com/ and sign up / log in.
2. Create an API key.
3. In the project root, copy the template:
   ```bash
   cp .env.example .env
   ```
4. Open `.env` and paste your key:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
5. `.env` is already in `.gitignore` — **never commit your real key**.

---

## 5. Test the AI agent from the command line

```bash
python agent/churn_agent.py
```

Try asking:
- "How many customers do we have and what's the overall churn rate?"
- "Who are the top 5 highest risk customers?"
- "What's the churn rate by contract type?"
- "What's the risk for customer CUST-00006?"

Read `agent/tools.py` to see exactly what data each answer is coming
from. Read `agent/churn_agent.py` to see how tool calling works: the
agent sends your question + tool definitions to Claude, Claude
decides which tool(s) to call, the agent runs them and sends the
results back, and Claude writes the final natural-language answer.

### Your task on the agent

Add at least **one new tool** (ideas are listed at the bottom of
`agent/tools.py`, e.g. estimating total revenue at risk, or a
retention-action recommender). To add a tool:

1. Write the Python function in `tools.py` and add it to
   `TOOL_REGISTRY`.
2. Add a matching JSON schema entry to `TOOL_SCHEMAS` in
   `churn_agent.py` so Claude knows the tool exists and what
   arguments it takes.
3. Test it by asking the agent a question that should trigger it.

## How much monthly revenue is at risk from churn? ##

---

## 6. Run the Streamlit dashboard locally

```bash
streamlit run app/streamlit_app.py
```

This opens a browser tab with two tabs:
- **Dashboard** — KPIs and charts computed from the scored dataset
- **Ask the Agent** — chat interface wired to `churn_agent.py`

### Your task on the dashboard

Extend `app/streamlit_app.py`:
- Add at least 2 new charts or a working sidebar filter (by
  `contract_type`, `location`, or `risk_tier`)
- Adjust styling/layout to make it feel like your own product, not a
  copy of the starter

---

## 7. Build the Power BI dashboard

Follow `powerbi/POWER_BI_GUIDE.md`. In short:

1. Open Power BI Desktop
2. Import `powerbi/churn_dashboard_export.csv`
3. Build the pages described in the guide (Executive Overview,
   Customer Risk Explorer, Segment Deep Dive)
4. Save your `.pbix` file into the `powerbi/` folder
5. Take 2–3 screenshots for your report/README

---

## 8. Push your work to GitHub

```bash
git add .
git commit -m "Complete churn AI agent, dashboard, and Power BI report"
git push origin main
```

Double-check `.env` and `.streamlit/secrets.toml` were **not**
committed (run `git status` — they should not appear; `.gitignore`
handles this for you).

---

## 9. Deploy the Streamlit app publicly

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**, select your repository, branch `main`, and set
   the main file path to `app/streamlit_app.py`.
3. Before deploying, click **Advanced settings → Secrets** and paste:
   ```
   ANTHROPIC_API_KEY = "sk-ant-your-real-key"
   ```
4. Click **Deploy**. Wait for the build to finish.
5. Test the live URL — try both the Dashboard tab and the Agent tab.

**Checkpoint:** the deployed app should look and behave exactly like
it did locally, minus needing a local `.env` file (the secret is
pulled from Streamlit's secrets manager instead — this is what
`st.secrets.get(...)` in `streamlit_app.py` does for you).

---

## 10. Final submission checklist

- [ ] `models/` regenerated by running `train_model.py` and committed
      (or documented as regeneratable — check with your instructor
      whether binary model files should be committed or gitignored)
- [ ] At least one new agent tool added and working
- [ ] Streamlit dashboard extended with your own charts/filters
- [ ] Power BI `.pbix` file added to `powerbi/` with screenshots
- [ ] Public Streamlit Cloud URL working and included in your README
- [ ] Repository pushed to the GitHub Classroom repo you were given
- [ ] No API keys committed to GitHub

---

## Troubleshooting

**"No Anthropic API key found"** — check your `.env` file exists and
has `ANTHROPIC_API_KEY=` set with no quotes, and that you ran the app
from the project root.

**Streamlit Cloud app crashes on deploy** — check the app's logs in
the Streamlit Cloud dashboard. The most common cause is a missing
secret (step 9.3) or `models/` not being present in the repo — make
sure you committed the `models/` folder or add a build step that runs
`python src/train_model.py` before the app starts.

**Agent gives generic answers instead of real numbers** — this means
tool calling isn't triggering. Check `TOOL_SCHEMAS` names exactly
match the function names in `TOOL_REGISTRY`.
