"""
streamlit_app.py
================================================================
Customer Churn Dashboard + AI Agent (Streamlit)
================================================================
Two tabs:
  1. Dashboard - churn KPIs and charts (provided, working)
  2. Ask the Agent - chat interface backed by agent/churn_agent.py

New: Upload your own CSV / Excel file to score and explore instead
of the built-in dataset. Invalid file types are rejected with a
clear message.

Run locally with:
    streamlit run app/streamlit_app.py

Deploy on Streamlit Community Cloud by pointing it at this file
after pushing your repo to GitHub (see STUDENT_GUIDE.md).

YOUR TASK: this file gives you a working baseline. You are expected
to customize/extend the dashboard (new charts, filters, styling) as
part of your deliverable - see the "YOUR TASK" markers below.
================================================================
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "agent"))

from predict import ChurnPredictor  # noqa: E402

st.set_page_config(
    page_title="Customer Churn AI Dashboard",
    page_icon="📉",
    layout="wide",
)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}


@st.cache_resource
def get_predictor():
    return ChurnPredictor()


@st.cache_data
def get_scored_data():
    predictor = get_predictor()
    return predictor.load_full_dataset_scored()


def read_uploaded_file(uploaded_file):
    """
    Read an uploaded CSV or Excel file into a DataFrame.
    Raises ValueError with a user-facing message on invalid input.
    """
    filename = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"'.{ext}' is not a valid file type. Please upload a CSV "
            f"(.csv) or Excel (.xlsx / .xls) file."
        )

    try:
        if ext == "csv":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        raise ValueError(f"Could not read '{filename}': {e}")

    if df.empty:
        raise ValueError(f"'{filename}' was read successfully but contains no rows.")

    return df


def score_uploaded_data(predictor, df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the uploaded dataframe through the churn model via
    ChurnPredictor.predict_batch(), which expects every column in
    predictor.feature_names to be present with raw (un-encoded) values.
    """
    missing_features = [c for c in predictor.feature_names if c not in df.columns]
    if missing_features:
        raise ValueError(
            "The uploaded file is missing column(s) the model needs: "
            + ", ".join(missing_features)
            + ". Required columns: "
            + ", ".join(predictor.feature_names)
        )

    df = df.copy()

    # customer_id isn't a model feature but the dashboard displays it -
    # synthesize one if it's absent so nothing downstream breaks.
    if "customer_id" not in df.columns:
        df.insert(0, "customer_id", [f"UPLOAD-{i+1:05d}" for i in range(len(df))])

    return predictor.predict_batch(df)


def render_upload_section():
    """
    Sidebar uploader that lets the user replace the built-in dataset
    with their own CSV/Excel file. Returns the scored dataframe to use
    (uploaded + scored if present, otherwise None to signal "use default").
    """
    st.sidebar.header("Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your own data (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Must contain the same columns the model expects "
             "(e.g. contract_type, tenure_months, monthly_charges, etc.)",
    )

    if uploaded_file is None:
        st.session_state.pop("uploaded_scored_df", None)
        st.session_state.pop("uploaded_file_name", None)
        return None

    # Avoid re-processing the same file on every rerun
    if st.session_state.get("uploaded_file_name") == uploaded_file.name and \
            "uploaded_scored_df" in st.session_state:
        st.sidebar.success(f"Using uploaded file: {uploaded_file.name}")
        return st.session_state["uploaded_scored_df"]

    try:
        raw_df = read_uploaded_file(uploaded_file)
        predictor = get_predictor()
        scored_df = score_uploaded_data(predictor, raw_df)
    except ValueError as e:
        st.sidebar.error(str(e))
        return None
    except Exception as e:
        st.sidebar.error(f"Unexpected error while processing the file: {e}")
        return None

    st.session_state["uploaded_scored_df"] = scored_df
    st.session_state["uploaded_file_name"] = uploaded_file.name
    st.sidebar.success(f"Loaded and scored {len(scored_df)} rows from {uploaded_file.name}")

    if st.sidebar.button("Clear uploaded data"):
        st.session_state.pop("uploaded_scored_df", None)
        st.session_state.pop("uploaded_file_name", None)
        st.rerun()

    return scored_df


def styled_metric(container, label: str, value: str, color: str):
    """
    Render a metric that looks like st.metric but with a custom value
    color (st.metric itself doesn't expose a way to color the value).
    """
    container.markdown(
        f"""
        <div style="display:flex; flex-direction:column; gap:2px;">
            <span style="font-size:0.875rem; color:rgba(250,250,250,0.6);">{label}</span>
            <span style="font-size:2.25rem; font-weight:600; color:{color}; line-height:1.2;">
                {value}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(df: pd.DataFrame):
    # ============================================================
    # Sidebar Filters I Added
    # ============================================================
    st.sidebar.header("Dashboard Filters")

    # Contract Type filter
    contract = st.sidebar.selectbox(
        "Contract Type",
        ["All"] + sorted(df["contract_type"].unique())
    )

    # Location filter
    location = st.sidebar.selectbox(
        "Location",
        ["All"] + sorted(df["location"].unique())
    )

    # Risk Tier filter
    risk = st.sidebar.selectbox(
        "Risk Tier",
        ["All"] + sorted(df["risk_tier"].unique())
    )

    # Apply filters
    if contract != "All":
        df = df[df["contract_type"] == contract]

    if location != "All":
        df = df[df["location"] == location]

    if risk != "All":
        df = df[df["risk_tier"] == risk]
    ###################################################################

    has_actual_churn = "churn" in df.columns

    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    styled_metric(col1, "Total Customers", f"{len(df):,}", color="#21ba45")
    if has_actual_churn:
        col2.metric("Actual Churn Rate", f"{(df['churn'] == 'Yes').mean() * 100:.1f}%")
    else:
        col2.metric("Predicted Churn Rate", f"{(df['churn_prediction'] == 'Yes').mean() * 100:.1f}%")
    styled_metric(col3, "Predicted High Risk", f"{int((df['risk_tier'] == 'High Risk').sum()):,}", color="#db2828")
    col4.metric(
        "Avg. Monthly Charges (High Risk)",
        f"€{df.loc[df['risk_tier'] == 'High Risk', 'monthly_charges'].mean():.2f}",
    )

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        churn_col = "churn" if has_actual_churn else "churn_prediction"
        label = "Actual" if has_actual_churn else "Predicted"
        st.markdown(f"**{label} Churn Rate by Contract Type**")
        rate = (
            df.groupby("contract_type")[churn_col]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate (%)")
        )
        fig = px.bar(rate, x="contract_type", y="Churn Rate (%)", color="contract_type")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Predicted Risk Tier Distribution**")
        tier_counts = df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        fig = px.pie(tier_counts, names="Risk Tier", values="Count", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Churn Probability vs. Tenure**")
        fig = px.scatter(
            df, x="tenure_months", y="churn_probability", color="risk_tier",
            hover_data=["customer_id", "monthly_charges"],
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        churn_col = "churn" if has_actual_churn else "churn_prediction"
        label = "Actual" if has_actual_churn else "Predicted"
        st.markdown(f"**{label} Churn Rate by Internet Service**")
        rate2 = (
            df.groupby("internet_service")[churn_col]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate (%)")
        )
        fig = px.bar(rate2, x="internet_service", y="Churn Rate (%)", color="internet_service")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Segment Summary — by Contract Type**")
    st.caption(
        "Each column is colored on its own scale (darker = higher), so you can "
        "spot which segments stand out per metric."
    )

    churn_col = "churn" if has_actual_churn else "churn_prediction"
    churn_label = "Churn Rate %" if has_actual_churn else "Predicted Churn Rate %"

    segment_summary = (
        df.groupby("contract_type")
        .agg(
            Customers=("customer_id", "count"),
            **{churn_label: (churn_col, lambda x: (x == "Yes").mean() * 100)},
            **{"Avg. Churn Probability": ("churn_probability", "mean")},
            **{"High Risk %": ("risk_tier", lambda x: (x == "High Risk").mean() * 100)},
            **{"Avg. Monthly Charges (€)": ("monthly_charges", "mean")},
            **{"Avg. Tenure (mo.)": ("tenure_months", "mean")},
        )
        .reset_index()
        .rename(columns={"contract_type": "Contract Type"})
    )

    numeric_cols = [c for c in segment_summary.columns if c not in ("Contract Type", "Customers")]

    styled_summary = (
        segment_summary.style
        .format({c: "{:.1f}" for c in numeric_cols})
        .format({"Customers": "{:,}"})
    )
    # Color each metric on its own scale so a small column doesn't get
    # washed out by a large one (e.g. Customers vs. a percentage column).
    for col, cmap in [
        ("Customers", "Blues"),
        (churn_label, "Reds"),
        ("Avg. Churn Probability", "Reds"),
        ("High Risk %", "Reds"),
        ("Avg. Monthly Charges (€)", "Greens"),
        ("Avg. Tenure (mo.)", "Purples"),
    ]:
        styled_summary = styled_summary.background_gradient(cmap=cmap, subset=[col])

    st.dataframe(styled_summary, use_container_width=True)
    ###################################################################

    # ============================================================
    # YOUR TASK: add more charts / filters here, e.g.:
    #   - a sidebar filter by contract_type / location / risk_tier
    #   - a churn rate by payment_method chart
    #   - a satisfaction score vs churn probability chart
    # ============================================================

    st.markdown("### Monthly Revenue by Risk Tier")

    revenue = (
        df.groupby("risk_tier")["monthly_charges"]
        .sum()
        .reset_index(name="Monthly Revenue")
    )

    fig = px.bar(
        revenue,
        x="risk_tier",
        y="Monthly Revenue",
        color="risk_tier",
        text_auto=".2f",
    )

    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)
    ###############################################################################


def render_agent_tab():
    st.subheader("Ask the Churn AI Agent")
    st.caption(
        "Ask things like: 'Who are the top 5 highest risk customers?', "
        "'What is the churn rate by payment method?', "
        "'What's the risk for customer CUST-00006?'"
    )

    try:
        from churn_agent import ChurnAgent
    except ValueError as e:
        st.error(str(e))
        st.info("Add your ANTHROPIC_API_KEY to a .env file or to Streamlit secrets.")
        return

    if "agent" not in st.session_state:
        try:
            st.session_state.agent = ChurnAgent(
                api_key=st.secrets.get("GROQ_API_KEY", None) if hasattr(st, "secrets") else None
            )
        except Exception as e:
            st.error(f"Could not start the agent: {e}")
            return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("Ask a question about customer churn...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.agent.ask(user_input)
                st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))


def main():
    st.title("📉 Customer Churn Prediction — AI Agent & Dashboard")

    uploaded_df = render_upload_section()
    df = uploaded_df if uploaded_df is not None else get_scored_data()

    tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Ask the Agent"])
    with tab1:
        render_dashboard(df)
    with tab2:
        render_agent_tab()


if __name__ == "__main__":
    main()