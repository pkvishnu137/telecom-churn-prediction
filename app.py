"""
app.py
------
Streamlit dashboard for the Telecom Customer Churn Prediction System.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub
    2. Go to https://share.streamlit.io -> New app -> select repo -> main file: app.py
    3. Done.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import full_pipeline, clean_data, engineer_features
from train_model import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS / HTML STYLING
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background: radial-gradient(circle at 10% 0%, #131b2e 0%, #0b0f1a 45%, #07090f 100%);
    color: #e7ecf5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #0a0e18 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * { color: #cfd7e8 !important; }

/* Hero header */
.hero-banner {
    padding: 2.1rem 2.4rem;
    border-radius: 18px;
    background: linear-gradient(120deg, #1e3a8a 0%, #5b21b6 50%, #9d174d 100%);
    box-shadow: 0 10px 40px rgba(91, 33, 182, 0.35);
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: "";
    position: absolute; top: -50%; right: -10%;
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 70%);
}
.hero-title {
    font-size: 2.05rem; font-weight: 800; color: #ffffff;
    letter-spacing: -0.5px; margin: 0;
}
.hero-sub {
    font-size: 1.0rem; color: rgba(255,255,255,0.85);
    margin-top: 0.4rem; font-weight: 400; max-width: 720px;
}
.badge-row { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.badge {
    display: inline-block; padding: 0.28rem 0.85rem; border-radius: 999px;
    background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25);
    color: #fff; font-size: 0.78rem; font-weight: 600; backdrop-filter: blur(6px);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #141b2e, #0f1422);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.2rem 1.3rem;
    box-shadow: 0 6px 22px rgba(0,0,0,0.35);
    transition: transform .15s ease;
    height: 100%;
}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(124,158,255,0.4); }
.metric-label { font-size: 0.78rem; color: #93a0bd; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; }
.metric-value { font-size: 1.9rem; font-weight: 800; color: #ffffff; margin-top: 0.25rem; }
.metric-delta-good { color: #34d399; font-size: 0.85rem; font-weight: 600; margin-top: 0.2rem;}
.metric-delta-bad { color: #f87171; font-size: 0.85rem; font-weight: 600; margin-top: 0.2rem;}

/* Section headers */
.section-title {
    font-size: 1.3rem; font-weight: 700; color: #fff;
    margin: 1.8rem 0 0.8rem 0; padding-bottom: 0.4rem;
    border-bottom: 2px solid rgba(124,158,255,0.25);
}

/* Risk badge for predictions */
.risk-high {
    background: linear-gradient(120deg, #7f1d1d, #b91c1c);
    color: #fff; padding: 1rem 1.4rem; border-radius: 14px;
    font-weight: 700; font-size: 1.15rem; text-align: center;
    box-shadow: 0 6px 20px rgba(185,28,28,0.4);
}
.risk-medium {
    background: linear-gradient(120deg, #78350f, #d97706);
    color: #fff; padding: 1rem 1.4rem; border-radius: 14px;
    font-weight: 700; font-size: 1.15rem; text-align: center;
    box-shadow: 0 6px 20px rgba(217,119,6,0.4);
}
.risk-low {
    background: linear-gradient(120deg, #064e3b, #059669);
    color: #fff; padding: 1rem 1.4rem; border-radius: 14px;
    font-weight: 700; font-size: 1.15rem; text-align: center;
    box-shadow: 0 6px 20px rgba(5,150,105,0.4);
}

.factor-chip-risk {
    display:inline-block; background: rgba(248,113,113,0.15); color:#fca5a5;
    border: 1px solid rgba(248,113,113,0.4); border-radius: 10px;
    padding: 0.5rem 0.8rem; margin: 0.25rem 0.25rem 0.25rem 0; font-size: 0.88rem;
}
.factor-chip-retain {
    display:inline-block; background: rgba(52,211,153,0.15); color:#6ee7b7;
    border: 1px solid rgba(52,211,153,0.4); border-radius: 10px;
    padding: 0.5rem 0.8rem; margin: 0.25rem 0.25rem 0.25rem 0; font-size: 0.88rem;
}

/* DataFrames */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(120deg, #4f46e5, #7c3aed);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 0.55rem 1.3rem;
    transition: all 0.2s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform: translateY(-2px); box-shadow: 0 8px 18px rgba(124,58,237,0.4);
}

footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">📡 Telecom Customer Churn Intelligence Platform</p>
    <p class="hero-sub">An end-to-end ML system combining Random Forest, XGBoost &amp; LightGBM in a
    stacked ensemble, balanced with SMOTE, and explained with SHAP — built for proactive retention decisions.</p>
    <div class="badge-row">
        <span class="badge">🧠 Stacked Ensemble</span>
        <span class="badge">⚖️ SMOTE Balanced</span>
        <span class="badge">🔍 SHAP Explainable</span>
        <span class="badge">📊 Live Dashboard</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD ARTIFACTS (cached)
# ============================================================
MODELS_DIR = "models"
DATA_PATHS = ["data/telco_churn.csv", "data/telco_churn_synthetic.csv"]


@st.cache_resource(show_spinner=True)
def load_model_artifacts():
    preprocessor = joblib.load(f"{MODELS_DIR}/preprocessor.pkl")
    model = joblib.load(f"{MODELS_DIR}/churn_model.pkl")
    with open(f"{MODELS_DIR}/feature_names.json") as f:
        feature_names = json.load(f)
    with open(f"{MODELS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    background = pd.read_csv(f"{MODELS_DIR}/shap_background.csv")
    return preprocessor, model, feature_names, metrics, background


@st.cache_data(show_spinner=True)
def load_dataset():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return full_pipeline(p), p
    return None, None


artifacts_missing = not all(
    os.path.exists(f"{MODELS_DIR}/{f}")
    for f in ["preprocessor.pkl", "churn_model.pkl", "feature_names.json", "metrics.json", "shap_background.csv"]
)

if artifacts_missing:
    st.error(
        "⚠️ Model artifacts not found. Please run the training pipeline first:\n\n"
        "```bash\npython data/generate_sample_data.py   # or place real data at data/telco_churn.csv\npython src/train_model.py\n```"
    )
    st.stop()

preprocessor, model, feature_names, metrics, background = load_model_artifacts()
df, data_path = load_dataset()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 🧭 Navigate")
page = st.sidebar.radio(
    "",
    ["📊 Overview & EDA", "🤖 Model Performance", "🔮 Predict Single Customer",
     "📁 Batch Prediction (CSV)", "💡 Business Insights"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Dataset in use")
st.sidebar.code(data_path if data_path else "No dataset found", language="text")
st.sidebar.markdown(
    "Place the real Kaggle dataset at `data/telco_churn.csv` and retrain for production-grade accuracy."
)

# ============================================================
# PAGE 1: OVERVIEW & EDA
# ============================================================
if page == "📊 Overview & EDA":
    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)

    churn_rate = (df["Churn"] == "Yes").mean()
    total_customers = len(df)
    avg_tenure = df["tenure"].mean()
    avg_monthly = df["MonthlyCharges"].mean()

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Total Customers", f"{total_customers:,}"),
        (c2, "Churn Rate", f"{churn_rate:.1%}"),
        (c3, "Avg. Tenure", f"{avg_tenure:.1f} mo"),
        (c4, "Avg. Monthly Charge", f"${avg_monthly:.2f}"),
    ]
    for col, label, value in cards:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔎 Churn Drivers — Exploratory Analysis</div>', unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Churn by Contract Type**")
        contract_churn = (
            df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean()).sort_values(ascending=False)
        )
        st.bar_chart(contract_churn)

        st.markdown("**Churn by Internet Service**")
        net_churn = (
            df.groupby("InternetService")["Churn"].apply(lambda x: (x == "Yes").mean()).sort_values(ascending=False)
        )
        st.bar_chart(net_churn)

    with colB:
        st.markdown("**Churn by Payment Method**")
        pay_churn = (
            df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x == "Yes").mean()).sort_values(ascending=False)
        )
        st.bar_chart(pay_churn)

        st.markdown("**Churn by Tenure Group**")
        tenure_churn = (
            df.groupby("TenureGroup", observed=True)["Churn"].apply(lambda x: (x == "Yes").mean())
        )
        st.bar_chart(tenure_churn)

    st.markdown('<div class="section-title">📄 Raw Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=320)

# ============================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":
    st.markdown('<div class="section-title">🤖 Model Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    This system uses a **Stacking Ensemble** rather than a single algorithm, which consistently
    outperforms any individual base model on this dataset:

    - **Random Forest** — captures non-linear interactions, robust to outliers
    - **XGBoost** — gradient-boosted trees, strong on tabular imbalanced data
    - **LightGBM** — fast, leaf-wise boosting, complements XGBoost's splits
    - **Logistic Regression (meta-learner)** — blends the three base models' probability outputs into a final, calibrated prediction

    Class imbalance (≈27% churn) is corrected using **SMOTE** (Synthetic Minority Over-sampling)
    applied *only* to the training set after the train/test split, preventing data leakage.
    """)

    st.markdown('<div class="section-title">📈 Test Set Metrics</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    metric_items = [
        (m1, "Accuracy", metrics["accuracy"]),
        (m2, "Precision", metrics["precision"]),
        (m3, "Recall", metrics["recall"]),
        (m4, "F1 Score", metrics["f1_score"]),
        (m5, "ROC-AUC", metrics["roc_auc"]),
    ]
    for col, label, val in metric_items:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:1rem; color:#9aa6c4; font-size:0.9rem;">
    5-fold cross-validated ROC-AUC (on SMOTE-balanced training data):
    <b style="color:#fff;">{metrics.get('cv_roc_auc_mean','-')}</b> ± {metrics.get('cv_roc_auc_std','-')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🌍 Global Feature Importance (SHAP)</div>', unsafe_allow_html=True)
    st.info("Computing SHAP values on a background sample — this runs once and is cached.")

    @st.cache_resource(show_spinner="Computing SHAP values...")
    def compute_global_shap():
        sample = background.sample(min(60, len(background)), random_state=42)
        sample_proc = preprocessor.transform(sample)
        if hasattr(sample_proc, "toarray"):
            sample_proc = sample_proc.toarray()

        def predict_fn(X_proc):
            return model.predict_proba(X_proc)[:, 1]

        explainer = shap.Explainer(predict_fn, sample_proc)
        shap_values = explainer(sample_proc)
        shap_values.feature_names = feature_names
        return shap_values, sample

    try:
        shap_values, sample = compute_global_shap()
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.bar(shap_values, max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning(f"SHAP global plot could not be rendered in this environment: {e}")

# ============================================================
# PAGE 3: PREDICT SINGLE CUSTOMER
# ============================================================
elif page == "🔮 Predict Single Customer":
    st.markdown('<div class="section-title">🔮 Predict Churn Risk for a Customer</div>', unsafe_allow_html=True)
    st.markdown("Fill in the customer's profile, or load a random example from the dataset.")

    if st.button("🎲 Load Random Customer from Dataset"):
        st.session_state["sample_row"] = df.sample(1).iloc[0]

    sample_row = st.session_state.get("sample_row", df.iloc[0])

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(sample_row["gender"]))
            senior = st.selectbox("Senior Citizen", [0, 1], index=int(sample_row["SeniorCitizen"]))
            partner = st.selectbox("Has Partner", ["Yes", "No"], index=["Yes", "No"].index(sample_row["Partner"]))
            dependents = st.selectbox("Has Dependents", ["Yes", "No"], index=["Yes", "No"].index(sample_row["Dependents"]))
            tenure = st.slider("Tenure (months)", 0, 72, int(sample_row["tenure"]))
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                                     index=["Month-to-month", "One year", "Two year"].index(sample_row["Contract"]))
        with c2:
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=["Yes", "No"].index(sample_row["PhoneService"]))
            multiline_opts = ["Yes", "No", "No phone service"]
            multiline = st.selectbox("Multiple Lines", multiline_opts, index=multiline_opts.index(sample_row["MultipleLines"]))
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"],
                                     index=["DSL", "Fiber optic", "No"].index(sample_row["InternetService"]))
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=["Yes", "No"].index(sample_row["PaperlessBilling"]))
            payment_opts = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            payment = st.selectbox("Payment Method", payment_opts, index=payment_opts.index(sample_row["PaymentMethod"]))
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, float(sample_row["MonthlyCharges"]))
        with c3:
            internet_opts3 = ["Yes", "No", "No internet service"]
            sec = st.selectbox("Online Security", internet_opts3, index=internet_opts3.index(sample_row["OnlineSecurity"]))
            backup = st.selectbox("Online Backup", internet_opts3, index=internet_opts3.index(sample_row["OnlineBackup"]))
            device = st.selectbox("Device Protection", internet_opts3, index=internet_opts3.index(sample_row["DeviceProtection"]))
            tech = st.selectbox("Tech Support", internet_opts3, index=internet_opts3.index(sample_row["TechSupport"]))
            tv = st.selectbox("Streaming TV", internet_opts3, index=internet_opts3.index(sample_row["StreamingTV"]))
            movies = st.selectbox("Streaming Movies", internet_opts3, index=internet_opts3.index(sample_row["StreamingMovies"]))
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(sample_row["TotalCharges"]))

        submitted = st.form_submit_button("🔮 Predict Churn Risk")

    if submitted:
        raw = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multiline, "InternetService": internet,
            "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": device, "TechSupport": tech,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total_charges,
        }])
        engineered = engineer_features(clean_data(raw))

        X_input = engineered[ALL_FEATURES]
        X_proc = preprocessor.transform(X_input)
        proba = model.predict_proba(X_proc)[0, 1]

        st.markdown("###")
        risk_col, gauge_col = st.columns([1, 2])
        with risk_col:
            if proba >= 0.6:
                st.markdown(f'<div class="risk-high">🔴 HIGH RISK<br>{proba:.1%} churn probability</div>', unsafe_allow_html=True)
            elif proba >= 0.35:
                st.markdown(f'<div class="risk-medium">🟠 MEDIUM RISK<br>{proba:.1%} churn probability</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">🟢 LOW RISK<br>{proba:.1%} churn probability</div>', unsafe_allow_html=True)

        with gauge_col:
            st.progress(min(int(proba * 100), 100))
            st.caption(f"Model confidence: churn probability = {proba:.2%}")

        st.markdown('<div class="section-title">🧩 Why this prediction? (SHAP Explanation)</div>', unsafe_allow_html=True)
        with st.spinner("Computing local SHAP explanation..."):
            bg_sample = background.sample(min(40, len(background)), random_state=42)
            bg_proc = preprocessor.transform(bg_sample)
            if hasattr(bg_proc, "toarray"):
                bg_proc = bg_proc.toarray()

            def predict_fn(X_proc):
                return model.predict_proba(X_proc)[:, 1]

            explainer = shap.Explainer(predict_fn, bg_proc)
            X_input_proc = preprocessor.transform(X_input)
            if hasattr(X_input_proc, "toarray"):
                X_input_proc = X_input_proc.toarray()
            shap_vals = explainer(X_input_proc)
            shap_vals.feature_names = feature_names

        values = shap_vals.values[0]
        names = feature_names
        pairs = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
        increasing = [p for p in pairs if p[1] > 0][:5]
        decreasing = [p for p in pairs if p[1] < 0][:5]

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**🔺 Factors increasing churn risk**")
            if increasing:
                chips = "".join([f'<span class="factor-chip-risk">{n}: +{v:.3f}</span>' for n, v in increasing])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No strong risk-increasing factors found.")
        with cc2:
            st.markdown("**🔻 Factors reducing churn risk**")
            if decreasing:
                chips = "".join([f'<span class="factor-chip-retain">{n}: {v:.3f}</span>' for n, v in decreasing])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.caption("No strong risk-reducing factors found.")

        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.caption(f"(Waterfall plot unavailable: {e})")

# ============================================================
# PAGE 4: BATCH PREDICTION
# ============================================================
elif page == "📁 Batch Prediction (CSV)":
    st.markdown('<div class="section-title">📁 Batch Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV with the same columns as the Telco dataset (excluding `Churn`) to score multiple customers at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    use_demo = st.checkbox("Or use a demo batch from the holdout set", value=not bool(uploaded))

    batch_df = None
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        batch_df = clean_data(batch_df)
        batch_df = engineer_features(batch_df)
    elif use_demo and os.path.exists(f"{MODELS_DIR}/holdout_sample.csv"):
        batch_df = pd.read_csv(f"{MODELS_DIR}/holdout_sample.csv")
        batch_df = engineer_features(clean_data(batch_df.drop(columns=["Churn"], errors="ignore")))

    if batch_df is not None:
        X_batch = batch_df[ALL_FEATURES]
        X_proc = preprocessor.transform(X_batch)
        proba = model.predict_proba(X_proc)[:, 1]
        pred = (proba >= 0.5).astype(int)

        results = batch_df.copy()
        results["Churn_Probability"] = np.round(proba, 4)
        results["Churn_Prediction"] = np.where(pred == 1, "Yes", "No")
        results["Risk_Tier"] = pd.cut(proba, bins=[-0.01, 0.35, 0.6, 1.0], labels=["Low", "Medium", "High"])

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-label">Customers Scored</div><div class="metric-value">{len(results)}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-label">Predicted Churners</div><div class="metric-value">{(pred==1).sum()}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-label">High Risk Customers</div><div class="metric-value">{(results["Risk_Tier"]=="High").sum()}</div></div>', unsafe_allow_html=True)

        st.markdown("####")
        st.dataframe(
            results[["tenure", "Contract", "MonthlyCharges", "Churn_Probability", "Churn_Prediction", "Risk_Tier"]]
            .sort_values("Churn_Probability", ascending=False),
            use_container_width=True, height=420
        )

        csv_out = results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Scored CSV", csv_out, "churn_predictions.csv", "text/csv")
    else:
        st.info("Upload a CSV file or check the demo box above to see batch predictions.")

# ============================================================
# PAGE 5: BUSINESS INSIGHTS
# ============================================================
elif page == "💡 Business Insights":
    st.markdown('<div class="section-title">💡 Actionable Business Insights</div>', unsafe_allow_html=True)

    insights = [
        ("📄 Contract Type", "Month-to-month customers churn at a dramatically higher rate than 1- or 2-year contract holders.",
         "Offer discounted incentives for switching to annual contracts, especially within a customer's first 6 months."),
        ("💳 Payment Method", "Customers using Electronic Check have the highest churn rate among all payment methods.",
         "Encourage migration to automatic bank transfer or credit card payments via small fee waivers or loyalty points."),
        ("🌐 Fiber Optic Customers", "Fiber optic subscribers without security/tech-support add-ons show elevated churn.",
         "Bundle free trial months of Tech Support / Online Security for new fiber customers."),
        ("⏳ Early Tenure Risk", "Customers within their first 0-6 months are the most likely to churn.",
         "Deploy a structured onboarding & check-in program during the first 90 days."),
        ("👴 Senior Citizens", "Senior citizen customers churn somewhat more than non-seniors.",
         "Provide simplified billing and a dedicated senior support line."),
    ]

    for title, finding, action in insights:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:0.9rem;">
            <div style="font-weight:700; font-size:1.05rem; color:#fff;">{title}</div>
            <div style="margin-top:0.4rem; color:#c2cbe3;"><b>Finding:</b> {finding}</div>
            <div style="margin-top:0.3rem; color:#93e0c4;"><b>Recommended Action:</b> {action}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📌 Suggested Retention Strategy Funnel</div>', unsafe_allow_html=True)
    st.markdown("""
    1. **Score** the full customer base weekly using the Batch Prediction page.
    2. **Segment** into Low / Medium / High risk tiers.
    3. **Prioritize** High Risk + High Value (top MonthlyCharges tercile) customers for proactive outreach.
    4. **Personalize** offers using the top SHAP risk factors for each customer (Predict Single Customer page).
    5. **Track** churn rate trend monthly to measure retention campaign ROI.
    """)

st.markdown("""
<div style="text-align:center; margin-top:3rem; padding:1rem; color:#5b6584; font-size:0.8rem;">
Built with Streamlit · Scikit-learn · XGBoost · LightGBM · SHAP &nbsp;|&nbsp; Customer Churn Prediction System
</div>
""", unsafe_allow_html=True)
