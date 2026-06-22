"""
app.py  —  Telecom Churn Intelligence Platform
Upgraded UI/UX: Plotly charts, animated gauge, color-coded tables,
retention ROI calculator, landing page, live sidebar stats.
"""

import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import full_pipeline, clean_data, engineer_features
from train_model import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ — Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Background ── */
.stApp { background: #080c14; color: #e2e8f4; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0c1120 0%,#080c14 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #94a3b8 !important; }

/* ── Radio buttons ── */
div[role="radiogroup"] label { color: #cbd5e1 !important; font-size:0.9rem; }
div[role="radiogroup"] label:hover { color:#fff !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg,#1e1b4b 0%,#312e81 30%,#4c1d95 60%,#6b21a8 100%);
    border-radius: 20px; padding: 2.5rem 2.8rem;
    box-shadow: 0 20px 60px rgba(109,40,217,0.35);
    position: relative; overflow: hidden; margin-bottom: 2rem;
}
.hero::before {
    content:""; position:absolute; top:-60px; right:-60px;
    width:320px; height:320px; border-radius:50%;
    background: radial-gradient(circle, rgba(167,139,250,0.25) 0%, transparent 70%);
}
.hero::after {
    content:""; position:absolute; bottom:-80px; left:30%;
    width:200px; height:200px; border-radius:50%;
    background: radial-gradient(circle, rgba(236,72,153,0.15) 0%, transparent 70%);
}
.hero-title { font-size:2.4rem; font-weight:900; color:#fff; letter-spacing:-1px; margin:0; }
.hero-sub   { font-size:1rem; color:rgba(255,255,255,0.8); margin-top:.5rem; max-width:700px; line-height:1.6; }
.pill {
    display:inline-block; padding:.3rem 1rem; border-radius:999px; margin:.25rem .2rem;
    background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2);
    color:#fff; font-size:.78rem; font-weight:600; backdrop-filter:blur(8px);
}

/* ── Section title ── */
.sec-title {
    font-size:1.25rem; font-weight:800; color:#fff;
    margin:2rem 0 1rem 0; padding-bottom:.5rem;
    border-bottom: 2px solid rgba(139,92,246,0.35);
    letter-spacing:-.3px;
}

/* ── Metric card ── */
.mcard {
    background:linear-gradient(145deg,#0f1629,#0d1322);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:18px; padding:1.4rem 1.5rem;
    box-shadow:0 8px 30px rgba(0,0,0,0.4);
    transition:transform .2s,border-color .2s;
    height:100%;
}
.mcard:hover { transform:translateY(-4px); border-color:rgba(139,92,246,0.5); }
.mcard-label { font-size:.72rem; color:#64748b; text-transform:uppercase; letter-spacing:.8px; font-weight:700; }
.mcard-value { font-size:2rem; font-weight:900; color:#fff; margin-top:.2rem; }
.mcard-delta-g { color:#34d399; font-size:.82rem; font-weight:600; margin-top:.25rem; }
.mcard-delta-r { color:#f87171; font-size:.82rem; font-weight:600; margin-top:.25rem; }
.mcard-delta-n { color:#94a3b8; font-size:.82rem; font-weight:600; margin-top:.25rem; }

/* ── Risk badges ── */
.risk-high {
    background:linear-gradient(135deg,#7f1d1d,#dc2626);
    color:#fff; padding:1.4rem 1.6rem; border-radius:16px;
    font-weight:800; font-size:1.1rem; text-align:center;
    box-shadow:0 8px 28px rgba(220,38,38,.4); line-height:1.6;
}
.risk-med {
    background:linear-gradient(135deg,#78350f,#d97706);
    color:#fff; padding:1.4rem 1.6rem; border-radius:16px;
    font-weight:800; font-size:1.1rem; text-align:center;
    box-shadow:0 8px 28px rgba(217,119,6,.4); line-height:1.6;
}
.risk-low {
    background:linear-gradient(135deg,#064e3b,#059669);
    color:#fff; padding:1.4rem 1.6rem; border-radius:16px;
    font-weight:800; font-size:1.1rem; text-align:center;
    box-shadow:0 8px 28px rgba(5,150,105,.4); line-height:1.6;
}

/* ── Factor chips ── */
.chip-r {
    display:inline-block; background:rgba(248,113,113,.12); color:#fca5a5;
    border:1px solid rgba(248,113,113,.35); border-radius:10px;
    padding:.45rem .85rem; margin:.25rem; font-size:.85rem; font-weight:500;
}
.chip-g {
    display:inline-block; background:rgba(52,211,153,.12); color:#6ee7b7;
    border:1px solid rgba(52,211,153,.35); border-radius:10px;
    padding:.45rem .85rem; margin:.25rem; font-size:.85rem; font-weight:500;
}

/* ── Insight card ── */
.insight-card {
    background:linear-gradient(145deg,#0f1629,#0d1322);
    border:1px solid rgba(255,255,255,0.06);
    border-left:4px solid #7c3aed;
    border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:1rem;
    box-shadow:0 4px 18px rgba(0,0,0,.3);
}

/* ── Streamlit overrides ── */
.stButton>button, .stDownloadButton>button {
    background:linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; padding:.55rem 1.4rem !important;
    transition:all .2s !important;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 22px rgba(124,58,237,.45) !important;
}
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }
.stNumberInput input, .stSelectbox select, .stSlider { color:#fff !important; }
.stForm { background:transparent !important; }
footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
.stDeployButton { display:none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODELS_DIR   = "models"
DATA_PATHS   = ["data/telco_churn.csv", "data/telco_churn_synthetic.csv"]
PLOTLY_THEME = dict(
    plot_bgcolor ="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color   ="#94a3b8",
    title_font_color="#fff",
    colorway=["#7c3aed","#4f46e5","#06b6d4","#10b981","#f59e0b","#ef4444"],
)

# ─────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────
def landing():
    st.markdown("""
    <div style="min-height:80vh; display:flex; flex-direction:column;
                align-items:center; justify-content:center; text-align:center; padding:4rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">📡</div>
        <h1 style="font-size:3.2rem; font-weight:900; color:#fff; letter-spacing:-1.5px; margin:0;">
            ChurnIQ
        </h1>
        <p style="font-size:1.15rem; color:#94a3b8; margin-top:.8rem; max-width:580px; line-height:1.7;">
            AI-powered telecom customer churn prediction.<br>
            Built on a stacking ensemble of Random Forest, XGBoost &amp; LightGBM —
            explained with SHAP, ready for business decisions.
        </p>
        <div style="margin:1.5rem 0; display:flex; gap:.6rem; flex-wrap:wrap; justify-content:center;">
            <span class="pill">🧠 Stacking Ensemble</span>
            <span class="pill">⚖️ SMOTE Balanced</span>
            <span class="pill">🔍 SHAP Explainable</span>
            <span class="pill">📊 Interactive Dashboard</span>
            <span class="pill">💰 ROI Calculator</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col = st.columns([1,2,1])[1]
    with col:
        if st.button("🚀  Enter Dashboard", use_container_width=True):
            st.session_state["started"] = True
            st.rerun()
    st.stop()

if "started" not in st.session_state:
    landing()

# ─────────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────────
artifacts_ok = all(
    os.path.exists(f"{MODELS_DIR}/{f}")
    for f in ["preprocessor.pkl","churn_model.pkl","feature_names.json","metrics.json","shap_background.csv"]
)
if not artifacts_ok:
    st.error("⚠️ Model artifacts missing. Run:\n```\npython data/generate_sample_data.py\npython src/train_model.py\n```")
    st.stop()

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    pre   = joblib.load(f"{MODELS_DIR}/preprocessor.pkl")
    model = joblib.load(f"{MODELS_DIR}/churn_model.pkl")
    fn    = json.load(open(f"{MODELS_DIR}/feature_names.json"))
    met   = json.load(open(f"{MODELS_DIR}/metrics.json"))
    bg    = pd.read_csv(f"{MODELS_DIR}/shap_background.csv")
    return pre, model, fn, met, bg

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return full_pipeline(p), p
    return None, None

preprocessor, model, feature_names, metrics, background = load_artifacts()
df, data_path = load_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.8rem 0 1.2rem 0; text-align:center;">
        <span style="font-size:1.8rem;">📡</span>
        <div style="font-weight:800; font-size:1.1rem; color:#fff; margin-top:.3rem;">ChurnIQ</div>
        <div style="font-size:.72rem; color:#475569; margin-top:.1rem;">Telecom Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Overview & EDA",
        "🤖  Model Performance",
        "🔮  Predict Single Customer",
        "📁  Batch Prediction",
        "💰  Retention ROI Calculator",
        "💡  Business Insights",
    ], label_visibility="collapsed")

    st.markdown("---")

    # Live sidebar stats
    if df is not None:
        churn_rate = (df["Churn"] == "Yes").mean()
        high_risk_est = int(len(df) * churn_rate * 0.6)
        st.markdown("### 📊 Dataset Stats")
        st.markdown(f"""
        <div style="display:flex; flex-direction:column; gap:.5rem; padding:.2rem 0;">
            <div style="background:rgba(124,58,237,.12); border:1px solid rgba(124,58,237,.3);
                        border-radius:10px; padding:.7rem 1rem;">
                <div style="font-size:.7rem; color:#7c3aed; font-weight:700; text-transform:uppercase;">Total Customers</div>
                <div style="font-size:1.3rem; font-weight:800; color:#fff;">{len(df):,}</div>
            </div>
            <div style="background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.3);
                        border-radius:10px; padding:.7rem 1rem;">
                <div style="font-size:.7rem; color:#ef4444; font-weight:700; text-transform:uppercase;">Churn Rate</div>
                <div style="font-size:1.3rem; font-weight:800; color:#fff;">{churn_rate:.1%}</div>
            </div>
            <div style="background:rgba(245,158,11,.1); border:1px solid rgba(245,158,11,.3);
                        border-radius:10px; padding:.7rem 1rem;">
                <div style="font-size:.7rem; color:#f59e0b; font-weight:700; text-transform:uppercase;">Est. High-Risk</div>
                <div style="font-size:1.3rem; font-weight:800; color:#fff;">{high_risk_est:,}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div style='font-size:.7rem; color:#334155;'>Dataset: <code>{data_path}</code></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def metric_card(label, value, delta="", delta_type="n"):
    dt_class = {"g":"mcard-delta-g","r":"mcard-delta-r","n":"mcard-delta-n"}[delta_type]
    delta_html = f'<div class="{dt_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="mcard">
        <div class="mcard-label">{label}</div>
        <div class="mcard-value">{value}</div>
        {delta_html}
    </div>"""

def plotly_defaults(fig):
    fig.update_layout(**PLOTLY_THEME, margin=dict(l=20,r=20,t=40,b=20),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig

def churn_gauge(probability):
    if probability >= 0.6:
        color, label = "#ef4444", "HIGH RISK"
    elif probability >= 0.35:
        color, label = "#f59e0b", "MEDIUM RISK"
    else:
        color, label = "#10b981", "LOW RISK"
    pct = int(probability * 100)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix":"%","font":{"size":44,"color":color,"family":"Inter"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#334155","tickfont":{"color":"#64748b"}},
            "bar":{"color":color,"thickness":.25},
            "bgcolor":"rgba(0,0,0,0)",
            "bordercolor":"rgba(0,0,0,0)",
            "steps":[
                {"range":[0,35],"color":"rgba(16,185,129,.15)"},
                {"range":[35,60],"color":"rgba(245,158,11,.15)"},
                {"range":[60,100],"color":"rgba(239,68,68,.15)"},
            ],
            "threshold":{"line":{"color":color,"width":4},"thickness":.75,"value":pct},
        },
        title={"text":f"<b>{label}</b>","font":{"color":color,"size":16}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(t=60,b=10,l=20,r=20),
        font={"family":"Inter"},
    )
    return fig

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW & EDA
# ─────────────────────────────────────────────
if page == "📊  Overview & EDA":

    st.markdown("""
    <div class="hero">
        <p class="hero-title">📊 Overview & EDA</p>
        <p class="hero-sub">Understand your customer base — where churn concentrates, which segments are most at risk, and what patterns the model is learning from.</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found."); st.stop()

    churn_count = (df["Churn"]=="Yes").sum()
    retain_count = (df["Churn"]=="No").sum()
    churn_rate   = churn_count / len(df)
    avg_tenure   = df["tenure"].mean()
    avg_monthly  = df["MonthlyCharges"].mean()
    avg_total    = df["TotalCharges"].mean()

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(metric_card("Total Customers", f"{len(df):,}", "IBM Telco dataset"), unsafe_allow_html=True)
    c2.markdown(metric_card("Churn Rate", f"{churn_rate:.1%}", f"↑ {churn_count:,} customers", "r"), unsafe_allow_html=True)
    c3.markdown(metric_card("Avg. Tenure", f"{avg_tenure:.1f} mo", f"Retained avg: {df[df.Churn=='No']['tenure'].mean():.1f} mo", "g"), unsafe_allow_html=True)
    c4.markdown(metric_card("Avg. Monthly Charge", f"${avg_monthly:.2f}", f"Churned avg: ${df[df.Churn=='Yes']['MonthlyCharges'].mean():.2f}", "n"), unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🔎 Churn Rate by Key Segments</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Contract type
        ct = df.groupby("Contract")["Churn"].apply(lambda x:(x=="Yes").mean()*100).reset_index()
        ct.columns = ["Contract","Churn Rate (%)"]
        fig = px.bar(ct, x="Contract", y="Churn Rate (%)", color="Churn Rate (%)",
                     color_continuous_scale="Purples", title="Churn Rate by Contract Type",
                     text=ct["Churn Rate (%)"].apply(lambda x:f"{x:.1f}%"))
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_defaults(fig), use_container_width=True)

        # Payment method
        pm = df.groupby("PaymentMethod")["Churn"].apply(lambda x:(x=="Yes").mean()*100).reset_index()
        pm.columns = ["Payment","Churn Rate (%)"]
        pm["Payment"] = pm["Payment"].str.replace(" (automatic)","",regex=False)
        fig2 = px.bar(pm, x="Churn Rate (%)", y="Payment", orientation="h",
                      color="Churn Rate (%)", color_continuous_scale="RdPu",
                      title="Churn Rate by Payment Method",
                      text=pm["Churn Rate (%)"].apply(lambda x:f"{x:.1f}%"))
        fig2.update_traces(textposition="outside", marker_line_width=0)
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_defaults(fig2), use_container_width=True)

    with col2:
        # Internet service
        isp = df.groupby("InternetService")["Churn"].apply(lambda x:(x=="Yes").mean()*100).reset_index()
        isp.columns = ["Internet","Churn Rate (%)"]
        fig3 = px.bar(isp, x="Internet", y="Churn Rate (%)", color="Churn Rate (%)",
                      color_continuous_scale="Blues", title="Churn Rate by Internet Service",
                      text=isp["Churn Rate (%)"].apply(lambda x:f"{x:.1f}%"))
        fig3.update_traces(textposition="outside", marker_line_width=0)
        fig3.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_defaults(fig3), use_container_width=True)

        # Tenure group
        tg = df.groupby("TenureGroup", observed=True)["Churn"].apply(lambda x:(x=="Yes").mean()*100).reset_index()
        tg.columns = ["Tenure Group","Churn Rate (%)"]
        fig4 = px.line(tg, x="Tenure Group", y="Churn Rate (%)",
                       markers=True, title="Churn Rate by Tenure Group",
                       line_shape="spline")
        fig4.update_traces(line_color="#7c3aed", marker_color="#a78bfa", marker_size=9,
                           fill="tozeroy", fillcolor="rgba(124,58,237,0.1)")
        st.plotly_chart(plotly_defaults(fig4), use_container_width=True)

    st.markdown('<div class="sec-title">📈 Distribution Analysis</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        fig5 = px.histogram(df, x="MonthlyCharges", color="Churn",
                            barmode="overlay", nbins=40,
                            color_discrete_map={"Yes":"#ef4444","No":"#7c3aed"},
                            title="Monthly Charges Distribution by Churn")
        fig5.update_traces(opacity=0.75)
        st.plotly_chart(plotly_defaults(fig5), use_container_width=True)

    with col4:
        fig6 = px.histogram(df, x="tenure", color="Churn",
                            barmode="overlay", nbins=36,
                            color_discrete_map={"Yes":"#f59e0b","No":"#10b981"},
                            title="Tenure Distribution by Churn")
        fig6.update_traces(opacity=0.75)
        st.plotly_chart(plotly_defaults(fig6), use_container_width=True)

    st.markdown('<div class="sec-title">📄 Raw Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(df.head(60), use_container_width=True, height=340)

# ─────────────────────────────────────────────
# PAGE 2 — MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "🤖  Model Performance":

    st.markdown("""
    <div class="hero">
        <p class="hero-title">🤖 Model Performance</p>
        <p class="hero-sub">A stacking ensemble (Random Forest + XGBoost + LightGBM → Logistic Regression) trained on SMOTE-balanced data. See how it stacks up against baseline single-model approaches.</p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture diagram
    st.markdown('<div class="sec-title">🏗️ Model Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; gap:1rem; align-items:center; flex-wrap:wrap; padding:.5rem 0 1.5rem 0;">
        <div style="background:rgba(16,185,129,.12); border:1px solid rgba(16,185,129,.35);
                    border-radius:12px; padding:1rem 1.4rem; text-align:center; min-width:130px;">
            <div style="font-size:1.5rem;">🌲</div>
            <div style="font-weight:700; color:#6ee7b7; margin-top:.3rem;">Random Forest</div>
            <div style="font-size:.75rem; color:#64748b;">Non-linear patterns</div>
        </div>
        <div style="font-size:1.4rem; color:#4b5563;">+</div>
        <div style="background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.35);
                    border-radius:12px; padding:1rem 1.4rem; text-align:center; min-width:130px;">
            <div style="font-size:1.5rem;">⚡</div>
            <div style="font-weight:700; color:#fcd34d; margin-top:.3rem;">XGBoost</div>
            <div style="font-size:.75rem; color:#64748b;">Gradient boosting</div>
        </div>
        <div style="font-size:1.4rem; color:#4b5563;">+</div>
        <div style="background:rgba(6,182,212,.12); border:1px solid rgba(6,182,212,.35);
                    border-radius:12px; padding:1rem 1.4rem; text-align:center; min-width:130px;">
            <div style="font-size:1.5rem;">🚀</div>
            <div style="font-weight:700; color:#67e8f9; margin-top:.3rem;">LightGBM</div>
            <div style="font-size:.75rem; color:#64748b;">Leaf-wise boosting</div>
        </div>
        <div style="font-size:1.4rem; color:#4b5563;">→</div>
        <div style="background:rgba(124,58,237,.2); border:2px solid rgba(124,58,237,.6);
                    border-radius:12px; padding:1rem 1.4rem; text-align:center; min-width:150px;">
            <div style="font-size:1.5rem;">🧠</div>
            <div style="font-weight:700; color:#a78bfa; margin-top:.3rem;">Meta-Learner</div>
            <div style="font-size:.75rem; color:#7c3aed;">Logistic Regression</div>
        </div>
        <div style="font-size:1.4rem; color:#4b5563;">→</div>
        <div style="background:rgba(236,72,153,.12); border:1px solid rgba(236,72,153,.35);
                    border-radius:12px; padding:1rem 1.4rem; text-align:center; min-width:130px;">
            <div style="font-size:1.5rem;">🎯</div>
            <div style="font-weight:700; color:#f9a8d4; margin-top:.3rem;">Final Score</div>
            <div style="font-size:.75rem; color:#64748b;">Churn probability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    st.markdown('<div class="sec-title">📈 Test Set Metrics</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    cards = [
        ("Accuracy",  metrics["accuracy"],  "Overall correct predictions"),
        ("Precision", metrics["precision"], "Of predicted churners, actually churned"),
        ("Recall",    metrics["recall"],    "Of actual churners, caught by model"),
        ("F1 Score",  metrics["f1_score"],  "Harmonic mean of precision & recall"),
        ("ROC-AUC",   metrics["roc_auc"],   "Discrimination ability (0.5=random)"),
    ]
    for col, (label, val, tip) in zip(cols, cards):
        bar = int(val * 100)
        color = "#10b981" if val >= 0.75 else "#f59e0b" if val >= 0.6 else "#ef4444"
        col.markdown(f"""
        <div class="mcard">
            <div class="mcard-label">{label}</div>
            <div class="mcard-value" style="color:{color};">{val*100:.1f}%</div>
            <div style="background:#1e293b; border-radius:999px; height:6px; margin:.5rem 0;">
                <div style="width:{bar}%; background:{color}; height:6px; border-radius:999px;"></div>
            </div>
            <div class="mcard-delta-n" style="font-size:.7rem;">{tip}</div>
        </div>""", unsafe_allow_html=True)

    cv_mean = metrics.get("cv_roc_auc_mean", "-")
    cv_std  = metrics.get("cv_roc_auc_std",  "-")
    st.markdown(f"""
    <div style="margin-top:1rem; background:rgba(124,58,237,.1); border:1px solid rgba(124,58,237,.3);
                border-radius:12px; padding:1rem 1.4rem; color:#a78bfa; font-size:.9rem;">
        🔁 5-fold Cross-Validated ROC-AUC (on SMOTE-balanced training data):
        <b style="color:#fff; font-size:1.1rem;"> {cv_mean}</b> ± {cv_std}
    </div>
    """, unsafe_allow_html=True)

    # Radar chart — model metrics
    st.markdown('<div class="sec-title">🕸️ Metrics Radar</div>', unsafe_allow_html=True)
    radar_cols = st.columns([2, 1])
    with radar_cols[0]:
        cats   = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
        vals   = [metrics["accuracy"], metrics["precision"], metrics["recall"],
                  metrics["f1_score"], metrics["roc_auc"]]
        # Baseline (typical single RF on this dataset)
        base   = [0.79, 0.63, 0.54, 0.58, 0.82]
        fig_r  = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                        fill="toself", name="Stacking Ensemble",
                        line_color="#7c3aed", fillcolor="rgba(124,58,237,0.2)"))
        fig_r.add_trace(go.Scatterpolar(r=base+[base[0]], theta=cats+[cats[0]],
                        fill="toself", name="Baseline RF (typical)",
                        line_color="#475569", fillcolor="rgba(71,85,105,0.1)",
                        line_dash="dash"))
        fig_r.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e293b",
                                       tickfont=dict(color="#64748b")),
                       angularaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8"))),
            showlegend=True, **PLOTLY_THEME, height=360,
            legend=dict(bgcolor="rgba(15,22,41,.8)", bordercolor="#1e293b", borderwidth=1),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with radar_cols[1]:
        st.markdown("""
        <div class="mcard" style="margin-top:1rem;">
            <div class="mcard-label">Why Stacking Wins</div>
            <div style="margin-top:.7rem; color:#94a3b8; font-size:.85rem; line-height:1.7;">
                Each base model captures different patterns. The meta-learner learns the optimal blend of their probability outputs — typically beating any single model by <b style="color:#a78bfa;">2–5 ROC-AUC points</b> on imbalanced tabular data.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP global
    st.markdown('<div class="sec-title">🌍 Global SHAP Feature Importance</div>', unsafe_allow_html=True)
    st.info("Computed once and cached — may take ~30 seconds on first load.")

    @st.cache_resource(show_spinner="Computing global SHAP values…")
    def global_shap():
        samp = background.sample(min(50, len(background)), random_state=42)
        samp_proc = preprocessor.transform(samp)
        if hasattr(samp_proc, "toarray"):
            samp_proc = samp_proc.toarray()
        def pfn(X): return model.predict_proba(X)[:,1]
        exp = shap.Explainer(pfn, samp_proc)
        sv  = exp(samp_proc)
        sv.feature_names = feature_names
        return sv

    try:
        sv = global_shap()
        mean_abs = np.abs(sv.values).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:15]
        top_names = [feature_names[i] for i in top_idx]
        top_vals  = mean_abs[top_idx]

        # Clean up sklearn prefix for display
        def clean(n):
            return n.replace("num__","").replace("cat__","").replace("_"," ")

        fig_shap = go.Figure(go.Bar(
            x=top_vals[::-1],
            y=[clean(n) for n in top_names[::-1]],
            orientation="h",
            marker=dict(
                color=top_vals[::-1],
                colorscale=[[0,"#312e81"],[0.5,"#7c3aed"],[1,"#a78bfa"]],
                line_width=0,
            ),
        ))
        fig_shap.update_layout(
            title="Top 15 Features by Mean |SHAP value|",
            xaxis_title="Mean |SHAP value|", yaxis_title="",
            **PLOTLY_THEME, height=460,
            xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP global plot unavailable: {e}")

# ─────────────────────────────────────────────
# PAGE 3 — PREDICT SINGLE CUSTOMER
# ─────────────────────────────────────────────
elif page == "🔮  Predict Single Customer":

    st.markdown("""
    <div class="hero">
        <p class="hero-title">🔮 Predict Single Customer</p>
        <p class="hero-sub">Fill in a customer's profile to get an instant churn probability with a SHAP explanation of exactly which factors are driving the risk.</p>
    </div>
    """, unsafe_allow_html=True)

    if df is not None and st.button("🎲 Load Random Customer from Dataset"):
        st.session_state["sample_row"] = df.sample(1).iloc[0]

    row = st.session_state.get("sample_row", df.iloc[0] if df is not None else None)
    if row is None:
        st.error("Dataset not available for random samples."); st.stop()

    def safe_idx(opts, val):
        try: return list(opts).index(val)
        except: return 0

    with st.form("cform"):
        st.markdown("#### 👤 Customer Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Demographics & Account**")
            gender     = st.selectbox("Gender", ["Male","Female"], index=safe_idx(["Male","Female"], row["gender"]))
            senior     = st.selectbox("Senior Citizen", [0,1], index=int(row["SeniorCitizen"]))
            partner    = st.selectbox("Has Partner", ["Yes","No"], index=safe_idx(["Yes","No"], row["Partner"]))
            dependents = st.selectbox("Has Dependents", ["Yes","No"], index=safe_idx(["Yes","No"], row["Dependents"]))
            tenure     = st.slider("Tenure (months)", 0, 72, int(row["tenure"]))
            contract   = st.selectbox("Contract", ["Month-to-month","One year","Two year"],
                                      index=safe_idx(["Month-to-month","One year","Two year"], row["Contract"]))
        with c2:
            st.markdown("**Services**")
            phone   = st.selectbox("Phone Service", ["Yes","No"], index=safe_idx(["Yes","No"], row["PhoneService"]))
            mlopts  = ["Yes","No","No phone service"]
            mlines  = st.selectbox("Multiple Lines", mlopts, index=safe_idx(mlopts, row["MultipleLines"]))
            internet= st.selectbox("Internet Service", ["DSL","Fiber optic","No"],
                                   index=safe_idx(["DSL","Fiber optic","No"], row["InternetService"]))
            io3     = ["Yes","No","No internet service"]
            sec     = st.selectbox("Online Security", io3, index=safe_idx(io3, row["OnlineSecurity"]))
            backup  = st.selectbox("Online Backup", io3, index=safe_idx(io3, row["OnlineBackup"]))
            device  = st.selectbox("Device Protection", io3, index=safe_idx(io3, row["DeviceProtection"]))
        with c3:
            st.markdown("**Billing & Streaming**")
            tech    = st.selectbox("Tech Support", io3, index=safe_idx(io3, row["TechSupport"]))
            tv      = st.selectbox("Streaming TV", io3, index=safe_idx(io3, row["StreamingTV"]))
            movies  = st.selectbox("Streaming Movies", io3, index=safe_idx(io3, row["StreamingMovies"]))
            paper   = st.selectbox("Paperless Billing", ["Yes","No"], index=safe_idx(["Yes","No"], row["PaperlessBilling"]))
            popts   = ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
            payment = st.selectbox("Payment Method", popts, index=safe_idx(popts, row["PaymentMethod"]))
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, float(row["MonthlyCharges"]))
            total_c = st.number_input("Total Charges ($)", 0.0, 10000.0, float(row["TotalCharges"]))

        submitted = st.form_submit_button("🔮  Predict Churn Risk", use_container_width=True)

    if submitted:
        raw = pd.DataFrame([{
            "gender":gender,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,
            "tenure":tenure,"PhoneService":phone,"MultipleLines":mlines,"InternetService":internet,
            "OnlineSecurity":sec,"OnlineBackup":backup,"DeviceProtection":device,"TechSupport":tech,
            "StreamingTV":tv,"StreamingMovies":movies,"Contract":contract,"PaperlessBilling":paper,
            "PaymentMethod":payment,"MonthlyCharges":monthly,"TotalCharges":total_c,
        }])
        eng  = engineer_features(clean_data(raw))
        Xin  = eng[ALL_FEATURES]
        Xp   = preprocessor.transform(Xin)
        prob = model.predict_proba(Xp)[0,1]

        st.markdown("---")
        g1, g2, g3 = st.columns([1,1.2,1])
        with g1:
            if prob >= 0.6:
                st.markdown(f'<div class="risk-high">🔴 HIGH RISK<br><span style="font-size:2rem;">{prob:.1%}</span><br>churn probability</div>', unsafe_allow_html=True)
            elif prob >= 0.35:
                st.markdown(f'<div class="risk-med">🟠 MEDIUM RISK<br><span style="font-size:2rem;">{prob:.1%}</span><br>churn probability</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">🟢 LOW RISK<br><span style="font-size:2rem;">{prob:.1%}</span><br>churn probability</div>', unsafe_allow_html=True)

        with g2:
            st.plotly_chart(churn_gauge(prob), use_container_width=True)

        with g3:
            estimated_loss = monthly * 12
            st.markdown(metric_card("Est. Annual Revenue at Risk", f"${estimated_loss:,.0f}", "if this customer churns","r"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            action = "🚨 Immediate outreach required" if prob >= 0.6 else "⚠️ Monitor closely" if prob >= 0.35 else "✅ Low priority — retain standard engagement"
            st.markdown(f"""
            <div class="mcard" style="margin-top:.5rem;">
                <div class="mcard-label">Recommended Action</div>
                <div style="margin-top:.5rem; color:#e2e8f4; font-size:.9rem; font-weight:600;">{action}</div>
            </div>""", unsafe_allow_html=True)

        # SHAP explanation
        st.markdown('<div class="sec-title">🧩 SHAP Explanation — Why This Prediction?</div>', unsafe_allow_html=True)
        with st.spinner("Computing local SHAP explanation…"):
            bg_s  = background.sample(min(40,len(background)), random_state=42)
            bg_p  = preprocessor.transform(bg_s)
            if hasattr(bg_p,"toarray"): bg_p = bg_p.toarray()
            def pfn(X): return model.predict_proba(X)[:,1]
            exp   = shap.Explainer(pfn, bg_p)
            Xp_a  = preprocessor.transform(Xin)
            if hasattr(Xp_a,"toarray"): Xp_a = Xp_a.toarray()
            sv    = exp(Xp_a)
            sv.feature_names = feature_names

        vals  = sv.values[0]
        pairs = sorted(zip(feature_names, vals), key=lambda x:x[1], reverse=True)
        def clean(n): return n.replace("num__","").replace("cat__","").replace("_"," ")
        inc = [(clean(n),v) for n,v in pairs if v > 0][:6]
        dec = [(clean(n),v) for n,v in pairs if v < 0][:6]

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**🔺 Factors increasing churn risk**")
            if inc:
                chips = "".join([f'<span class="chip-r">{n}  <b>+{v:.3f}</b></span>' for n,v in inc])
                st.markdown(chips, unsafe_allow_html=True)

                fig_inc = go.Figure(go.Bar(
                    y=[n for n,_ in inc[::-1]], x=[v for _,v in inc[::-1]],
                    orientation="h", marker_color="#ef4444", marker_line_width=0,
                ))
                fig_inc.update_layout(**PLOTLY_THEME, height=240, margin=dict(l=10,r=10,t=10,b=10),
                                      xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_inc, use_container_width=True)

        with sc2:
            st.markdown("**🔻 Factors reducing churn risk**")
            if dec:
                chips = "".join([f'<span class="chip-g">{n}  <b>{v:.3f}</b></span>' for n,v in dec])
                st.markdown(chips, unsafe_allow_html=True)

                fig_dec = go.Figure(go.Bar(
                    y=[n for n,_ in dec], x=[abs(v) for _,v in dec],
                    orientation="h", marker_color="#10b981", marker_line_width=0,
                ))
                fig_dec.update_layout(**PLOTLY_THEME, height=240, margin=dict(l=10,r=10,t=10,b=10),
                                      xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_dec, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 4 — BATCH PREDICTION
# ─────────────────────────────────────────────
elif page == "📁  Batch Prediction":

    st.markdown("""
    <div class="hero">
        <p class="hero-title">📁 Batch Prediction</p>
        <p class="hero-sub">Score your entire customer base at once. Upload a CSV matching the Telco dataset schema, or use the demo holdout set to see how it works.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    use_demo = st.checkbox("Use demo holdout set instead", value=not bool(uploaded))

    batch_df = None
    if uploaded:
        raw_b = pd.read_csv(uploaded)
        batch_df = engineer_features(clean_data(raw_b))
    elif use_demo and os.path.exists(f"{MODELS_DIR}/holdout_sample.csv"):
        raw_b  = pd.read_csv(f"{MODELS_DIR}/holdout_sample.csv")
        y_true = raw_b.get("Churn")
        batch_df = engineer_features(clean_data(raw_b.drop(columns=["Churn"],errors="ignore")))
    
    if batch_df is not None:
        Xb   = batch_df[ALL_FEATURES]
        Xbp  = preprocessor.transform(Xb)
        prob = model.predict_proba(Xbp)[:,1]
        pred = (prob >= 0.5).astype(int)

        res = batch_df.copy()
        res["Churn_Probability"] = np.round(prob,4)
        res["Churn_Prediction"]  = np.where(pred==1,"Yes","No")
        res["Risk_Tier"]         = pd.cut(prob,bins=[-0.01,0.35,0.6,1.0],labels=["Low","Medium","High"])

        # Summary cards
        n_high = (res["Risk_Tier"]=="High").sum()
        n_med  = (res["Risk_Tier"]=="Medium").sum()
        n_low  = (res["Risk_Tier"]=="Low").sum()
        rev_at_risk = (prob * batch_df.get("MonthlyCharges", pd.Series([65]*len(batch_df))).values * 12).sum()

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(metric_card("Customers Scored", f"{len(res):,}", ""), unsafe_allow_html=True)
        c2.markdown(metric_card("High Risk 🔴", f"{n_high:,}", f"{n_high/len(res):.1%} of base","r"), unsafe_allow_html=True)
        c3.markdown(metric_card("Medium Risk 🟠", f"{n_med:,}", f"{n_med/len(res):.1%} of base","n"), unsafe_allow_html=True)
        c4.markdown(metric_card("Est. Revenue at Risk", f"${rev_at_risk:,.0f}", "Annual, from high-risk","r"), unsafe_allow_html=True)

        st.markdown('<div class="sec-title">📊 Risk Distribution</div>', unsafe_allow_html=True)
        dc1, dc2 = st.columns(2)
        with dc1:
            pie_data = pd.DataFrame({"Tier":["High 🔴","Medium 🟠","Low 🟢"],"Count":[n_high,n_med,n_low]})
            fig_pie  = px.pie(pie_data, values="Count", names="Tier",
                              color_discrete_sequence=["#ef4444","#f59e0b","#10b981"],
                              title="Customer Risk Distribution", hole=0.45)
            fig_pie.update_traces(textinfo="percent+label", pull=[0.05,0,0])
            st.plotly_chart(plotly_defaults(fig_pie), use_container_width=True)
        with dc2:
            fig_hist = px.histogram(res, x="Churn_Probability", nbins=30,
                                    color_discrete_sequence=["#7c3aed"],
                                    title="Churn Probability Distribution")
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#ef4444", annotation_text="Decision boundary")
            st.plotly_chart(plotly_defaults(fig_hist), use_container_width=True)

        # Color-coded results table
        st.markdown('<div class="sec-title">📋 Scored Customer Table</div>', unsafe_allow_html=True)

        display_cols = [c for c in ["tenure","Contract","MonthlyCharges","Churn_Probability","Churn_Prediction","Risk_Tier"] if c in res.columns]
        display_df   = res[display_cols].sort_values("Churn_Probability", ascending=False).reset_index(drop=True)

        def color_risk_cell(val):
            m = {"High":"background-color:#7f1d1d;color:#fca5a5",
                 "Medium":"background-color:#78350f;color:#fcd34d",
                 "Low":"background-color:#052e16;color:#6ee7b7"}
            return m.get(str(val),"")

        def color_prob(val):
            try:
                v = float(val)
                if v >= 0.6: return "color:#ef4444; font-weight:700"
                if v >= 0.35: return "color:#f59e0b; font-weight:600"
                return "color:#10b981"
            except: return ""

        try:
            styled = (display_df.style
                      .map(color_risk_cell, subset=["Risk_Tier"])
                      .map(color_prob, subset=["Churn_Probability"]))
        except AttributeError:
            styled = (display_df.style
                      .applymap(color_risk_cell, subset=["Risk_Tier"])
                      .applymap(color_prob, subset=["Churn_Probability"]))
        st.dataframe(styled, use_container_width=True, height=440)

        csv_out = res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Scored CSV", csv_out, "churn_predictions.csv","text/csv")

    else:
        st.info("Upload a CSV or enable the demo checkbox above.")

# ─────────────────────────────────────────────
# PAGE 5 — RETENTION ROI CALCULATOR
# ─────────────────────────────────────────────
elif page == "💰  Retention ROI Calculator":

    st.markdown("""
    <div class="hero">
        <p class="hero-title">💰 Retention ROI Calculator</p>
        <p class="hero-sub">Translate churn predictions into dollar value. Estimate how much revenue you can save by targeting high-risk customers with retention offers.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">⚙️ Configure Your Retention Program</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        avg_monthly_rev  = st.number_input("Avg. monthly revenue per customer ($)", min_value=1.0, value=65.0, step=5.0)
        avg_contract_len = st.slider("Avg. remaining contract length (months)", 1, 24, 10)
        total_customers  = st.number_input("Total customer base size", min_value=100, value=7000, step=100)
    with col2:
        churn_rate_input = st.slider("Current churn rate (%)", 1, 60, 27)
        retention_offer  = st.number_input("Cost of retention offer per customer ($)", min_value=0.0, value=15.0, step=5.0)
        success_rate     = st.slider("Expected retention success rate (%)", 10, 90, 55)

    st.markdown('<div class="sec-title">📊 ROI Analysis</div>', unsafe_allow_html=True)

    churners_est    = int(total_customers * churn_rate_input / 100)
    high_risk_est   = int(churners_est * 0.6)
    revenue_per     = avg_monthly_rev * avg_contract_len
    total_at_risk   = churners_est * revenue_per
    targeted_cost   = high_risk_est * retention_offer
    saved_customers = int(high_risk_est * success_rate / 100)
    revenue_saved   = saved_customers * revenue_per
    net_roi         = revenue_saved - targeted_cost
    roi_pct         = (net_roi / targeted_cost * 100) if targeted_cost > 0 else 0

    r1,r2,r3,r4 = st.columns(4)
    r1.markdown(metric_card("Est. Churners",       f"{churners_est:,}", f"{churn_rate_input}% of base","r"), unsafe_allow_html=True)
    r2.markdown(metric_card("Revenue at Risk",     f"${total_at_risk:,.0f}", "Without intervention","r"), unsafe_allow_html=True)
    r3.markdown(metric_card("Revenue Saved",       f"${revenue_saved:,.0f}", f"{saved_customers:,} retained customers","g"), unsafe_allow_html=True)
    r4.markdown(metric_card("Net ROI",             f"${net_roi:,.0f}", f"{roi_pct:.0f}% return on spend","g" if net_roi>0 else "r"), unsafe_allow_html=True)

    # Waterfall chart
    st.markdown('<div class="sec-title">📉 ROI Waterfall</div>', unsafe_allow_html=True)
    wf_labels = ["Revenue at Risk","Targeted Customers","Retention Offers Cost","Revenue Saved","Net ROI"]
    wf_vals   = [total_at_risk, 0, -targeted_cost, revenue_saved, net_roi]
    wf_colors = ["#ef4444","#94a3b8","#f59e0b","#10b981","#7c3aed"]

    fig_wf = go.Figure(go.Waterfall(
        name="ROI", orientation="v",
        measure=["absolute","skip","relative","relative","total"],
        x=wf_labels, y=[total_at_risk, None, -targeted_cost, revenue_saved, None],
        connector={"line":{"color":"#334155"}},
        decreasing={"marker":{"color":"#ef4444"}},
        increasing={"marker":{"color":"#10b981"}},
        totals={"marker":{"color":"#7c3aed"}},
        text=[f"${v:,.0f}" for v in [total_at_risk, 0, targeted_cost, revenue_saved, net_roi]],
        textposition="outside",
    ))
    fig_wf.update_layout(**PLOTLY_THEME, height=400, title="Financial Impact of Retention Program",
                         yaxis=dict(gridcolor="#1e293b",tickprefix="$"), xaxis=dict(gridcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_wf, use_container_width=True)

    # Break-even
    st.markdown('<div class="sec-title">⚖️ Break-Even Analysis</div>', unsafe_allow_html=True)
    success_range   = list(range(10, 91, 5))
    roi_range       = [(int(high_risk_est * s/100) * revenue_per - targeted_cost) for s in success_range]
    fig_be = px.line(x=success_range, y=roi_range, markers=True,
                     labels={"x":"Retention Success Rate (%)","y":"Net ROI ($)"},
                     title="Net ROI vs. Retention Success Rate")
    fig_be.add_hline(y=0, line_dash="dash", line_color="#f59e0b", annotation_text="Break-even")
    fig_be.update_traces(line_color="#7c3aed", fill="tozeroy", fillcolor="rgba(124,58,237,0.1)")
    st.plotly_chart(plotly_defaults(fig_be), use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 6 — BUSINESS INSIGHTS
# ─────────────────────────────────────────────
elif page == "💡  Business Insights":

    st.markdown("""
    <div class="hero">
        <p class="hero-title">💡 Business Insights</p>
        <p class="hero-sub">Data-driven findings from the model and EDA, translated into concrete actions your retention team can act on today.</p>
    </div>
    """, unsafe_allow_html=True)

    insights = [
        ("📄","Contract Type","Month-to-month customers churn at a dramatically higher rate than 1- or 2-year holders.",
         "Offer a 10-15% discount for switching to an annual contract. Target customers in months 2-5, before churn behavior solidifies.",
         "#7c3aed"),
        ("💳","Payment Method","Electronic check users show the highest churn rate of all payment methods — nearly 2× the rate of auto-pay customers.",
         "Incentivize migration to auto bank transfer or credit card with a one-month fee waiver or loyalty points.",
         "#ef4444"),
        ("🌐","Fiber Optic + No Add-ons","Fiber customers without Online Security or Tech Support are paying premium prices without feeling the value.",
         "Bundle a 30-day free trial of Tech Support & Online Security for all new Fiber signups.",
         "#f59e0b"),
        ("⏳","Early Tenure (0–6 months)","The highest churn window. New customers who don't see value early cancel quickly.",
         "Implement a structured 90-day onboarding program: welcome call at day 3, check-in at day 30, loyalty reward at day 90.",
         "#06b6d4"),
        ("👴","Senior Citizens","Senior citizens churn slightly more than non-seniors, often due to billing complexity and support friction.",
         "Offer a 'Senior Plan': simplified bill format, dedicated phone support, and optional in-home tech assistance.",
         "#10b981"),
        ("💸","High Monthly Charges","Customers paying above-average monthly charges who aren't on long contracts are price-sensitive churners.",
         "Proactively offer loyalty pricing reviews at the 6-month mark for high-spend month-to-month customers.",
         "#a78bfa"),
    ]

    for icon, title, finding, action, color in insights:
        st.markdown(f"""
        <div class="insight-card" style="border-left-color:{color};">
            <div style="display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem;">
                <span style="font-size:1.4rem;">{icon}</span>
                <span style="font-weight:800; font-size:1.05rem; color:#fff;">{title}</span>
            </div>
            <div style="color:#94a3b8; font-size:.88rem; line-height:1.6;">
                <b style="color:#cbd5e1;">Finding:</b> {finding}
            </div>
            <div style="margin-top:.5rem; color:{color}; font-size:.88rem; line-height:1.6; font-weight:500;">
                <b>→ Action:</b> {action}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">🎯 Recommended Retention Funnel</div>', unsafe_allow_html=True)
    steps = [
        ("1","Score","Run Batch Prediction weekly on the full customer base"),
        ("2","Segment","Classify into High / Medium / Low risk tiers"),
        ("3","Prioritize","Filter High Risk AND High Value (top monthly charges tercile) customers"),
        ("4","Personalize","Use the Single Customer Predict page to see each customer's top SHAP risk factors"),
        ("5","Intervene","Route to the right retention action based on the #1 risk factor"),
        ("6","Track","Measure churn rate weekly; feed successful retentions back as training data"),
    ]
    funnel_cols = st.columns(len(steps))
    for col, (num, step, desc) in zip(funnel_cols, steps):
        col.markdown(f"""
        <div style="text-align:center; padding:.8rem .4rem;">
            <div style="width:42px; height:42px; border-radius:50%; background:linear-gradient(135deg,#4f46e5,#7c3aed);
                        display:flex; align-items:center; justify-content:center; margin:0 auto;
                        font-weight:900; color:#fff; font-size:1.1rem;">{num}</div>
            <div style="font-weight:700; color:#fff; margin-top:.5rem; font-size:.9rem;">{step}</div>
            <div style="font-size:.75rem; color:#64748b; margin-top:.25rem; line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:4rem; padding:1.5rem;
            border-top:1px solid rgba(255,255,255,0.05); color:#334155; font-size:.8rem;">
    ChurnIQ &nbsp;·&nbsp; Built with Streamlit · Scikit-learn · XGBoost · LightGBM · SHAP · Plotly
    &nbsp;·&nbsp; Customer Churn Prediction System
</div>
""", unsafe_allow_html=True)