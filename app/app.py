import streamlit as st
import pandas as pd
import joblib
import subprocess
import sys
from pathlib import Path
from PIL import Image

# Paths
DATA_PATH = Path("data/processed/telco_clean.csv")
MODEL_PATH = Path("reports/models/logreg_churn.pkl")
SHAP_IMG_PATH = Path("reports/figures/shap_feature_importance.png")
PIPELINE_SCRIPT = Path("src/run_pipeline.py")
XTRAIN_PATH = Path("data/processed/X_train.csv")

def ensure_artifacts():
    needed = [
        DATA_PATH,
        MODEL_PATH,
        SHAP_IMG_PATH,
        XTRAIN_PATH
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        st.warning("Missing artifacts detected. Building pipeline (first run may take 1–3 minutes)...")
        result = subprocess.run([sys.executable, str(PIPELINE_SCRIPT)], capture_output=True, text=True)
        if result.returncode != 0:
            st.error("Pipeline build failed. See logs below.")
            st.code(result.stdout + "\n" + result.stderr)
            st.stop()
        st.success("Pipeline built successfully. Reloading app…")
        st.rerun()

ensure_artifacts()

st.set_page_config(
    page_title="Customer Churn Decision Platform",
    layout="wide"
)

st.title("📊 Customer Churn Decision Intelligence Platform")

st.markdown(
    """
    This app analyzes customer behavior to:
    - Understand churn risk
    - Explain *why* customers churn
    - Support data-driven business decisions
    """
)

# -----------------------------
# Load data & model
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# -----------------------------
# High-level metrics
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", len(df))

with col2:
    churn_rate = df["churn"].mean()
    st.metric("Churn Rate", f"{churn_rate:.1%}")

with col3:
    avg_tenure = df["tenure"].mean()
    st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")

st.divider()

# -----------------------------
# SHAP explanation
# -----------------------------
st.subheader("🔍 What Drives Customer Churn?")

if SHAP_IMG_PATH.exists():
    shap_img = Image.open(SHAP_IMG_PATH)
    st.image(shap_img, caption="Global Feature Importance (SHAP)", width='stretch')
else:
    st.warning("SHAP explanation image not found.")

st.markdown(
    """
    **How to read this:**
    - Higher bars = stronger influence on churn
    - Features at the top matter most
    - This helps decision-makers focus on the *right levers*
    """
)

st.divider()

# -----------------------------
# Simple takeaway
# -----------------------------
st.subheader("📌 Decision Insight")

st.markdown(
    """
    Customers with **short tenure**, **month-to-month contracts**, and **higher monthly charges**
    show significantly higher churn risk.

    **Business action:**  
    Prioritize retention offers for new customers on month-to-month plans.
    """
)
# -----------------------------
# Churn Risk Simulator
# -----------------------------
st.divider()
st.subheader("🧪 Churn Risk Simulator")

st.markdown(
    """
    Adjust customer characteristics below to estimate **churn probability**
    and see how risk changes.
    """
)

with st.form("churn_simulator"):
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 70)

    with col2:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"]
        )

    with col3:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])

    submitted = st.form_submit_button("Estimate Churn Risk")

if submitted:
    # Build a single-row input matching training features
    row = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [tenure * monthly_charges],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "PaperlessBilling_Yes": [1 if paperless == "Yes" else 0],
        "Contract_One year": [1 if contract == "One year" else 0],
        "Contract_Two year": [1 if contract == "Two year" else 0],
        "InternetService_Fiber optic": [1 if internet == "Fiber optic" else 0],
        "InternetService_No": [1 if internet == "No" else 0],
    })

    # Align columns with training data
    X_train = pd.read_csv("data/processed/X_train.csv")
    row = row.reindex(columns=X_train.columns, fill_value=0)

    churn_prob = model.predict_proba(row)[0, 1]

    st.metric("Estimated Churn Risk", f"{churn_prob:.1%}")

    if churn_prob >= 0.5:
        st.error("⚠️ High churn risk — proactive retention recommended.")
    else:
        st.success("✅ Low churn risk — customer likely to stay.")
