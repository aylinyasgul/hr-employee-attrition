"""Streamlit UI for the Employee Attrition Prediction model.

Deployable on Streamlit Community Cloud straight from this repo. The app is
self-contained: it trains the XGBoost model at startup from the committed,
pre-processed training data (data/processed/train.csv) and reuses the
committed StandardScaler so that a new employee record is transformed with the
exact same pipeline as Stage 02 of the project.

Entry point for Streamlit Cloud: streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "processed"


@st.cache_resource(show_spinner=False)
def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
THRESHOLD_LOW = CONFIG["risk_thresholds"]["low"]     # < 0.35  -> Low
THRESHOLD_HIGH = CONFIG["risk_thresholds"]["high"]   # >= 0.60 -> High
MODEL_VERSION = CONFIG["api"]["model_version"]


# ---------------------------------------------------------------------------
# Artifacts: scaler, feature columns, and the model (trained on startup)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training model…")
def load_artifacts():
    scaler = joblib.load(DATA_DIR / "scaler.joblib")
    cols_to_scale = joblib.load(DATA_DIR / "cols_to_scale.joblib")
    feature_columns = [
        line.strip()
        for line in (DATA_DIR / "feature_columns.txt").read_text().splitlines()
        if line.strip()
    ]

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    X_train = train_df[feature_columns]
    y_train = train_df["Attrition"]

    params = {
        "n_estimators": CONFIG["model"]["n_estimators"],
        "max_depth": CONFIG["model"]["max_depth"],
        "learning_rate": CONFIG["model"]["learning_rate"],
        "scale_pos_weight": CONFIG["model"]["scale_pos_weight"],
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": CONFIG["model"]["random_state"],
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    return model, scaler, cols_to_scale, feature_columns


MODEL, SCALER, COLS_TO_SCALE, FEATURE_COLUMNS = load_artifacts()


# ---------------------------------------------------------------------------
# One-hot maps — must mirror Stage 02 exactly
# ---------------------------------------------------------------------------
OHE_MAPS = {
    "BusinessTravel": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": [
        "Human Resources",
        "Life Sciences",
        "Marketing",
        "Medical",
        "Other",
        "Technical Degree",
    ],
    "JobRole": [
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ],
    "MaritalStatus": ["Divorced", "Married", "Single"],
}


def preprocess(record: dict) -> pd.DataFrame:
    """Turn a raw employee record into the scaled 53-column model input."""
    d = dict(record)

    d["OverTime"] = 1 if d["OverTime"] == "Yes" else 0
    d["Gender"] = 1 if d["Gender"] == "Male" else 0

    # Engineered features (Stage 02)
    d["PromotionStagnationRatio"] = d["YearsSinceLastPromotion"] / max(
        1, d["YearsAtCompany"]
    )
    d["WorkloadPayPressure"] = d["OverTime"] * d["MonthlyIncome"]
    d["AverageSatisfaction"] = float(
        np.mean(
            [
                d["JobSatisfaction"],
                d["EnvironmentSatisfaction"],
                d["RelationshipSatisfaction"],
                d["WorkLifeBalance"],
            ]
        )
    )
    years = d["YearsAtCompany"]
    d["TenureBucket"] = 0 if years <= 2 else (1 if years <= 7 else 2)

    # One-hot encode the categoricals
    for col, categories in OHE_MAPS.items():
        value = d.pop(col)
        for cat in categories:
            d[f"{col}_{cat}"] = 1 if value == cat else 0

    row = {col: d.get(col, 0) for col in FEATURE_COLUMNS}
    frame = pd.DataFrame([row])[FEATURE_COLUMNS]

    # Scale the numeric/engineered columns with the fitted StandardScaler
    frame[COLS_TO_SCALE] = SCALER.transform(frame[COLS_TO_SCALE])
    return frame


def risk_tier(probability: float) -> str:
    if probability >= THRESHOLD_HIGH:
        return "High"
    if probability >= THRESHOLD_LOW:
        return "Medium"
    return "Low"


RISK_COLOR = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🧑‍💼",
    layout="wide",
)

st.title("🧑‍💼 Employee Attrition Predictor")
st.caption(
    "IE University · MBD 2025 · Group 6 — predicts an employee's voluntary "
    "attrition risk with an XGBoost model trained on the IBM HR Analytics dataset."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        f"""
- **Model:** XGBoost (v{MODEL_VERSION})
- **Risk tiers:** Low `< {THRESHOLD_LOW:g}` · Medium `{THRESHOLD_LOW:g}–{THRESHOLD_HIGH:g}` · High `≥ {THRESHOLD_HIGH:g}`
- **Engineered features:** PromotionStagnationRatio, WorkloadPayPressure, AverageSatisfaction, TenureBucket

Enter an employee's details and press **Predict**.
        """
    )
    st.markdown("[Source on GitHub →](https://github.com/aylinyasgul/hr-employee-attrition)")

st.subheader("Employee details")

with st.form("employee_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics & role**")
        Age = st.slider("Age", 18, 65, 35)
        Gender = st.selectbox("Gender", ["Male", "Female"])
        MaritalStatus = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
        Department = st.selectbox(
            "Department", ["Sales", "Research & Development", "Human Resources"]
        )
        JobRole = st.selectbox("Job role", OHE_MAPS["JobRole"])
        JobLevel = st.slider("Job level", 1, 5, 1)
        Education = st.slider("Education (1–5)", 1, 5, 3)
        EducationField = st.selectbox("Education field", OHE_MAPS["EducationField"])
        DistanceFromHome = st.slider("Distance from home (km)", 0, 30, 5)

    with c2:
        st.markdown("**Work & compensation**")
        BusinessTravel = st.selectbox(
            "Business travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        )
        OverTime = st.selectbox("Over time", ["No", "Yes"])
        MonthlyIncome = st.number_input("Monthly income", 1000, 25000, 5000, step=100)
        DailyRate = st.number_input("Daily rate", 0, 1500, 800, step=10)
        HourlyRate = st.number_input("Hourly rate", 0, 120, 60, step=1)
        MonthlyRate = st.number_input("Monthly rate", 0, 30000, 15000, step=100)
        PercentSalaryHike = st.slider("Percent salary hike", 0, 30, 13)
        StockOptionLevel = st.slider("Stock option level", 0, 3, 0)
        NumCompaniesWorked = st.slider("Num companies worked", 0, 10, 2)

    with c3:
        st.markdown("**Tenure & satisfaction**")
        TotalWorkingYears = st.slider("Total working years", 0, 40, 10)
        YearsAtCompany = st.slider("Years at company", 0, 40, 5)
        YearsInCurrentRole = st.slider("Years in current role", 0, 20, 3)
        YearsSinceLastPromotion = st.slider("Years since last promotion", 0, 15, 1)
        YearsWithCurrManager = st.slider("Years with current manager", 0, 20, 3)
        TrainingTimesLastYear = st.slider("Trainings last year", 0, 6, 2)
        JobSatisfaction = st.slider("Job satisfaction (1–4)", 1, 4, 3)
        EnvironmentSatisfaction = st.slider("Environment satisfaction (1–4)", 1, 4, 3)
        RelationshipSatisfaction = st.slider("Relationship satisfaction (1–4)", 1, 4, 3)
        JobInvolvement = st.slider("Job involvement (1–4)", 1, 4, 3)
        PerformanceRating = st.slider("Performance rating (1–4)", 1, 4, 3)
        WorkLifeBalance = st.slider("Work-life balance (1–4)", 1, 4, 3)

    submitted = st.form_submit_button("🔮 Predict attrition risk", use_container_width=True)

if submitted:
    record = {
        "Age": Age,
        "BusinessTravel": BusinessTravel,
        "DailyRate": DailyRate,
        "Department": Department,
        "DistanceFromHome": DistanceFromHome,
        "Education": Education,
        "EducationField": EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender": Gender,
        "HourlyRate": HourlyRate,
        "JobInvolvement": JobInvolvement,
        "JobLevel": JobLevel,
        "JobRole": JobRole,
        "JobSatisfaction": JobSatisfaction,
        "MaritalStatus": MaritalStatus,
        "MonthlyIncome": MonthlyIncome,
        "MonthlyRate": MonthlyRate,
        "NumCompaniesWorked": NumCompaniesWorked,
        "OverTime": OverTime,
        "PercentSalaryHike": PercentSalaryHike,
        "PerformanceRating": PerformanceRating,
        "RelationshipSatisfaction": RelationshipSatisfaction,
        "StockOptionLevel": StockOptionLevel,
        "TotalWorkingYears": TotalWorkingYears,
        "TrainingTimesLastYear": TrainingTimesLastYear,
        "WorkLifeBalance": WorkLifeBalance,
        "YearsAtCompany": YearsAtCompany,
        "YearsInCurrentRole": YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
    }

    features = preprocess(record)
    probability = float(MODEL.predict_proba(features)[0, 1])
    tier = risk_tier(probability)
    color = RISK_COLOR[tier]

    st.subheader("Result")
    m1, m2 = st.columns([1, 2])
    with m1:
        st.markdown(
            f"""
<div style="border-radius:12px;padding:24px;background:{color};color:white;text-align:center">
  <div style="font-size:0.9rem;opacity:0.9;">Attrition risk</div>
  <div style="font-size:2.4rem;font-weight:700;">{tier}</div>
  <div style="font-size:1.1rem;">{probability:.1%} probability</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.progress(min(probability, 1.0))
        st.metric("Attrition probability", f"{probability:.1%}")
        if tier == "High":
            st.warning("High risk — consider a retention conversation and review "
                       "compensation, workload, and growth path.")
        elif tier == "Medium":
            st.info("Medium risk — monitor engagement and satisfaction signals.")
        else:
            st.success("Low risk — no immediate action indicated.")

    with st.expander("Model input (post-preprocessing)"):
        st.dataframe(features.T.rename(columns={0: "value"}))
