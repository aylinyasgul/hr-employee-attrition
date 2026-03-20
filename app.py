"""FastAPI service for Employee Attrition Prediction.

Loads model from local models/ directory (no MLflow server needed).
Designed to run inside Docker as a self-contained service.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

THRESHOLD_HIGH = 0.60
THRESHOLD_MEDIUM = 0.35

# Load model and feature columns at module level
model = joblib.load(MODEL_DIR / "model.joblib")
FEATURE_COLUMNS = joblib.load(MODEL_DIR / "feature_columns.joblib")
MODEL_VERSION = "1.0.0"

print(f"[startup] Model loaded from {MODEL_DIR}/")


def get_risk_tier(probability: float) -> str:
    if probability >= THRESHOLD_HIGH:
        return "High"
    elif probability >= THRESHOLD_MEDIUM:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class EmployeeRequest(BaseModel):
    Age: int = Field(..., ge=18, le=65)
    BusinessTravel: str = Field(
        ..., description="Non-Travel | Travel_Rarely | Travel_Frequently"
    )
    DailyRate: int = Field(..., ge=0)
    Department: str = Field(
        ..., description="Sales | Research & Development | Human Resources"
    )
    DistanceFromHome: int = Field(..., ge=0)
    Education: int = Field(..., ge=1, le=5)
    EducationField: str = Field(
        ...,
        description="Life Sciences | Medical | Marketing | Technical Degree | Human Resources | Other",
    )
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    Gender: str = Field(..., description="Male | Female")
    HourlyRate: int = Field(..., ge=0)
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobRole: str = Field(
        ..., description="e.g. Sales Executive | Research Scientist | ..."
    )
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MaritalStatus: str = Field(..., description="Single | Married | Divorced")
    MonthlyIncome: int = Field(..., ge=0)
    MonthlyRate: int = Field(..., ge=0)
    NumCompaniesWorked: int = Field(..., ge=0)
    OverTime: str = Field(..., description="Yes | No")
    PercentSalaryHike: int = Field(..., ge=0)
    PerformanceRating: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0)
    TrainingTimesLastYear: int = Field(..., ge=0)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0)
    YearsInCurrentRole: int = Field(..., ge=0)
    YearsSinceLastPromotion: int = Field(..., ge=0)
    YearsWithCurrManager: int = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 35,
                "BusinessTravel": "Travel_Frequently",
                "DailyRate": 800,
                "Department": "Sales",
                "DistanceFromHome": 20,
                "Education": 3,
                "EducationField": "Life Sciences",
                "EnvironmentSatisfaction": 2,
                "Gender": "Male",
                "HourlyRate": 60,
                "JobInvolvement": 2,
                "JobLevel": 1,
                "JobRole": "Sales Representative",
                "JobSatisfaction": 2,
                "MaritalStatus": "Single",
                "MonthlyIncome": 3500,
                "MonthlyRate": 15000,
                "NumCompaniesWorked": 5,
                "OverTime": "Yes",
                "PercentSalaryHike": 11,
                "PerformanceRating": 3,
                "RelationshipSatisfaction": 2,
                "StockOptionLevel": 0,
                "TotalWorkingYears": 6,
                "TrainingTimesLastYear": 1,
                "WorkLifeBalance": 1,
                "YearsAtCompany": 2,
                "YearsInCurrentRole": 1,
                "YearsSinceLastPromotion": 1,
                "YearsWithCurrManager": 0,
            }
        }
    }


class PredictionResponse(BaseModel):
    attrition: bool
    probability: float
    risk_level: str
    model_version: str


# ---------------------------------------------------------------------------
# Feature engineering — mirrors Stage 02 exactly
# ---------------------------------------------------------------------------
def preprocess(emp: EmployeeRequest) -> pd.DataFrame:
    d = emp.model_dump()

    d["OverTime"] = 1 if d["OverTime"] == "Yes" else 0
    d["Gender"] = 1 if d["Gender"] == "Male" else 0

    d["PromotionStagnationRatio"] = d["YearsSinceLastPromotion"] / max(
        1, d["YearsAtCompany"]
    )
    d["WorkloadPayPressure"] = d["OverTime"] * d["MonthlyIncome"]
    d["AverageSatisfaction"] = np.mean(
        [
            d["JobSatisfaction"],
            d["EnvironmentSatisfaction"],
            d["RelationshipSatisfaction"],
            d["WorkLifeBalance"],
        ]
    )
    years = d["YearsAtCompany"]
    d["TenureBucket"] = 0 if years <= 2 else (1 if years <= 7 else 2)

    ohe_maps = {
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

    for col, categories in ohe_maps.items():
        value = d.pop(col)
        for cat in categories:
            d[f"{col}_{cat}"] = 1 if value == cat else 0

    row = {col: d.get(col, 0) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row])[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predicts employee attrition probability.",
    version=MODEL_VERSION,
)


@app.get("/")
def root():
    return {"message": "Employee Attrition Prediction API — see /docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeRequest):
    try:
        features = preprocess(employee)
        prob = float(model.predict_proba(features)[0, 1])
        attrition = prob >= THRESHOLD_MEDIUM

        return PredictionResponse(
            attrition=attrition,
            probability=round(prob, 4),
            risk_level=get_risk_tier(prob),
            model_version=MODEL_VERSION,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=False)
