"""Simulate real-time prediction requests to the running FastAPI service.

Sends real employee records from the IBM HR dataset to /predict
and builds up data/predictions.csv for drift monitoring.

Usage:
    python simulate.py
"""

import time
import requests
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = "http://localhost:9696/predict"
DATA_DIR = "../data"

# Categorical columns to send as-is (reverse of Stage 02 encoding)
# We use the raw dataset so the API receives real human-readable values
RAW_DATA_PATH = f"{DATA_DIR}/IBM_emp_attrition.csv"


def load_raw_data(n_rows: int = 200) -> pd.DataFrame:
    """Load a sample of raw IBM HR data for simulation."""
    print(f"📥 Loading {n_rows} employee records for simulation ...")
    df = pd.read_csv(RAW_DATA_PATH, sep=";")

    # Drop columns not needed by the API
    drop_cols = [
        "EmployeeCount",
        "StandardHours",
        "Over18",
        "EmployeeNumber",
        "Attrition",
    ]
    df = df.drop(columns=drop_cols)
    df = df.sample(n=n_rows, random_state=42).reset_index(drop=True)
    print(f"✓ Loaded {len(df)} rows")
    return df


def build_payload(row: pd.Series) -> dict:
    """Convert a raw dataset row to API payload format."""
    return {
        "Age": int(row["Age"]),
        "BusinessTravel": str(row["BusinessTravel"]),
        "DailyRate": int(row["DailyRate"]),
        "Department": str(row["Department"]),
        "DistanceFromHome": int(row["DistanceFromHome"]),
        "Education": int(row["Education"]),
        "EducationField": str(row["EducationField"]),
        "EnvironmentSatisfaction": int(row["EnvironmentSatisfaction"]),
        "Gender": str(row["Gender"]),
        "HourlyRate": int(row["HourlyRate"]),
        "JobInvolvement": int(row["JobInvolvement"]),
        "JobLevel": int(row["JobLevel"]),
        "JobRole": str(row["JobRole"]),
        "JobSatisfaction": int(row["JobSatisfaction"]),
        "MaritalStatus": str(row["MaritalStatus"]),
        "MonthlyIncome": int(row["MonthlyIncome"]),
        "MonthlyRate": int(row["MonthlyRate"]),
        "NumCompaniesWorked": int(row["NumCompaniesWorked"]),
        "OverTime": str(row["OverTime"]),
        "PercentSalaryHike": int(row["PercentSalaryHike"]),
        "PerformanceRating": int(row["PerformanceRating"]),
        "RelationshipSatisfaction": int(row["RelationshipSatisfaction"]),
        "StockOptionLevel": int(row["StockOptionLevel"]),
        "TotalWorkingYears": int(row["TotalWorkingYears"]),
        "TrainingTimesLastYear": int(row["TrainingTimesLastYear"]),
        "WorkLifeBalance": int(row["WorkLifeBalance"]),
        "YearsAtCompany": int(row["YearsAtCompany"]),
        "YearsInCurrentRole": int(row["YearsInCurrentRole"]),
        "YearsSinceLastPromotion": int(row["YearsSinceLastPromotion"]),
        "YearsWithCurrManager": int(row["YearsWithCurrManager"]),
    }


def simulate_requests(df: pd.DataFrame, sleep_s: float = 0.05):
    """Send each row to the prediction API."""
    success = 0
    failed = 0

    for i, row in df.iterrows():
        payload = build_payload(row)
        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            success += 1

            if (i + 1) % 50 == 0:
                print(
                    f"   Progress: {i + 1}/{len(df)}  "
                    f"(last: prob={data['probability']:.3f}, "
                    f"risk={data['risk_level']})"
                )

        except Exception as e:
            failed += 1
            print(f"⚠️  Request {i} failed: {e}")

        time.sleep(sleep_s)

    return success, failed


def main():
    print("\n🚀 Starting simulation ...\n")
    df = load_raw_data(n_rows=200)
    success, failed = simulate_requests(df)
    print(f"\n✅ Simulation complete: {success} succeeded, {failed} failed")
    print(f"📄 Predictions logged to: 05-monitoring/data/predictions.csv")
    print(f"Next step: python monitor.py")


if __name__ == "__main__":
    main()
