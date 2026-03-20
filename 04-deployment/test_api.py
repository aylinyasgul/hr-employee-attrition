"""External API tests for the running Employee Attrition FastAPI service.

Requires the server already running (python app.py showing Uvicorn on port 9696).
Issues real HTTP requests — same pattern as the class deployment example.
"""
import requests

BASE_URL = "http://localhost:9696"

# High-risk employee profile (OverTime, low income, frequent travel, low satisfaction)
HIGH_RISK_EMPLOYEE = {
    "Age": 28,
    "BusinessTravel": "Travel_Frequently",
    "DailyRate": 500,
    "Department": "Sales",
    "DistanceFromHome": 25,
    "Education": 2,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 1,
    "Gender": "Male",
    "HourlyRate": 45,
    "JobInvolvement": 2,
    "JobLevel": 1,
    "JobRole": "Sales Representative",
    "JobSatisfaction": 1,
    "MaritalStatus": "Single",
    "MonthlyIncome": 2500,
    "MonthlyRate": 10000,
    "NumCompaniesWorked": 4,
    "OverTime": "Yes",
    "PercentSalaryHike": 10,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 2,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 4,
    "TrainingTimesLastYear": 1,
    "WorkLifeBalance": 1,
    "YearsAtCompany": 1,
    "YearsInCurrentRole": 0,
    "YearsSinceLastPromotion": 0,
    "YearsWithCurrManager": 0,
}

# Low-risk employee profile (no overtime, high income, high satisfaction)
LOW_RISK_EMPLOYEE = {
    "Age": 45,
    "BusinessTravel": "Non-Travel",
    "DailyRate": 1200,
    "Department": "Research & Development",
    "DistanceFromHome": 3,
    "Education": 4,
    "EducationField": "Medical",
    "EnvironmentSatisfaction": 4,
    "Gender": "Female",
    "HourlyRate": 90,
    "JobInvolvement": 4,
    "JobLevel": 4,
    "JobRole": "Research Director",
    "JobSatisfaction": 4,
    "MaritalStatus": "Married",
    "MonthlyIncome": 15000,
    "MonthlyRate": 25000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 18,
    "PerformanceRating": 4,
    "RelationshipSatisfaction": 4,
    "StockOptionLevel": 3,
    "TotalWorkingYears": 20,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 4,
    "YearsAtCompany": 15,
    "YearsInCurrentRole": 8,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 7,
}


def test_health_endpoint():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code}"
    data = resp.json()
    assert data.get("status") == "ok"
    assert isinstance(data.get("run_id"), str) and len(data["run_id"]) > 5
    print("✓ /health passed")


def test_predict_high_risk():
    resp = requests.post(f"{BASE_URL}/predict", json=HIGH_RISK_EMPLOYEE)
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    data = resp.json()
    assert "attrition"     in data
    assert "probability"   in data
    assert "risk_level"    in data
    assert "model_version" in data
    assert isinstance(data["probability"], float)
    assert 0 <= data["probability"] <= 1
    assert data["risk_level"] in ["Low", "Medium", "High"]
    print(f"✓ /predict (high-risk) passed — probability: {data['probability']:.3f}, risk: {data['risk_level']}")


def test_predict_low_risk():
    resp = requests.post(f"{BASE_URL}/predict", json=LOW_RISK_EMPLOYEE)
    assert resp.status_code == 200, f"Unexpected status: {resp.status_code} — {resp.text}"
    data = resp.json()
    assert isinstance(data["probability"], float)
    assert 0 <= data["probability"] <= 1
    print(f"✓ /predict (low-risk)  passed — probability: {data['probability']:.3f}, risk: {data['risk_level']}")


if __name__ == "__main__":
    test_health_endpoint()
    test_predict_high_risk()
    test_predict_low_risk()
    print("\n✅ All tests passed.")
