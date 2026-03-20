"""API tests for the Employee Attrition prediction service."""
import requests

BASE_URL = "http://localhost:9696"

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
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


def test_predict_high_risk():
    resp = requests.post(f"{BASE_URL}/predict", json=HIGH_RISK_EMPLOYEE)
    assert resp.status_code == 200, f"{resp.status_code} — {resp.text}"
    data = resp.json()
    assert "attrition"   in data
    assert "probability" in data
    assert "risk_level"  in data
    assert 0 <= data["probability"] <= 1
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_low_risk():
    resp = requests.post(f"{BASE_URL}/predict", json=LOW_RISK_EMPLOYEE)
    assert resp.status_code == 200, f"{resp.status_code} — {resp.text}"
    data = resp.json()
    assert 0 <= data["probability"] <= 1
