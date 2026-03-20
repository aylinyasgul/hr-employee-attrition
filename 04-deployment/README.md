# 04 – Deployment

Serve the trained XGBoost attrition model as a REST API using FastAPI + MLflow.

---

## What This Stage Does

- `train.py` — retrains the XGBoost model (same params as Stage 03 winner), logs to MLflow, writes `run_id.txt`
- `app.py` — FastAPI service that loads the model from MLflow and exposes `/predict`
- `test_api.py` — tests the running API with a high-risk and low-risk employee profile

---

## Folder Structure

```
04-deployment/
├── train.py          # Train & log model artifact
├── app.py            # FastAPI service
├── test_api.py       # API tests
├── requirements.txt  # Dependencies
├── run_id.txt        # Generated after train.py runs
└── README.md         # This file
```

---

## How to Run

### Step 1 — Start MLflow server (new terminal)

```bash
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///backend.db --host 127.0.0.1 --port 5001
```

### Step 2 — Train the model (new terminal)

```bash
source .venv/bin/activate
cd 04-deployment
python train.py
```

Expected output:
```
=== Employee Attrition — Deployment Training ===
📥 Loading processed data from Stage 02 ...
✓ Train: (1176, 53)  |  attrition rate: 16.2%
✓ Test : (294, 53)   |  attrition rate: 15.6%
🚀 Training model ...
✓ F1        : 0.xxx
✓ ROC-AUC   : 0.xxx
✓ Precision : 0.xxx
✓ Recall    : 0.xxx
✓ Log Loss  : 0.xxx
💾 Saved run_id.txt
✅ Training complete. Next: python app.py
```

### Step 3 — Start the API (new terminal)

```bash
cd 04-deployment
python app.py
```

API runs at http://localhost:9696

### Step 4 — Test the API

```bash
cd 04-deployment
pytest -q test_api.py
```

Expect `3 passed`.

Or open http://localhost:9696/docs for the Swagger UI.

---

## API Endpoints

| Endpoint  | Method | Description                    |
|-----------|--------|--------------------------------|
| `/`       | GET    | Welcome message                |
| `/health` | GET    | Service + model status         |
| `/predict`| POST   | Predict attrition for employee |
| `/docs`   | GET    | Swagger UI                     |

---

## Example Request

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "YearsWithCurrManager": 0
  }'
```

## Example Response

```json
{
  "attrition": true,
  "probability": 0.823,
  "risk_level": "High",
  "model_version": "e07d46bac57c4e209f5b44ed5a80f7c0"
}
```

---

## Risk Tiers

| Tier   | Probability Threshold | HR Action                        |
|--------|-----------------------|----------------------------------|
| High   | >= 0.60               | Immediate retention conversation |
| Medium | 0.35 – 0.59           | Monitor and check in             |
| Low    | < 0.35                | No immediate action needed       |

---

## Next Step

→ `05-monitoring/`
