# Employee Attrition Prediction — End-to-End MLOps Pipeline

**IE University | MBD 2025 | Group 6**
Francisco Concha · Aylin Yasgul · Martin Schneider · Bader Al Eisa · Quifeng Cai

---

## Overview

Voluntary employee attrition costs organizations between 50–200% of an employee's annual salary. This project builds an end-to-end MLOps pipeline that predicts each employee's attrition risk using the IBM HR Analytics dataset (1,470 employees, 16.1% attrition rate).

The system outputs a risk score (Low / Medium / High) and probability for each employee, enabling HR teams to intervene before resignations occur.

**Live API:** https://hr-employee-attrition.onrender.com/docs

---

## Pipeline Architecture

```
01 EDA → 02 Feature Engineering → 03 Experiment Tracking → 04 Deployment → 05 Monitoring → 06 CI/CD
```

| Stage | Folder | Description |
|-------|--------|-------------|
| 01 | `01-initial-notebook/` | Exploratory data analysis, fairness audit, naïve baseline |
| 02 | `02-feature-engineering/` | 4 engineered features, train/test split, scaling |
| 03 | `03-experiment-tracking/` | MLflow: 4 models compared, XGBoost selected |
| 04 | `04-deployment/` | FastAPI service, MLflow model loading |
| 05 | `05-monitoring/` | Prediction logging, Evidently drift report |
| 06 | `06-cicd/` | Docker, GitHub Actions CI/CD, Render deployment |

---

## Model

**Algorithm:** XGBoost (selected over Logistic Regression, Random Forest, SVM)
**Key features engineered:**
- `PromotionStagnationRatio` — years since promotion / years at company
- `WorkloadPayPressure` — overtime × monthly income
- `AverageSatisfaction` — mean of 4 satisfaction scores
- `TenureBucket` — 0–2 yrs = 0, 3–7 yrs = 1, 8+ yrs = 2

**Risk tiers:** Low (< 0.35) · Medium (0.35–0.60) · High (≥ 0.60)

---

## Quick Start

### Prerequisites
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r 06-cicd/requirements.txt
```

### Run locally
```bash
cd 06-cicd
python train.py       # train model, save to models/
python app.py         # start API on port 9696
```

### Test the API
```bash
pytest -q 06-cicd/test_api.py
```

### API endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Service status |
| `/predict` | POST | Predict attrition risk |
| `/docs` | GET | Swagger UI |

---

## CI/CD Pipeline

Every push to `main` automatically:
1. Trains the model
2. Lints the code (flake8)
3. Builds a Docker image
4. Runs API tests inside the container
5. Pushes the image to GitHub Container Registry
6. Render pulls the new image and redeploys

**Workflows:** `.github/workflows/ci-cd.yml` · `.github/workflows/train.yml`

---

## Repository Structure

```
.
├── .github/workflows/     # CI/CD workflows
├── 01-initial-notebook/   # EDA
├── 02-feature-engineering/# Feature pipeline
├── 03-experiment-tracking/# MLflow experiments
├── 04-deployment/         # FastAPI + MLflow
├── 05-monitoring/         # Evidently drift monitoring
├── 06-cicd/               # Docker + CI/CD + Render
├── data/
│   └── processed/         # train.csv, test.csv, scaler
├── plots/                 # EDA visualizations
├── render.yaml            # Render deployment manifest
└── README.md
```

---

## Dataset

IBM HR Analytics Employee Attrition dataset — 1,470 employees, 35 features, binary target (Attrition Yes/No).
