# 05 – Monitoring

Extend the deployment with prediction logging and drift detection using Evidently.

---

## What This Stage Adds Over Stage 04

- `app.py` — same as Stage 04 but every `/predict` call is logged to `data/predictions.csv`
- `simulate.py` — sends 200 real employee records from the dataset to the API to build up prediction history
- `monitor.py` — reads the prediction log, splits into reference vs current, and generates an Evidently HTML drift report

---

## Folder Structure

```
05-monitoring/
├── train.py              # Same as 04 — trains and logs model
├── app.py                # FastAPI + prediction logging
├── simulate.py           # Sends bulk requests to build prediction history
├── monitor.py            # Generates Evidently drift report
├── test_api.py           # API tests
├── run_id.txt            # Generated after train.py
├── data/
│   └── predictions.csv   # Generated after simulate.py
├── monitoring_report.html # Generated after monitor.py
└── README.md
```

---

## How to Run

### Step 1 — Start MLflow server (new terminal)
```bash
source .venv/bin/activate
cd 05-monitoring
mlflow server --backend-store-uri sqlite:///backend.db --host 127.0.0.1 --port 5001
```

### Step 2 — Train the model (new terminal)
```bash
cd 05-monitoring
python train.py
```

### Step 3 — Start the API (new terminal)
```bash
cd 05-monitoring
python app.py
```

### Step 4 — Run simulation to generate prediction history (new terminal)
```bash
cd 05-monitoring
python simulate.py
```
This sends 200 employee records to the API and logs predictions to `data/predictions.csv`.

### Step 5 — Generate drift report
```bash
cd 05-monitoring
python monitor.py
```
Opens `monitoring_report.html` — open in your browser.

### Step 6 — Run API tests
```bash
cd 05-monitoring
pytest -q test_api.py
```

---

## What the Drift Report Shows

Evidently splits predictions into two halves — reference (older) vs current (recent) — and checks:

- **Data Drift** — have the input feature distributions shifted?
- **Classification Performance** — has prediction quality changed?

In a real deployment this would run on a schedule (e.g. monthly) and trigger retraining if drift is detected.

---

## Install Evidently

```bash
pip install evidently
```

---

## Next Step

→ `06-cicd/`
