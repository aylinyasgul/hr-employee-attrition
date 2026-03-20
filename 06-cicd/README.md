# 06 – CI/CD

Automated training, testing, and deployment using GitHub Actions and Render.

---

## What This Stage Does

Every push to `main` triggers:
1. **Train** — runs `train.py`, saves model to `models/`
2. **Lint** — flake8 checks `app.py` and `train.py`
3. **Build** — builds Docker image with model baked in
4. **Test** — starts container, runs `pytest test_api.py` (3 tests)
5. **Push** — pushes image to GitHub Container Registry (GHCR) with `:latest` and commit SHA tags
6. **Deploy** — Render pulls the new image (manual redeploy trigger)

---

## Key Difference from Stages 04/05

Stages 04 and 05 relied on a running MLflow server to load the model at startup.
Stage 06 is **self-contained** — `train.py` saves the model as a `.joblib` file,
and `app.py` loads it directly. No MLflow server needed inside Docker.

---

## Folder Structure

```
06-cicd/
├── train.py              # Trains model, saves to models/
├── app.py                # FastAPI — loads model from models/
├── test_api.py           # 3 API tests
├── requirements.txt      # Pinned dependencies
├── Dockerfile            # Builds self-contained image
├── train.yml             # Reusable training workflow
├── ci-cd.yml             # Main CI/CD orchestrator
└── README.md
```

---

## Setup Instructions

### 1. Create GitHub Repository

```bash
cd [project root]
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### 2. Add GitHub Actions Workflows

Copy the workflow files to the correct location:

```bash
mkdir -p .github/workflows
cp 06-cicd/train.yml  .github/workflows/train.yml
cp 06-cicd/ci-cd.yml  .github/workflows/ci-cd.yml
git add .github/
git commit -m "Add CI/CD workflows"
git push
```

### 3. Add Processed Data to Repo

The CI pipeline needs `data/processed/train.csv` and `data/processed/test.csv`.
Add them to the repo (they are small enough):

```bash
git add data/processed/
git commit -m "Add processed data"
git push
```

### 4. Watch the Pipeline Run

Go to your GitHub repo → **Actions** tab.
You should see the CI/CD Pipeline running automatically.

### 5. Deploy on Render

1. Go to [render.com](https://render.com) and sign up / log in
2. Click **New → Web Service**
3. Choose **Deploy an existing image from a registry**
4. Image URL: `ghcr.io/<your-username>/<repo-name>:latest`
5. Set **Port** to `9696`
6. Click **Deploy**

After the first CI run pushes the image to GHCR, Render will pull and deploy it.
For subsequent updates, click **Manual Deploy** on Render after each CI run.

---

## Local Testing

Train and test locally before pushing:

```bash
cd 06-cicd
python train.py          # creates models/
python app.py            # starts API on port 9696
pytest -q test_api.py    # run tests
```

Build and test Docker locally:

```bash
cd 06-cicd
docker build -t attrition-app .
docker run -p 9696:9696 attrition-app
```

---

## API Endpoints

| Endpoint   | Method | Description              |
|------------|--------|--------------------------|
| `/`        | GET    | Welcome message          |
| `/health`  | GET    | Service status           |
| `/predict` | POST   | Predict attrition        |
| `/docs`    | GET    | Swagger UI               |
