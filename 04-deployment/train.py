"""Train Employee Attrition XGBoost model and log to MLflow.

Mirrors the experiment from Stage 03 (run: e07d46bac57c4e209f5b44ed5a80f7c0):
1. Load processed train/test data from Stage 02
2. Build feature engineering + XGBoost pipeline
3. Train with class weighting for imbalance
4. Log params, metrics (F1, ROC-AUC, Precision, Recall, Log Loss) and model artifact
5. Write run_id.txt for app.py
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, log_loss
)
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost
import mlflow.sklearn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
EXPERIMENT_NAME     = "employee-attrition"
RANDOM_STATE        = 42

# XGBoost params from Stage 03 experiment
# Run ID: e07d46bac57c4e209f5b44ed5a80f7c0
PARAMS = {
    "n_estimators"    : 200,
    "max_depth"       : 4,
    "learning_rate"   : 0.05,
    "scale_pos_weight": 5.20,   # 1233 / 237 — handles class imbalance
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "random_state"    : RANDOM_STATE,
    "eval_metric"     : "logloss",
    "verbosity"       : 0,
}

DATA_DIR = os.getenv("DATA_DIR", "../data/processed")


def load_data():
    """Load processed train/test splits from Stage 02."""
    print("📥 Loading processed data from Stage 02 ...")
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")

    X_train = train_df.drop(columns=["Attrition"])
    y_train = train_df["Attrition"]
    X_test  = test_df.drop(columns=["Attrition"])
    y_test  = test_df["Attrition"]

    print(f"✓ Train: {X_train.shape}  |  attrition rate: {y_train.mean()*100:.1f}%")
    print(f"✓ Test : {X_test.shape}   |  attrition rate: {y_test.mean()*100:.1f}%")
    return X_train, y_train, X_test, y_test


def train_and_log(X_train, y_train, X_test, y_test):
    """Train XGBoost model, evaluate, and log everything to MLflow."""
    print("🚀 Training model ...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = XGBClassifier(**PARAMS)

    with mlflow.start_run(run_name="xgboost-deployment") as run:
        # Train
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        logloss   = log_loss(y_test, y_pred_prob)

        print(f"✓ F1        : {f1:.3f}   (target >= 0.75)")
        print(f"✓ ROC-AUC   : {roc_auc:.3f}   (target >= 0.80)")
        print(f"✓ Precision : {precision:.3f}")
        print(f"✓ Recall    : {recall:.3f}")
        print(f"✓ Log Loss  : {logloss:.3f}")

        # Log params and metrics
        mlflow.log_params(PARAMS)
        mlflow.log_param("train_rows",    len(y_train))
        mlflow.log_param("test_rows",     len(y_test))
        mlflow.log_param("n_features",    X_train.shape[1])
        mlflow.log_metric("f1",           f1)
        mlflow.log_metric("roc_auc",      roc_auc)
        mlflow.log_metric("precision",    precision)
        mlflow.log_metric("recall",       recall)
        mlflow.log_metric("log_loss",     logloss)
        mlflow.set_tag("model_type",      "XGBoost")
        mlflow.set_tag("stage",           "deployment")

        # Log model artifact as sklearn flavor so predict_proba works in app.py
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.iloc[:1]
        )

        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print(f"💾 Saved run_id.txt (run: {run_id})")
        print(f"🖥  View MLflow UI: {MLFLOW_TRACKING_URI}")
        return run_id, model


def main():
    print("\n=== Employee Attrition — Deployment Training ===\n")
    X_train, y_train, X_test, y_test = load_data()
    run_id, model = train_and_log(X_train, y_train, X_test, y_test)
    print("\n✅ Training complete. Next: python app.py\n")
    return run_id


if __name__ == "__main__":
    main()
