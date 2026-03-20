"""Train Employee Attrition XGBoost model and save artifacts locally.

For CI/CD we save the model and feature columns directly to models/
instead of relying on a running MLflow server — this makes the Docker
image fully self-contained.
"""

from __future__ import annotations

import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, log_loss
)
from xgboost import XGBClassifier
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_DIR     = os.getenv("DATA_DIR", "../data/processed")
MODEL_DIR    = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

PARAMS = {
    "n_estimators"    : 200,
    "max_depth"       : 4,
    "learning_rate"   : 0.05,
    "scale_pos_weight": 5.20,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "random_state"    : RANDOM_STATE,
    "eval_metric"     : "logloss",
    "verbosity"       : 0,
}


def load_data():
    print("📥 Loading processed data ...")
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df  = pd.read_csv(f"{DATA_DIR}/test.csv")

    X_train = train_df.drop(columns=["Attrition"])
    y_train = train_df["Attrition"]
    X_test  = test_df.drop(columns=["Attrition"])
    y_test  = test_df["Attrition"]

    print(f"✓ Train: {X_train.shape}  |  attrition rate: {y_train.mean()*100:.1f}%")
    print(f"✓ Test : {X_test.shape}   |  attrition rate: {y_test.mean()*100:.1f}%")
    return X_train, y_train, X_test, y_test


def train_and_save(X_train, y_train, X_test, y_test):
    print("🚀 Training model ...")

    model = XGBClassifier(**PARAMS)
    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1"       : round(f1_score(y_test, y_pred), 4),
        "roc_auc"  : round(roc_auc_score(y_test, y_pred_prob), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall"   : round(recall_score(y_test, y_pred), 4),
        "log_loss" : round(log_loss(y_test, y_pred_prob), 4),
    }

    for k, v in metrics.items():
        print(f"✓ {k:12s}: {v}")

    # Save artifacts
    joblib.dump(model, MODEL_DIR / "model.joblib")
    joblib.dump(list(X_train.columns), MODEL_DIR / "feature_columns.joblib")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n💾 Saved model to {MODEL_DIR}/")
    return model


def main():
    print("\n=== Employee Attrition — CI/CD Training ===\n")
    X_train, y_train, X_test, y_test = load_data()
    train_and_save(X_train, y_train, X_test, y_test)
    print("\n✅ Training complete.\n")


if __name__ == "__main__":
    main()
