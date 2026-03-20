"""Generate an Evidently drift report from logged predictions.

Splits predictions into reference (older half) vs current (recent half),
then generates a data drift report to detect distribution shifts.

Usage:
    python monitor.py
"""

import pandas as pd
from pathlib import Path
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_PATH    = Path("data/predictions.csv")
REPORT_PATH = Path("monitoring_report.html")


def main():
    print("\n📊 Starting monitoring report ...\n")

    if not LOG_PATH.exists():
        raise FileNotFoundError(
            "❌ No predictions found. Run simulate.py first!"
        )

    df = pd.read_csv(LOG_PATH, parse_dates=["ts"])
    df = df[df["attrition"] != "attrition"]  # remove duplicate header rows
    f = df.dropna()
    df["attrition"] = df["attrition"].astype(int)
    df["probability"] = df["probability"].astype(float)
    print(f"✓ Loaded {len(df)} logged predictions")

    if len(df) < 20:
        raise ValueError(
            f"❌ Only {len(df)} predictions found — need at least 20. "
            "Run simulate.py to generate more."
        )

    # Split into reference (older) vs current (recent)
    df = df.sort_values("ts")
    midpoint  = len(df) // 2
    reference = df.iloc[:midpoint].copy()
    current   = df.iloc[midpoint:].copy()

    print(f"Reference period : {len(reference)} predictions")
    print(f"Current period   : {len(current)} predictions")

    # Column mapping for Evidently
    column_mapping = ColumnMapping(
        target          = "attrition",
        prediction      = "probability",
        numerical_features = [
            "probability", "MonthlyIncome", "JobSatisfaction",
            "AverageSatisfaction", "TenureBucket",
            "PromotionStagnationRatio"
        ],
        categorical_features = ["OverTime", "risk_level"],
    )

    # Build report — data drift + classification performance
    print("\n🧮 Generating Evidently drift report ...")
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset(),
    ])

    report.run(
        reference_data = reference,
        current_data   = current,
        column_mapping = column_mapping,
    )

    report.save_html(str(REPORT_PATH))
    print(f"✅ Report saved: {REPORT_PATH.resolve()}")
    print("Open monitoring_report.html in your browser to explore drift metrics.")


if __name__ == "__main__":
    main()
