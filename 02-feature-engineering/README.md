# 02 – Feature Engineering

Transform the raw IBM HR dataset into a clean, model-ready format. This stage produces the processed files that every training script in Stage 03 and beyond reads from.

---

## What This Stage Does

Applies a reproducible preprocessing pipeline in the following order:

1. **Drop useless columns** — `EmployeeCount`, `StandardHours`, `Over18` (constant), `EmployeeNumber` (ID)
2. **Encode target** — `Attrition`: Yes → 1, No → 0
3. **Engineer 4 domain features** — built before encoding/scaling to avoid leakage
4. **Encode categoricals** — binary for `OverTime`/`Gender`; one-hot for `BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`
5. **Train/test split** — 80/20, stratified to preserve 16.1% attrition rate in both splits
6. **Scale numerics** — `StandardScaler` fitted on training data only
7. **Save outputs** — two CSVs + feature list + fitted scaler

---

## Engineered Features

| Feature | Formula | Captures |
|---------|---------|---------|
| `PromotionStagnationRatio` | `YearsSinceLastPromotion / max(1, YearsAtCompany)` | Career stagnation relative to tenure |
| `WorkloadPayPressure` | `OverTime × MonthlyIncome` | Overwork-income interaction |
| `AverageSatisfaction` | Mean of 4 satisfaction scores | Composite wellbeing index |
| `TenureBucket` | 0–2=0, 3–7=1, 8+=2 | Non-linear new-hire vulnerability |

All 4 will be validated via ablation test in Stage 03 (raw features vs raw + engineered).

---

## Class Imbalance Strategy

Per the proposal, we do **not** apply SMOTE in this stage. The sequence is:

1. `class_weight="balanced"` in all models (Stage 03) — adjusts loss function
2. Threshold tuning on precision-recall curve (Stage 03) — adjusts operating point
3. SMOTE inside CV folds only (Stage 03) — only if steps 1 and 2 are insufficient

---

## Folder Structure

```
02-features/
├── attrition_fe.ipynb    # Feature engineering notebook
└── README.md             # This file
```

---

## How to Run

```bash
source .venv/bin/activate
pip install scikit-learn joblib pandas numpy matplotlib seaborn

cd 02-features
jupyter notebook attrition_fe.ipynb
```

Run all cells. Outputs are written to `data/processed/`.

---

## Design Decisions

| Decision | Reason |
|----------|--------|
| StandardScaler over MinMax | More robust to outliers in income/rate columns |
| One-hot for BusinessTravel | Proposal specifies one-hot for all multi-class categoricals |
| 80/20 split (no val set) | 5-fold CV in Stage 03 handles model selection within train |
| Scaler saved as `.joblib` | `app.py` must apply identical scaling at inference time |
| Stratified split | Ensures both splits preserve the original 16.1% attrition rate |

---

## Outputs

```
data/processed/
├── train.csv              # 1,176 rows — used for all model training in Stage 03
├── test.csv               # 294 rows  — held out until final evaluation
├── feature_columns.txt    # Ordered feature names (guarantees column order at inference)
├── scaler.joblib          # Fitted StandardScaler (used in app.py)
└── cols_to_scale.joblib   # List of column names that were scaled
```

---

## Next Step

→ `03-experiments/attrition_experiments.ipynb`
