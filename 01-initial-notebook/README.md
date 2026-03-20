# 01 – Exploratory Data Analysis

Understand the IBM HR dataset before touching any model. Every preprocessing and modelling decision in later stages is justified by findings here.

---

## What This Stage Does

Loads the raw IBM HR dataset (1,470 employees, 35 columns) and produces visual and statistical insights across eight sections:

1. Data quality check — confirms no missing values, no duplicates, identifies 4 useless columns
2. Target variable analysis — visualises the 16.1% attrition rate and class imbalance
3. Naïve baseline — establishes the floor every model must beat (F1=0.00 at 83.9% accuracy)
4. Categorical feature analysis — attrition rate by OverTime, BusinessTravel, Department, JobRole, MaritalStatus, EducationField
5. Numeric feature analysis — distribution comparisons between leavers and stayers
6. Tenure stage analysis — validates the `TenureBucket` engineered feature
7. Correlation analysis — ranks all features by Pearson correlation with attrition
8. Fairness audit — attrition rates by Age, Gender, and MaritalStatus (protected attributes)

---

## Folder Structure

```
01-notebook/
├── attrition_eda.ipynb    # Full EDA notebook
└── README.md              # This file

data/
└── WA_Fn-UseC_HR-Employee-Attrition.csv    # Raw IBM HR dataset (place here before running)
```

---

## How to Run

```bash
source .venv/bin/activate
cd 01-notebook
jupyter notebook attrition_eda.ipynb
```

Run all cells top-to-bottom. Plots are saved automatically to `data/plots/`.

---

## Key Findings

| Finding | Detail |
|---------|--------|
| No data quality issues | Zero missing values, zero duplicates |
| Severe class imbalance | 83.9% No / 16.1% Yes — F1 is the right metric, not accuracy |
| Strongest predictor | OverTime: 30.5% attrition vs 10.4% — 3x risk multiplier |
| New-hire vulnerability | 0–2 yr employees leave at 3x the rate of veterans |
| Income gap | Leavers earn $4,787/month vs $6,833 for stayers |
| Satisfaction matters | All 4 satisfaction scores inversely correlated with leaving |
| Fairness concern | Single employees (25.5%) and 18-25 age group (36.5%) show elevated rates |

---

## Outputs

```
data/plots/
├── 01_attrition_distribution.png
├── 02_categorical_attrition.png
├── 03_numeric_distributions.png
├── 04_tenure_attrition.png
├── 05_correlation_target.png
└── 06_fairness_audit.png
```

---

## Next Step

→ `02-features/attrition_fe.ipynb`
