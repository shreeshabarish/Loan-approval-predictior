CITS4700

# Loan Approval Prediction

A binary classification project that predicts whether a bank loan application will be
**approved** or **rejected**, using Logistic Regression on applicant demographics, income,
and credit history.

## Problem Statement

Lenders need to quickly assess whether an applicant is likely to be eligible for a loan based
on details submitted in an application form: gender, marital status, education, income,
co-applicant income, loan amount, loan term, credit history, and property area. This project
builds an interpretable model to automate that first-pass decision.

## Dataset

- **File:** `data/loan_data.csv`
- **Rows / Columns:** 614 rows × 13 columns
- **Schema:** matches the well-known Analytics Vidhya / Kaggle *Loan Prediction Practice
  Problem* dataset (`Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`,
  `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`,
  `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status`)

**Note on data source:** the original dataset requires a Kaggle login to download. To keep this
project self-contained and immediately runnable, `data/generate_data.py` produces a synthetic
dataset with the identical column structure, realistic missing-value rates, and an approval
probability driven mainly by credit history and income — mirroring the real dataset's known
patterns. To use the real data instead, download `train.csv` from Kaggle ("Loan Prediction
Problem Dataset") and save it as `data/loan_data.csv` — no code changes required.

| Column | Description |
|---|---|
| Gender | Male / Female |
| Married | Applicant married (Y/N) |
| Dependents | Number of dependents (0, 1, 2, 3+) |
| Education | Graduate / Not Graduate |
| Self_Employed | Self-employed (Y/N) |
| ApplicantIncome | Applicant's monthly income |
| CoapplicantIncome | Co-applicant's monthly income |
| LoanAmount | Loan amount requested (in thousands) |
| Loan_Amount_Term | Loan term in months |
| Credit_History | Meets credit guidelines (1) or not (0) |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Target: Y (approved) / N (rejected) |

## Approach

1. **EDA** — distribution of target, approval rate by credit history / education / property
   area, income and loan amount distributions, correlation heatmap.
2. **Preprocessing** — mode/median imputation for missing values, `Dependents` cast to
   numeric, log transforms for skewed income/loan features, one-hot encoding of categoricals.
3. **Modeling** — `LogisticRegression` (scikit-learn) trained on an 80/20 stratified split,
   with `StandardScaler` applied to all features.
4. **Evaluation** — accuracy, precision/recall/F1, confusion matrix, ROC curve, ROC-AUC.
5. **Interpretability** — model coefficients plotted to show which features push toward
   approval vs. rejection.

## Results

| Metric | Score |
|---|---|
| Accuracy | ~0.84 |
| ROC-AUC | ~0.77 |

`Credit_History` is by far the strongest predictor of approval, consistent with real-world
lending behavior, followed by total household income and property area.

*(Exact numbers vary slightly by random seed / regenerated data — see notebook output.)*

## Project Structure

```
loan_approval_predictor/
├── data/
│   ├── generate_data.py        # builds the synthetic dataset
│   └── loan_data.csv           # the dataset used for training
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── predictions.csv         # test-set predictions with probabilities
├── loan_approval_predictor.ipynb   # full walkthrough notebook (EDA + modeling)
├── loan_approval_predictor.py      # standalone script, same pipeline
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

# (optional) regenerate the dataset
python data/generate_data.py

# run the full pipeline
python loan_approval_predictor.py
```

Or open `loan_approval_predictor.ipynb` in Jupyter and run all cells in order.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (LogisticRegression, StandardScaler, train_test_split, metrics)
- matplotlib, seaborn

## Possible Extensions

- Compare against Decision Tree / Random Forest for an accuracy vs. interpretability trade-off
- Hyperparameter tuning via `GridSearchCV` (`C`, `penalty`, `solver`)
- Handle class imbalance with `class_weight="balanced"` or SMOTE
- Deploy as a simple Streamlit app for interactive predictions
