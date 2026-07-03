"""
Generates a synthetic loan-application dataset that mirrors the structure,
column names, and general statistical patterns of the well-known
'Loan Prediction Practice Problem' dataset (Analytics Vidhya / Kaggle).

Why synthetic? The original dataset lives behind a Kaggle login wall.
This generator produces realistic, internally-consistent data (614 rows,
same 13 columns, same missing-value patterns, same rough approval logic)
so the project can be built and run immediately.

To use the REAL dataset instead: download 'train.csv' from Kaggle
(search "Loan Prediction Problem Dataset") and drop it into this data/
folder as loan_data.csv, keeping the same column names. No other code
changes needed.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 614

genders = np.random.choice(["Male", "Female"], size=N, p=[0.81, 0.19])
married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], size=N, p=[0.58, 0.17, 0.17, 0.08])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.38, 0.38, 0.24])

applicant_income = np.round(np.random.lognormal(mean=8.35, sigma=0.55, size=N)).astype(int)
coapplicant_income = np.round(
    np.where(married == "Yes", np.random.lognormal(mean=7.3, sigma=0.9, size=N), 0)
).astype(int)
coapplicant_income = np.where(np.random.rand(N) < 0.15, 0, coapplicant_income)

total_income = applicant_income + coapplicant_income
loan_amount = np.round(total_income * np.random.uniform(0.08, 0.22, size=N) / 10).astype(float)

loan_amount_term = np.random.choice(
    [360, 180, 120, 300, 240, 60, 84, 36, 12],
    size=N,
    p=[0.71, 0.09, 0.05, 0.04, 0.04, 0.02, 0.02, 0.02, 0.01],
)

credit_history = np.random.choice([1.0, 0.0], size=N, p=[0.84, 0.16])

# Approval probability driven mainly by credit history, income-to-loan ratio,
# education, and property area -- mirrors real-world lending signal strength.
income_to_loan = total_income / (loan_amount * 1000 + 1)
score = (
    2.6 * credit_history
    + 0.35 * (income_to_loan > income_to_loan.mean()).astype(int)
    + 0.15 * (education == "Graduate").astype(int)
    + 0.20 * (property_area == "Semiurban").astype(int)
    - 0.10 * (property_area == "Rural").astype(int)
    + np.random.normal(0, 0.55, size=N)
)
approve_prob = 1 / (1 + np.exp(-(score - 1.2)))
loan_status = np.where(np.random.rand(N) < approve_prob, "Y", "N")

loan_id = [f"LP{str(100000 + i)[-6:]}" for i in range(1, N + 1)]

df = pd.DataFrame({
    "Loan_ID": loan_id,
    "Gender": genders,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

# Inject realistic missingness (mirrors the original dataset's NaN pattern)
for col, frac in [
    ("Gender", 0.021), ("Married", 0.008), ("Dependents", 0.024),
    ("Self_Employed", 0.052), ("LoanAmount", 0.036),
    ("Loan_Amount_Term", 0.023), ("Credit_History", 0.081),
]:
    idx = np.random.choice(df.index, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("/home/claude/loan_approval_predictor/data/loan_data.csv", index=False)
print(f"Generated {len(df)} rows -> data/loan_data.csv")
print(df["Loan_Status"].value_counts(normalize=True))
