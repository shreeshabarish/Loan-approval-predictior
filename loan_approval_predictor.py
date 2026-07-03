"""
Loan Approval Prediction using Logistic Regression
====================================================
Predicts whether a loan application will be approved (Y/N) based on
applicant demographics, income, and credit history.

Dataset: data/loan_data.csv (614 rows, 13 columns)
Model:   Logistic Regression (scikit-learn)

Run: python loan_approval_predictor.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/loan_data.csv")
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print(df.isnull().sum()[df.isnull().sum() > 0])

# ---------------------------------------------------------------------------
# 2. Clean & preprocess
# ---------------------------------------------------------------------------
df = df.drop(columns=["Loan_ID"])

for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "Loan_Amount_Term"]:
    df[col] = df[col].fillna(df[col].mode()[0])
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)

# Feature engineering: total household income and EMI-style ratio
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["LoanAmount_Log"] = np.log1p(df["LoanAmount"])
df["TotalIncome_Log"] = np.log1p(df["TotalIncome"])

cat_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

df_encoded["Loan_Status"] = df_encoded["Loan_Status"].map({"Y": 1, "N": 0})

# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------
X = df_encoded.drop(columns=["Loan_Status"])
y = df_encoded["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Train Logistic Regression
# ---------------------------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"\nAccuracy: {acc:.4f}")
print(f"ROC-AUC:  {auc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred), display_labels=["Rejected", "Approved"]
).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# ROC curve
fig, ax = plt.subplots(figsize=(5, 4))
RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
ax.set_title("ROC Curve")
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150)
plt.close()

# Feature importance (logistic regression coefficients)
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 6))
coefs.plot(kind="barh", ax=ax, color=np.where(coefs > 0, "#2a9d8f", "#e76f51"))
ax.set_title("Feature Impact on Approval (Logistic Regression Coefficients)")
ax.set_xlabel("Coefficient (standardized)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. Export predictions
# ---------------------------------------------------------------------------
results = X_test.copy()
results["Actual"] = y_test.map({1: "Y", 0: "N"}).values
results["Predicted"] = pd.Series(y_pred, index=X_test.index).map({1: "Y", 0: "N"})
results["Approval_Probability"] = np.round(y_proba, 3)
results.to_csv("outputs/predictions.csv", index=False)

print("\nSaved: outputs/confusion_matrix.png, outputs/roc_curve.png,")
print("       outputs/feature_importance.png, outputs/predictions.csv")
