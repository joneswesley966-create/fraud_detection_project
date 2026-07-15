"""
Financial Fraud Detection - Model Training
=============================================
Fraud in this dataset occurs ONLY in CASH_OUT and TRANSFER transactions
(0 fraud cases across 8,141 PAYMENT/CASH_IN/DEBIT rows). Training on the
full dataset would let a model "cheat" by just memorizing transaction
type. Instead, the classifier is trained on the CASH_OUT + TRANSFER
subset only, where fraud is genuinely hard to spot (~3% of rows) - this
is the standard, more rigorous approach for this kind of data and is
also what the deployed app relies on for its risk scores.

Trains Logistic Regression, Random Forest and XGBoost, compares them
on Precision / Recall / F1 / ROC-AUC / PR-AUC (accuracy is meaningless
here - predicting "not fraud" for everything already scores >97%), and
saves the best model plus a metrics/feature-importance bundle for the
Streamlit app.
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score,
                              confusion_matrix, roc_curve, precision_recall_curve)
from xgboost import XGBClassifier

df = pd.read_csv("data/fraud_data_clean.csv")

# ---------------------------------------------------------------
# Restrict training to the high-risk transaction types
# ---------------------------------------------------------------
model_df = df[df["isHighRiskType"] == 1].copy()
print(f"Training subset (CASH_OUT + TRANSFER): {len(model_df)} rows, "
      f"{model_df['IsFraud'].sum()} fraud ({model_df['IsFraud'].mean()*100:.2f}%)")

FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest", "origBalanceWiped", "destBalanceUnchanged",
    "amountToBalanceRatio", "unusuallogin", "nameDestIsMerchant", "DayOfWeek",
]
model_df["type_TRANSFER"] = (model_df["type"] == "TRANSFER").astype(int)
FEATURES.append("type_TRANSFER")

X = model_df[FEATURES]
y = model_df["IsFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
models = {}

# --- Logistic Regression (baseline, interpretable) ---
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
models["Logistic Regression"] = lr

# --- Random Forest ---
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
models["Random Forest"] = rf

# --- XGBoost ---
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)
models["XGBoost"] = xgb

for name, model in models.items():
    if name == "Logistic Regression":
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
    }
    print(f"\n{name}: {results[name]}")

# ---------------------------------------------------------------
# Pick the best model by PR-AUC (the right metric for rare-event
# fraud detection - ROC-AUC can look deceptively good under heavy
# class imbalance)
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["pr_auc"])
best_model = models[best_name]
print(f"\nBest model: {best_name}")

if best_name == "Logistic Regression":
    y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_best = best_model.predict(X_test_scaled)
    importances = dict(zip(FEATURES, np.abs(best_model.coef_[0])))
else:
    y_prob_best = best_model.predict_proba(X_test)[:, 1]
    y_pred_best = best_model.predict(X_test)
    importances = dict(zip(FEATURES, best_model.feature_importances_))

cm = confusion_matrix(y_test, y_pred_best).tolist()
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob_best)

bundle = {
    "best_model_name": best_name,
    "features": FEATURES,
    "results": results,
    "confusion_matrix": cm,
    "test_size": len(y_test),
    "test_fraud_count": int(y_test.sum()),
    "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    "pr_curve": {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()},
    "feature_importance": {k: round(float(v), 4) for k, v in
                            sorted(importances.items(), key=lambda x: -x[1])},
}

with open("models/metrics.json", "w") as f:
    json.dump(bundle, f, indent=2)

joblib.dump(best_model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(FEATURES, "models/features.pkl")
with open("models/best_model_name.txt", "w") as f:
    f.write(best_name)

print("\nSaved model, scaler, and metrics bundle to models/")
