"""
Telco Customer Churn Prediction - Final Project COMP6140001 Data Mining
BINUS University

Full KDD pipeline: load data, clean, transform, train 6 classification
models, evaluate, and save figures/tables for the report.

Run:  python src/churn_analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Telco-Customer-Churn.csv")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# STEP 1 - load data
print("=" * 70)
print("STEP 1  Loading data")
print("=" * 70)
df = pd.read_csv(DATA_PATH)
print(f"Raw shape          : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Target distribution:\n{df['Churn'].value_counts()}")


# STEP 2 - cleaning
print("\n" + "=" * 70)
print("STEP 2  Cleaning")
print("=" * 70)

# TotalCharges has 11 blank rows, all new customers with 0 tenure
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = df["TotalCharges"].isna().sum()
print(f"Blank TotalCharges rows removed : {n_missing}")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

df = df.drop(columns=["customerID"])  # just an ID, no predictive value
df["Churn"] = (df["Churn"] == "Yes").astype(int)

churn_rate = df["Churn"].mean()
print(f"Clean shape                     : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate (positive class)     : {churn_rate:.1%}  <-- imbalanced")

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in df.columns
                        if c not in numeric_features + ["Churn"]]

X = df.drop(columns=["Churn"])
y = df["Churn"]


# STEP 3 - EDA
print("\n" + "=" * 70)
print("STEP 3  EDA figures")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) target balance
sns.countplot(x="Churn", data=df, ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Churn Distribution (0 = Stay, 1 = Churn)")
axes[0, 0].set_xticklabels(["Stay", "Churn"])

# (b) churn rate by contract type
contract_churn = df.groupby("Contract")["Churn"].mean().sort_values()
sns.barplot(x=contract_churn.index, y=contract_churn.values,
            ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Churn Rate by Contract Type")
axes[0, 1].set_ylabel("Churn rate")

# (c) tenure distribution split by churn
sns.histplot(data=df, x="tenure", hue="Churn", bins=30,
             ax=axes[1, 0], palette="Set1", element="step")
axes[1, 0].set_title("Tenure Distribution by Churn")

# (d) monthly charges split by churn
sns.boxplot(x="Churn", y="MonthlyCharges", data=df,
            ax=axes[1, 1], palette="Set2")
axes[1, 1].set_title("Monthly Charges by Churn")
axes[1, 1].set_xticklabels(["Stay", "Churn"])

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "eda_overview.png"), dpi=150)
plt.close()
print("Saved figures/eda_overview.png")


# STEP 4 - preprocessing pipeline + split
print("\n" + "=" * 70)
print("STEP 4  Transformation & split")
print("=" * 70)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
     categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print("SMOTE applied inside the pipeline (train folds only)")


# STEP 5 - train 6 models
print("\n" + "=" * 70)
print("STEP 5  Training 6 models")
print("=" * 70)

# 5 classic models from the syllabus + XGBoost for the trends/LO3 angle
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "k-NN":                KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=12,
                                                  random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                         subsample=0.9, colsample_bytree=0.9,
                                         eval_metric="logloss", random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
roc_data = {}
fitted_pipelines = {}

for name, clf in models.items():
    pipe = ImbPipeline(steps=[
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", clf),
    ])

    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)

    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    fitted_pipelines[name] = pipe
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr)

    results.append({
        "Model":     name,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
        "CV ROC-AUC (mean)": cv_auc.mean(),
    })
    print(f"  {name:<20} F1={results[-1]['F1']:.3f}  ROC-AUC={results[-1]['ROC-AUC']:.3f}")

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


# STEP 6 - evaluation figures
print("\n" + "=" * 70)
print("STEP 6  Evaluation")
print("=" * 70)

# (a) comparison table
results_df.to_csv(os.path.join(RESULT_DIR, "model_comparison.csv"), index=False)
print("\nModel comparison (sorted by ROC-AUC):")
print(results_df.round(3).to_string(index=False))

# (b) F1 vs ROC-AUC bar chart
plot_df = results_df.melt(id_vars="Model", value_vars=["F1", "ROC-AUC"],
                          var_name="Metric", value_name="Score")
plt.figure(figsize=(11, 6))
sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="Set2")
plt.title("Model Comparison  -  F1 vs ROC-AUC (test set)")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=150)
plt.close()
print("Saved figures/model_comparison.png")

# (c) ROC curves overlay
plt.figure(figsize=(8, 7))
for name, (fpr, tpr) in roc_data.items():
    auc = results_df.loc[results_df["Model"] == name, "ROC-AUC"].values[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - All Models")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "roc_curves.png"), dpi=150)
plt.close()
print("Saved figures/roc_curves.png")

# (d) confusion matrix for the best model
best_name = results_df.iloc[0]["Model"]
best_pipe = fitted_pipelines[best_name]
cm = confusion_matrix(y_test, best_pipe.predict(X_test))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
plt.title(f"Confusion Matrix - {best_name} (best by ROC-AUC)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_best.png"), dpi=150)
plt.close()
print(f"Saved figures/confusion_matrix_best.png  (best model: {best_name})")

# (e) feature importance from Random Forest
rf_pipe = fitted_pipelines["Random Forest"]
feat_names = rf_pipe.named_steps["prep"].get_feature_names_out()
importances = rf_pipe.named_steps["model"].feature_importances_
imp_df = (pd.DataFrame({"feature": feat_names, "importance": importances})
          .sort_values("importance", ascending=False).head(15))
imp_df.to_csv(os.path.join(RESULT_DIR, "feature_importance_top.csv"), index=False)
print("Saved results/feature_importance_top.csv")
plt.figure(figsize=(9, 7))
sns.barplot(data=imp_df, x="importance", y="feature", palette="viridis")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"), dpi=150)
plt.close()
print("Saved figures/feature_importance.png")

# (f) classification report, text version for the appendix
report_txt = classification_report(y_test, best_pipe.predict(X_test),
                                   target_names=["Stay", "Churn"])
with open(os.path.join(RESULT_DIR, "best_model_classification_report.txt"), "w") as f:
    f.write(f"Best model: {best_name}\n\n{report_txt}\n")
print("\nBest model classification report:\n")
print(report_txt)

print("=" * 70)
print("DONE. See figures/ and results/ for all outputs.")
print("=" * 70)
