# Telco Customer Churn Prediction

Final Project, COMP6140001 Data Mining, BINUS University.

**Group:** Lionel Benedict Vincentino, Adrian, Keinen, Enrico

## 1. Overview

This project applies the full Knowledge Discovery in Databases (KDD) process to
predict customer churn for a telecommunications company. Using the IBM Telco
Customer Churn dataset, we build and compare six classification models to
identify which customers are likely to stop using the service, so the company
can target retention efforts before the customer leaves.

Churn prediction is a binary classification task: for each customer, predict
whether they will churn (Yes) or stay (No).

## 2. Dataset

- Name: Telco Customer Churn (IBM Sample Data Sets)
- Records: 7,043 customers, 21 columns (20 features plus the target `Churn`)
- Target: `Churn` (Yes or No), about 26.6 percent positive (imbalanced)
- Source: Kaggle, https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- The dataset is adapted from the IBM sample data sets and is intended for
  educational and research use.

A copy of the CSV is included in `data/` so the dataset link points to this
repository (publicly accessible, no login).

## 3. Project structure

```
telco-churn-datamining/
  data/
    Telco-Customer-Churn.csv        raw dataset
  src/
    churn_analysis.py               full KDD pipeline (load to evaluation)
  figures/                          all generated charts (PNG)
    eda_overview.png
    model_comparison.png
    roc_curves.png
    confusion_matrix_best.png
    feature_importance.png
  results/
    model_comparison.csv            metrics for every model
    best_model_classification_report.txt
    feature_importance_top.csv
  slides/
    Telco_Churn_Presentation.pptx   presentation deck
  requirements.txt
  README.md
```

## 4. How to run

```bash
pip install -r requirements.txt
python src/churn_analysis.py
```

All figures are written to `figures/` and all result tables to `results/`.

## 5. Method summary

1. Data selection: load the raw CSV.
2. Preprocessing: convert `TotalCharges` to numeric (11 blank rows removed),
   drop `customerID`, encode the target.
3. Transformation: one-hot encode categorical features, standardize numeric
   features, stratified 80/20 train and test split, and SMOTE on the training
   folds only to handle class imbalance.
4. Data mining: train six models, five from the syllabus (Logistic Regression,
   Decision Tree, k-NN, Naive Bayes, Random Forest) plus XGBoost as a modern
   method for the data mining trends learning outcome.
5. Evaluation: accuracy, precision, recall, F1, and ROC-AUC on the test set,
   plus 5-fold cross-validated ROC-AUC for robustness.
6. Interpretation: model comparison table, ROC curves, confusion matrix, and
   feature importance.

## 6. Results snapshot

Best model by ROC-AUC is Logistic Regression (ROC-AUC 0.833, F1 0.607),
closely followed by XGBoost (ROC-AUC 0.829, highest accuracy at 0.786). The top
churn drivers are month-to-month contracts, short tenure, no online security or
tech support, and electronic-check payment. Full numbers are in
`results/model_comparison.csv`.

## 7. Mapping to the report chapters

- Chapter 1 Introduction: Section 1 of this README plus the business problem.
- Chapter 2 Theoretical Basis: the six models and the evaluation metrics.
- Chapter 3 Methods: Section 5 above, the preprocessing and pipeline.
- Chapter 4 Implementation and Evaluation: `results/` and `figures/`.
- Chapter 5 Conclusion and Suggestions: the findings and retention recommendations.
- Chapter 6 Appendix: public links to slides, this repository, and the dataset.
