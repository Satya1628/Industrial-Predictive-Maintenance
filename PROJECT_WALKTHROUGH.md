# Project walkthrough — concepts to understand

## 1. Problem
Predict whether a machine failure is recorded from its operating measurements. This is binary classification.

## 2. Features
Machine type, air temperature, process temperature, rotational speed, torque and tool wear.

## 3. Preprocessing
- Median imputation fills missing numeric values.
- Most-frequent imputation fills missing machine types.
- One-hot encoding converts machine type categories into numeric columns.
- StandardScaler puts numeric inputs on a comparable scale. It is especially useful for Logistic Regression.

## 4. Train/test split
A stratified split keeps approximately the same rare-failure proportion in train and test sets.

## 5. Models
- Logistic Regression: a simple baseline.
- Random Forest: combines decision trees.
- XGBoost: builds boosted trees sequentially.
The project compares their F1-scores and selects the best one on the held-out test set.

## 6. Metrics
- Precision: of the cases predicted as failures, how many were failures?
- Recall: of all actual failures, how many did the model detect?
- F1-score: balances precision and recall.
- ROC-AUC: measures how well the model ranks failure cases above non-failure cases.

## 7. SHAP explainability
SHAP assigns each feature a contribution to a prediction.
- Global SHAP bar chart: average magnitude of feature contributions across test cases.
- SHAP waterfall plot: explains one example, showing which features push the prediction higher or lower relative to the baseline.

## 8. Data leakage
The failure-mode columns (TWF, HDF, PWF, OSF, RNF) are not used because they describe failure types and can reveal the target. Identifier columns are also excluded.

## 9. Limitations
This is benchmark data, not live factory data. Predictions should not be used as real maintenance or safety decisions without validation on real industrial data.
