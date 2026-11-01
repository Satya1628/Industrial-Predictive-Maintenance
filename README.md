# Industrial Predictive Maintenance

A practical machine learning project that predicts machine failure risk from operating measurements and uses SHAP to explain model predictions.

## Project features
- Downloads the UCI AI4I 2020 Predictive Maintenance dataset automatically.
- Performs preprocessing: imputation, scaling and one-hot encoding.
- Compares Logistic Regression, Random Forest and XGBoost.
- Evaluates models using precision, recall, F1-score and ROC-AUC.
- Selects the best model based on test-set F1-score.
- Streamlit interface for interactive failure-risk prediction.
- SHAP global feature importance and an individual prediction explanation.

## Dataset
UCI AI4I 2020 Predictive Maintenance Dataset:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset is downloaded automatically on first launch; it is not included in this repository.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install packages: `pip install -r requirements.txt`
4. Run: `streamlit run app.py`

The first launch downloads the dataset and trains three models.

## Deploy on Streamlit Community Cloud
1. Upload this project folder to a GitHub repository.
2. Open https://share.streamlit.io/
3. Create a new app and select your repository and `main` branch.
4. Set main file path to `app.py`.
5. Deploy. No API key is required.

## Inputs and target
Inputs: machine type, air temperature, process temperature, rotational speed, torque and tool wear.  
Target: `Machine failure` (0 = no recorded failure, 1 = recorded failure).

Identifier columns and failure-mode columns are excluded to avoid leakage. In particular, failure-mode labels can reveal information about the target.

## Why these models?
- Logistic Regression: simple baseline classifier.
- Random Forest: tree ensemble that learns non-linear relationships.
- XGBoost: boosted trees that build models sequentially to correct earlier errors.

## SHAP
SHAP (SHapley Additive exPlanations) helps explain predictions:
- Global importance: which features influence model predictions most across examples.
- Individual waterfall plot: how feature contributions move a prediction away from the model's baseline.

## Important limitations
The dataset is a public benchmark, not live factory sensor data. Failure cases are rare, so accuracy alone is misleading. This app is an educational prototype, not a certified industrial safety system.
