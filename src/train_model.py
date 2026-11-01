from pathlib import Path
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "machine_failure_model.joblib"
DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

FEATURE_COLUMNS = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]
TARGET = "Machine failure"

def load_data():
    DATA_PATH.parent.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        zip_path = DATA_PATH.parent / "ai4i_dataset.zip"
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            csv_name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
            DATA_PATH.write_bytes(archive.read(csv_name))
    df = pd.read_csv(DATA_PATH)
    return df[FEATURE_COLUMNS].copy(), df[TARGET].astype(int)

def make_preprocessor():
    numeric = FEATURE_COLUMNS[1:]
    categorical = ["Type"]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

def make_models():
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180, max_depth=4, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42,
            n_jobs=-1
        )
    }

def prepare_train_test():
    X, y = load_data()
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

def train_and_save():
    X_train, X_test, y_train, y_test = prepare_train_test()
    comparison = {}
    best_model, best_name, best_f1 = None, None, -1

    for name, classifier in make_models().items():
        pipeline = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", classifier)
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        f1 = f1_score(y_test, predictions, zero_division=0)
        comparison[name] = {
            "precision": round(precision_score(y_test, predictions, zero_division=0), 4),
            "recall": round(recall_score(y_test, predictions, zero_division=0), 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc_score(y_test, probabilities), 4)
        }
        if f1 > best_f1:
            best_model, best_name, best_f1 = pipeline, name, f1

    metrics = {"selected_model": best_name, "model_comparison": comparison}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "metrics": metrics}, MODEL_PATH)
    return {"model": best_model, "metrics": metrics}

def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_and_save()

if __name__ == "__main__":
    print(train_and_save()["metrics"])
