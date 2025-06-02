import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb

os.makedirs("images", exist_ok=True)
os.makedirs("joblib", exist_ok=True)

DATA_PATH = "data/input.csv"
MODEL_PATH = "joblib/model_xgb.joblib"
IMPUTER_PATH = "joblib/imputer.joblib"
ENCODERS_PATH = "joblib/label_encoders.joblib"
SCALER_PATH = "joblib/scaler.joblib"
FEATURE_COLS_PATH = "joblib/feature_columns.joblib"


def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    df.drop("id", axis=1, inplace=True)

    df["bmi"] = df["bmi"].replace("N/A", np.nan)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    imputer = SimpleImputer(strategy="mean")
    df["bmi"] = imputer.fit_transform(df[["bmi"]])

    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    df["age_glucose"] = df["age"] * df["avg_glucose_level"]

    X = df.drop("stroke", axis=1)
    y = df["stroke"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 1, 5]
    }

    search = RandomizedSearchCV(model, param_dist, n_iter=20, scoring="roc_auc", cv=3, verbose=1, random_state=42, n_jobs=-1)
    search.fit(X_train_res, y_train_res)
    best_model = search.best_estimator_

    feature_columns = list(X.columns)
    joblib.dump(feature_columns, FEATURE_COLS_PATH)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump((X_test, y_test), "test_data.joblib")


def test_model():
    model = joblib.load(MODEL_PATH)
    X_test, y_test = joblib.load("test_data.joblib")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("images/confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("images/roc_curve.png")
    plt.close()


def predict_new_patient(new_data: dict):
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLS_PATH)

    df_new = pd.DataFrame([new_data])
    df_new["bmi"] = df_new["bmi"].replace("N/A", np.nan)
    df_new["bmi"] = pd.to_numeric(df_new["bmi"], errors="coerce")
    df_new["bmi"] = imputer.transform(df_new[["bmi"]])

    for col, le in label_encoders.items():
        if col in df_new.columns:
            df_new[col] = le.transform(df_new[col])

    df_new["age_glucose"] = df_new["age"] * df_new["avg_glucose_level"]

    X = df_new[feature_columns]
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]
    return {"prediction": int(pred), "probability_of_stroke": int(float(proba * 100))}