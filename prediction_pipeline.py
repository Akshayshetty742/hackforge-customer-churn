"""Prediction and Evaluation Pipeline for Customer Churn Prediction.

This module provides reusable utilities for loading models, encoders,
preprocessing input records, running inference, and evaluating performance.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_FILE = os.path.join(BASE_DIR, "customer_churn_model_xgboost.pkl")
DEFAULT_ENCODER_FILE = os.path.join(BASE_DIR, "encoders.pkl")
DEFAULT_DATASET_FILE = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

EXPECTED_FEATURE_COLUMNS: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS: List[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERICAL_COLUMNS: List[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# Verified benchmark metrics from the saved XGBoost model tested on stratified 20% test split (random_state=42)
SAVED_BENCHMARK_METRICS: Dict[str, Any] = {
    "test_accuracy": 0.7566,
    "test_precision": 0.5292,
    "test_recall": 0.7513,
    "test_f1": 0.6210,
    "confusion_matrix": [[785, 250], [93, 281]],
    "test_samples": 1409,
    "majority_minority_ratio": 2.77,
}


def load_artifacts(
    model_path: str = DEFAULT_MODEL_FILE,
    encoder_path: str = DEFAULT_ENCODER_FILE,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """Loads the model and encoder artifacts safely from disk.

    Returns:
        (model, feature_names, encoders_dict)
    Raises:
        FileNotFoundError: If any of the artifact files do not exist.
        ValueError: If artifact contents are invalid or corrupted.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: '{model_path}'")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder artifact not found at: '{encoder_path}'")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

    if isinstance(model_data, dict):
        if "model" not in model_data:
            raise ValueError("Model dictionary missing 'model' key.")
        model = model_data["model"]
        feature_names = model_data.get("features_names", EXPECTED_FEATURE_COLUMNS)
    else:
        model = model_data
        feature_names = EXPECTED_FEATURE_COLUMNS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(encoder_path, "rb") as f:
            encoders = pickle.load(f)

    if not isinstance(encoders, dict):
        raise ValueError("Encoders artifact is not a valid dictionary.")

    return model, feature_names, encoders


def validate_customer_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates input payload fields, ranges, and consistency.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: List[str] = []

    # Check for missing required features
    for col in EXPECTED_FEATURE_COLUMNS:
        if col not in data or data[col] is None or data[col] == "":
            errors.append(f"Missing required field: '{col}'")

    if errors:
        return False, errors

    # Numerical validations
    try:
        tenure = float(data["tenure"])
        if tenure < 0 or tenure > 120:
            errors.append("Tenure must be between 0 and 120 months.")
    except (ValueError, TypeError):
        errors.append("Tenure must be a valid number.")

    try:
        monthly = float(data["MonthlyCharges"])
        if monthly < 0 or monthly > 500:
            errors.append("MonthlyCharges must be between $0.00 and $500.00.")
    except (ValueError, TypeError):
        errors.append("MonthlyCharges must be a valid number.")

    try:
        total = float(data["TotalCharges"])
        if total < 0 or total > 50000:
            errors.append("TotalCharges must be between $0.00 and $50,000.00.")
    except (ValueError, TypeError):
        errors.append("TotalCharges must be a valid number.")

    try:
        senior = int(data["SeniorCitizen"])
        if senior not in (0, 1):
            errors.append("SeniorCitizen must be either 0 (No) or 1 (Yes).")
    except (ValueError, TypeError):
        errors.append("SeniorCitizen must be 0 or 1.")

    # Consistency checks: PhoneService
    if data.get("PhoneService") == "No":
        if data.get("MultipleLines") not in ("No phone service", "No"):
            errors.append("When PhoneService is 'No', MultipleLines should be 'No phone service'.")

    # Consistency checks: InternetService
    if data.get("InternetService") == "No":
        internet_features = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        for feat in internet_features:
            if data.get(feat) not in ("No internet service", "No"):
                errors.append(
                    f"When InternetService is 'No', {feat} should be 'No internet service'."
                )

    is_valid = len(errors) == 0
    return is_valid, errors


def preprocess_input(
    raw_data: Union[Dict[str, Any], pd.DataFrame],
    encoders: Dict[str, Any],
    feature_names: List[str],
) -> pd.DataFrame:
    """Preprocesses a single dictionary input or DataFrame using fitted encoders.

    Ensures exact feature ordering, conversion, and validation against encoder classes.
    Raises:
        ValueError: If unknown categorical values or missing columns are encountered.
    """
    if isinstance(raw_data, dict):
        df = pd.DataFrame([raw_data])
    else:
        df = raw_data.copy()

    # Drop customerID if accidentally passed
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert numeric fields
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"]
            .replace({" ": "0.0", "": "0.0"})
            .fillna("0.0")
            .astype(float)
        )
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].astype(int)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Encode categoricals using existing fitted encoders
    for col, enc in encoders.items():
        if col in df.columns and col != "Churn":
            known_classes = set(enc.classes_)
            unseen = set(df[col].dropna()) - known_classes
            if unseen:
                raise ValueError(
                    f"Unseen category value(s) {unseen} for feature '{col}'. "
                    f"Valid categories are: {list(enc.classes_)}"
                )
            # Transform without refitting
            df[col] = enc.transform(df[col])

    # Reorder according to the trained model's feature names
    missing_features = [col for col in feature_names if col not in df.columns]
    if missing_features:
        raise ValueError(f"Input is missing required model features: {missing_features}")

    return df[feature_names]


def get_risk_tier(prob_churn: float) -> str:
    """Classifies churn probability into business operational risk tiers.

    Note on Risk Bands:
        - Low Risk (< 35%): Customer account indicates stable retention.
        - Moderate Risk (35% to 65%): Customer displays emerging churn warning signs.
        - High Risk (> 65%): Customer is at severe risk of cancellation.

    Disclaimer:
        These risk category tiers are heuristics designed for business workflow
        segmentation and retention intervention priority, not statistically validated cutoffs.
    """
    if prob_churn > 0.65:
        return "High"
    elif prob_churn >= 0.35:
        return "Moderate"
    else:
        return "Low"


def predict_churn(
    model: Any,
    processed_df: pd.DataFrame,
    threshold: float = 0.50,
) -> Tuple[int, float, Dict[str, float], str]:
    """Runs prediction on preprocessed dataframe with configurable decision threshold.

    Args:
        model: Trained model artifact supporting predict or predict_proba.
        processed_df: Preprocessed feature DataFrame with exact model feature order.
        threshold: Classification decision threshold for churn (default 0.50).

    Returns:
        (predicted_class, churn_probability, probabilities_dict, risk_tier)
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(processed_df)[0]
        prob_churn = float(probs[1]) if len(probs) > 1 else float(probs[0])
        prob_stay = float(probs[0]) if len(probs) > 1 else 1.0 - prob_churn
    else:
        raw_pred = int(model.predict(processed_df)[0])
        prob_churn = 1.0 if raw_pred == 1 else 0.0
        prob_stay = 1.0 - prob_churn

    # Determine predicted class based on configurable decision threshold
    prediction = 1 if prob_churn >= threshold else 0
    prob_dict = {"Stay": prob_stay, "Churn": prob_churn}
    risk_tier = get_risk_tier(prob_churn)

    return prediction, prob_churn, prob_dict, risk_tier


def evaluate_model_on_dataset(
    dataset_path: str = DEFAULT_DATASET_FILE,
    model_path: str = DEFAULT_MODEL_FILE,
    encoder_path: str = DEFAULT_ENCODER_FILE,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs a live evaluation of the saved model on the dataset test split.

    Uses the exact stratified test split protocol without retraining the model.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: '{dataset_path}'")

    model, feature_names, encoders = load_artifacts(model_path, encoder_path)

    df = pd.read_csv(dataset_path)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)

    for col, enc in encoders.items():
        if col in df.columns:
            df[col] = enc.transform(df[col])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_test_ordered = X_test[feature_names]
    y_pred = model.predict(X_test_ordered)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "total_test_samples": len(y_test),
        "test_churn_count": int(y_test.sum()),
        "test_stay_count": int(len(y_test) - y_test.sum()),
    }
