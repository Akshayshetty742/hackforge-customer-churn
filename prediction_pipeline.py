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

# Allowed values for every categorical field (same classes the LabelEncoders were fitted on)
VALID_CATEGORIES: Dict[str, List[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

# Inclusive (min, max) bounds for numeric fields
NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 120),
    "MonthlyCharges": (0.0, 500.0),
    "TotalCharges": (0.0, 50000.0),
}

# Add-on services that only make sense when the customer has internet
INTERNET_DEPENDENT_COLUMNS: List[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Allowed TotalCharges / (MonthlyCharges * tenure) ratio for the billing consistency guard
BILLING_RATIO_RANGE: Tuple[float, float] = (0.5, 2.0)

# Human readable names used in error messages and form labels
FIELD_LABELS: Dict[str, str] = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract Type",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
}


def _is_missing(value: Any) -> bool:
    """True for None, empty / whitespace-only strings and NaN."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_number(value: Any) -> float:
    """Converts a value to a finite float.

    Raises:
        TypeError: for booleans and other non numeric types.
        ValueError: for non numeric strings, NaN and infinity.
    """
    # bool is a subclass of int, but True/False is never a sensible tenure or charge
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("boolean values are not accepted")
    if isinstance(value, str):
        value = value.strip()
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("value must be a finite number")
    return number

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
    if not isinstance(data, dict):
        return False, ["Customer input must be a dictionary of field values."]

    for col in EXPECTED_FEATURE_COLUMNS:
        if col not in data or _is_missing(data[col]):
            errors.append(f"Missing required field: '{col}' ({FIELD_LABELS[col]})")

    if errors:
        return False, errors

    # Categorical validations
    for col, allowed in VALID_CATEGORIES.items():
        value = data[col]
        if not isinstance(value, str):
            errors.append(
                f"{FIELD_LABELS[col]} must be text, got {type(value).__name__} ({value!r})."
            )
        elif value not in allowed:
            errors.append(
                f"Invalid value {value!r} for {FIELD_LABELS[col]}. "
                f"Allowed values: {', '.join(allowed)}."
            )

    # Numerical validations
    numbers: Dict[str, float] = {}
    for col, (low, high) in NUMERIC_RANGES.items():
        try:
            numbers[col] = _to_number(data[col])
        except (ValueError, TypeError):
            errors.append(f"{col} must be a valid number, got {data[col]!r}.")
            continue

        if col == "SeniorCitizen":
            if numbers[col] not in (0, 1):
                errors.append("SeniorCitizen must be either 0 (No) or 1 (Yes).")
            continue

        if not low <= numbers[col] <= high:
            if col == "tenure":
                errors.append(f"Tenure must be between {low} and {high} months.")
            else:
                errors.append(f"{col} must be between ${low:,.2f} and ${high:,.2f}.")
        elif col == "tenure" and not float(numbers[col]).is_integer():
            # preprocess_input casts tenure to int, so 12.7 would silently become 12
            errors.append("Tenure must be a whole number of months.")

    # Billing consistency: TotalCharges should roughly equal MonthlyCharges * tenure.
    # In the Telco dataset the ratio is always between ~0.69 and ~1.57, so 0.5 - 2.0 is generous.
    if not errors:
        tenure, monthly, total = numbers["tenure"], numbers["MonthlyCharges"], numbers["TotalCharges"]
        if tenure == 0:
            if total > monthly:
                errors.append(
                    "TotalCharges cannot be more than MonthlyCharges for a new customer (tenure 0)."
                )
        else:
            expected = monthly * tenure
            low, high = BILLING_RATIO_RANGE
            if not expected * low <= total <= expected * high:
                errors.append(
                    f"TotalCharges (${total:,.2f}) does not match MonthlyCharges x tenure "
                    f"(about ${expected:,.2f}). Please double check the billing values."
                )

    # Consistency checks: PhoneService
    if data.get("PhoneService") == "No":
        if data.get("MultipleLines") not in ("No phone service", "No"):
            errors.append("When PhoneService is 'No', MultipleLines should be 'No phone service'.")
    elif data.get("PhoneService") == "Yes" and data.get("MultipleLines") == "No phone service":
        errors.append("MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'.")

    # Consistency checks: InternetService
    if data.get("InternetService") == "No":
        for feat in INTERNET_DEPENDENT_COLUMNS:
            if data.get(feat) not in ("No internet service", "No"):
                errors.append(
                    f"When InternetService is 'No', {feat} should be 'No internet service'."
                )
    elif data.get("InternetService") in ("DSL", "Fiber optic"):
        for feat in INTERNET_DEPENDENT_COLUMNS:
            if data.get(feat) == "No internet service":
                errors.append(
                    f"{feat} cannot be 'No internet service' when InternetService is "
                    f"'{data['InternetService']}'."
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


def predict_churn(
    model: Any,
    processed_df: pd.DataFrame,
) -> Tuple[int, float, Dict[str, float]]:
    """Runs prediction on preprocessed dataframe.

    Returns:
        (predicted_class, churn_probability, probabilities_dict)
    """
    prediction = int(model.predict(processed_df)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(processed_df)[0]
        prob_churn = float(probs[1]) if len(probs) > 1 else float(probs[0])
        prob_stay = float(probs[0]) if len(probs) > 1 else 1.0 - prob_churn
    else:
        prob_churn = 1.0 if prediction == 1 else 0.0
        prob_stay = 1.0 - prob_churn

    prob_dict = {"Stay": prob_stay, "Churn": prob_churn}
    return prediction, prob_churn, prob_dict


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
