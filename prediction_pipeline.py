"""Prediction and Evaluation Pipeline for Customer Churn Prediction.

This module provides reusable utilities for loading models, encoders,
preprocessing input records, running inference, and evaluating performance.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_FILE = os.path.join(BASE_DIR, "customer_churn_model_xgboost.pkl")
DEFAULT_ENCODER_FILE = os.path.join(BASE_DIR, "encoders.pkl")
DEFAULT_DATASET_FILE = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Deep (DML) and quantum (QML) models saved by 02_paradigm_benchmark.py for side-by-side predictions
PARADIGM_MODEL_FILES: Dict[str, str] = {
    "DML": os.path.join(BASE_DIR, "models", "dml_model.pkl"),
    "QML": os.path.join(BASE_DIR, "models", "qml_model.pkl"),
}

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
    "test_specificity": 0.7585,
    "test_roc_auc": 0.8412,
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


def clean_customer_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a normalised copy of a raw customer record.

    - strips surrounding whitespace from text values (" Yes " -> "Yes")
    - converts numeric strings to numbers ("29.85" -> 29.85)
    - drops customerID since it is never used by the model

    Anything that cannot be converted is left untouched so that
    validate_customer_input() can report a proper error for it.
    """
    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if key == "customerID":
            continue
        if isinstance(value, str):
            value = value.strip()
        if key in NUMERIC_RANGES and not _is_missing(value):
            try:
                number = _to_number(value)
                value = int(number) if key in ("tenure", "SeniorCitizen") and number.is_integer() else number
            except (ValueError, TypeError):
                pass
        cleaned[key] = value
    return cleaned


def _missing_field_errors(data: Dict[str, Any]) -> List[str]:
    return [
        f"Missing required field: '{col}' ({FIELD_LABELS[col]})"
        for col in EXPECTED_FEATURE_COLUMNS
        if col not in data or _is_missing(data[col])
    ]


def _categorical_errors(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
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
    return errors


def _numeric_errors(data: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
    """Returns the parsed numeric fields and the problems found with them."""
    errors: List[str] = []
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
    return numbers, errors


def _billing_errors(numbers: Dict[str, float]) -> List[str]:
    """TotalCharges should roughly equal MonthlyCharges * tenure.

    In the Telco dataset the ratio is always between ~0.69 and ~1.57, so 0.5 - 2.0 is generous.
    """
    tenure, monthly, total = numbers["tenure"], numbers["MonthlyCharges"], numbers["TotalCharges"]
    if tenure == 0:
        if total > monthly:
            return ["TotalCharges cannot be more than MonthlyCharges for a new customer (tenure 0)."]
        return []

    expected = monthly * tenure
    low, high = BILLING_RATIO_RANGE
    if not expected * low <= total <= expected * high:
        return [
            f"TotalCharges (${total:,.2f}) does not match MonthlyCharges x tenure "
            f"(about ${expected:,.2f}). Please double check the billing values."
        ]
    return []


def _phone_consistency_errors(data: Dict[str, Any]) -> List[str]:
    if data.get("PhoneService") == "No":
        if data.get("MultipleLines") not in ("No phone service", "No"):
            return ["When PhoneService is 'No', MultipleLines should be 'No phone service'."]
    elif data.get("PhoneService") == "Yes" and data.get("MultipleLines") == "No phone service":
        return ["MultipleLines cannot be 'No phone service' when PhoneService is 'Yes'."]
    return []


def _internet_consistency_errors(data: Dict[str, Any]) -> List[str]:
    internet = data.get("InternetService")
    if internet == "No":
        return [
            f"When InternetService is 'No', {feat} should be 'No internet service'."
            for feat in INTERNET_DEPENDENT_COLUMNS
            if data.get(feat) not in ("No internet service", "No")
        ]
    if internet in ("DSL", "Fiber optic"):
        return [
            f"{feat} cannot be 'No internet service' when InternetService is '{internet}'."
            for feat in INTERNET_DEPENDENT_COLUMNS
            if data.get(feat) == "No internet service"
        ]
    return []


def validate_customer_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates input payload fields, ranges, and consistency.

    Returns:
        (is_valid, list_of_error_messages)
    """
    if not isinstance(data, dict):
        return False, ["Customer input must be a dictionary of field values."]

    errors = _missing_field_errors(data)
    if errors:
        return False, errors

    errors += _categorical_errors(data)
    numbers, numeric_errors = _numeric_errors(data)
    errors += numeric_errors

    # The billing cross-check only means something once every field is individually valid
    if not errors:
        errors += _billing_errors(numbers)

    errors += _phone_consistency_errors(data)
    errors += _internet_consistency_errors(data)

    return not errors, errors


def _coerce_numeric_columns(df: pd.DataFrame) -> None:
    """Converts the numeric features in place, rejecting booleans, text and non-finite values."""
    if "TotalCharges" in df.columns:
        # blank TotalCharges means a brand new customer (same rule used in training)
        df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0", "": "0.0"}).fillna("0.0")

    for col in NUMERICAL_COLUMNS:
        if col not in df.columns:
            continue
        if df[col].map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise ValueError(f"Boolean values are not valid for numeric feature '{col}'.")
        converted = pd.to_numeric(df[col], errors="coerce")
        bad_rows = converted.isna() | ~np.isfinite(converted.fillna(0))
        if bad_rows.any():
            bad_values = df.loc[bad_rows, col].tolist()
            raise ValueError(f"Non-numeric or missing value(s) {bad_values} for feature '{col}'.")
        df[col] = converted.astype(float)

    for col in ("tenure", "SeniorCitizen"):
        if col in df.columns:
            if not (df[col] % 1 == 0).all():
                raise ValueError(f"Feature '{col}' must contain whole numbers only.")
            df[col] = df[col].astype(int)


def _encode_categoricals(df: pd.DataFrame, encoders: Dict[str, Any]) -> None:
    """Encodes categorical columns in place with the fitted encoders (never refits)."""
    for col, enc in encoders.items():
        if col in df.columns and col != "Churn":
            if df[col].isna().any():
                raise ValueError(f"Missing value for categorical feature '{col}'.")
            known_classes = set(enc.classes_)
            unseen = set(df[col].dropna()) - known_classes
            if unseen:
                raise ValueError(
                    f"Unseen category value(s) {unseen} for feature '{col}'. "
                    f"Valid categories are: {list(enc.classes_)}"
                )
            df[col] = enc.transform(df[col])


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

    _coerce_numeric_columns(df)
    _encode_categoricals(df, encoders)

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
    if prob_churn >= 0.35:
        return "Moderate"
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


def load_dataset(dataset_path: str = DEFAULT_DATASET_FILE) -> pd.DataFrame:
    """Loads the Telco CSV and applies the same cleaning as the training script.

    - drops customerID
    - blank TotalCharges (brand new customers) -> 0.0

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if the target or feature columns are missing.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: '{dataset_path}'")

    df = pd.read_csv(dataset_path)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    missing = [c for c in EXPECTED_FEATURE_COLUMNS + ["Churn"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    return df


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any = None,
) -> Dict[str, Any]:
    """Computes the churn (positive class = 1) metrics for a set of predictions.

    Returns accuracy, precision, recall, f1, specificity, the confusion matrix
    ([[TN, FP], [FN, TP]]) and sklearn's classification report. If churn
    probabilities are given, ROC-AUC is added too.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty set of predictions.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} rows but y_pred has {len(y_pred)}.")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0][0]), int(cm[0][1])

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # zero_division=0 so a model that never predicts churn gives 0 instead of a warning
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["Stay", "Churn"],
            output_dict=True, zero_division=0,
        ),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def confusion_matrix_breakdown(cm: Any) -> Dict[str, int]:
    """Names the 4 cells of a 2x2 confusion matrix laid out as [[TN, FP], [FN, TP]]."""
    cm = np.asarray(cm)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 confusion matrix, got shape {cm.shape}.")
    return {
        "true_negatives": int(cm[0][0]),   # loyal customers correctly predicted to stay
        "false_positives": int(cm[0][1]),  # loyal customers flagged as churn
        "false_negatives": int(cm[1][0]),  # churners the model missed
        "true_positives": int(cm[1][1]),   # churners correctly flagged
    }


def confusion_matrix_frame(cm: Any) -> pd.DataFrame:
    """Labelled DataFrame version of the confusion matrix for display."""
    return pd.DataFrame(
        np.asarray(cm),
        columns=["Pred: Stay", "Pred: Churn"],
        index=["Actual: Stay", "Actual: Churn"],
    )


def get_holdout_split(
    df: pd.DataFrame,
    encoders: Dict[str, Any],
    feature_names: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Encodes the cleaned dataset and returns the held-out test split (X_test, y_test).

    Uses the same stratified split as 01_churn_prediction_xgboost.py
    (test_size=0.2, random_state=42), so the model never saw these rows in training.
    The fitted encoders are only used to transform, never refit.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    df = df.copy()
    for col, enc in encoders.items():
        if col in df.columns:
            unseen = set(df[col].dropna()) - set(enc.classes_)
            if unseen:
                raise ValueError(f"Dataset has unseen value(s) {unseen} in column '{col}'.")
            df[col] = enc.transform(df[col])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_test[feature_names], y_test


def evaluate_model_on_dataset(
    dataset_path: str = DEFAULT_DATASET_FILE,
    model_path: str = DEFAULT_MODEL_FILE,
    encoder_path: str = DEFAULT_ENCODER_FILE,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Runs a live evaluation of the saved model on the dataset test split.

    Uses the exact stratified test split protocol without retraining the model.
    A customer is predicted to churn when P(churn) >= threshold (0.5 = model default).
    """
    if not 0 < threshold < 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}.")

    model, feature_names, encoders = load_artifacts(model_path, encoder_path)
    df = load_dataset(dataset_path)
    X_test, y_test = get_holdout_split(df, encoders, feature_names, test_size, random_state)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_prob = None
        y_pred = model.predict(X_test)

    results = compute_classification_metrics(y_test, y_pred, y_prob)
    results.update(
        {
            "total_test_samples": len(y_test),
            "test_churn_count": int(y_test.sum()),
            "test_stay_count": int(len(y_test) - y_test.sum()),
            "threshold": threshold,
        }
    )
    return results


def compare_with_benchmark(
    results: Dict[str, Any],
    benchmark: Optional[Dict[str, Any]] = None,
    tolerance: float = 0.005,
) -> Tuple[bool, pd.DataFrame]:
    """Compares live evaluation results with the saved benchmark metrics.

    Returns (all_match, table) where the table has one row per metric with the
    saved value, the live value, the difference and whether it is within tolerance.
    The confusion matrix has to match exactly. benchmark defaults to SAVED_BENCHMARK_METRICS.
    """
    if benchmark is None:
        benchmark = SAVED_BENCHMARK_METRICS
    rows = []
    for metric in ("accuracy", "precision", "recall", "f1"):
        saved = float(benchmark[f"test_{metric}"])
        live = float(results[metric])
        rows.append(
            {
                "Metric": metric.capitalize() if metric != "f1" else "F1-score",
                "Saved": f"{saved:.4f}",
                "Live": f"{live:.4f}",
                "Difference": f"{live - saved:+.4f}",
                "Match": abs(live - saved) <= tolerance,
            }
        )

    cm_match = np.array_equal(np.asarray(results["confusion_matrix"]), np.asarray(benchmark["confusion_matrix"]))
    rows.append(
        {
            "Metric": "Confusion matrix",
            "Saved": str(benchmark["confusion_matrix"]),
            "Live": str(results["confusion_matrix"]),
            "Difference": "-" if cm_match else "differs",
            "Match": bool(cm_match),
        }
    )
    table = pd.DataFrame(rows)
    return bool(table["Match"].all()), table


def explain_customer_risk(
    customer: Dict[str, Any],
    churn_prob: float,  # pylint: disable=unused-argument  # part of the public API used by app.py
) -> Dict[str, List[Dict[str, str]]]:
    """Analyzes a customer record against top XGBoost model drivers to produce
    explainable risk accelerators and protective anchors for decision support.

    The factors are rule based, so churn_prob does not change them; it is accepted
    so callers can pass the model output alongside the record.
    """
    risk_drivers: List[Dict[str, str]] = []
    protective_factors: List[Dict[str, str]] = []

    # 1. Contract Type (Feature importance ~39.1%)
    contract = customer.get("Contract")
    if contract == "Month-to-month":
        risk_drivers.append({
            "factor": "Contract Type",
            "detail": "Month-to-month contract (39.1% global feature importance) creates highest churn velocity.",
            "impact": "High Risk",
            "badge_color": "#dc2626",
        })
    elif contract in ("One year", "Two year"):
        protective_factors.append({
            "factor": "Contract Type",
            "detail": f"{contract} contract provides institutional commitment and retention stability.",
            "impact": "Protective",
            "badge_color": "#16a34a",
        })

    # 2. Tenure Duration (Early lifecycle vulnerability)
    try:
        tenure = int(customer.get("tenure", 0))
    except (ValueError, TypeError):
        tenure = 0

    if tenure <= 6:
        risk_drivers.append({
            "factor": "Account Tenure",
            "detail": f"New customer ({tenure} months) in high-hazard early lifecycle onboarding window.",
            "impact": "High Risk",
            "badge_color": "#dc2626",
        })
    elif tenure >= 24:
        protective_factors.append({
            "factor": "Account Tenure",
            "detail": f"Established customer relationship ({tenure} months) demonstrates brand loyalty.",
            "impact": "Protective",
            "badge_color": "#16a34a",
        })

    # 3. Payment Method
    payment = customer.get("PaymentMethod")
    if payment == "Electronic check":
        risk_drivers.append({
            "factor": "Payment Method",
            "detail": "Electronic check carries elevated churn rates compared to automated options.",
            "impact": "Moderate Risk",
            "badge_color": "#d97706",
        })
    elif payment in ("Bank transfer (automatic)", "Credit card (automatic)"):
        protective_factors.append({
            "factor": "Payment Method",
            "detail": f"Automated billing ({payment}) minimizes payment friction and involuntary churn.",
            "impact": "Protective",
            "badge_color": "#16a34a",
        })

    # 4. Internet Service & Support Add-ons
    internet = customer.get("InternetService")
    security = customer.get("OnlineSecurity")
    support = customer.get("TechSupport")

    if internet == "Fiber optic" and (security == "No" or support == "No"):
        risk_drivers.append({
            "factor": "Service Ecosystem",
            "detail": (
                "Premium Fiber Optic plan lacking Online Security / Tech Support add-ons "
                "reduces product stickiness."
            ),
            "impact": "High Risk",
            "badge_color": "#dc2626",
        })
    elif security == "Yes" and support == "Yes":
        protective_factors.append({
            "factor": "Product Stickiness",
            "detail": "Subscribed to Online Security and Tech Support add-ons (higher switching barrier).",
            "impact": "Protective",
            "badge_color": "#16a34a",
        })

    # 5. Billing Burden
    try:
        monthly = float(customer.get("MonthlyCharges", 0))
    except (ValueError, TypeError):
        monthly = 0.0

    if monthly > 80.0 and tenure < 12:
        risk_drivers.append({
            "factor": "Price Sensitivity",
            "detail": f"High monthly billing (${monthly:,.2f}/mo) on newer account elevates price sensitivity.",
            "impact": "Moderate Risk",
            "badge_color": "#d97706",
        })

    return {
        "risk_drivers": risk_drivers,
        "protective_factors": protective_factors,
    }


def score_customer_batch(
    df: pd.DataFrame,
    model: Any,
    encoders: Dict[str, Any],
    feature_names: List[str],
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Vectorized scoring of customer records returning probabilities and risk tiers."""
    clean_df = df.copy()
    customer_ids = (
        clean_df["customerID"].copy()
        if "customerID" in clean_df.columns
        else [f"CUST-{i+1:04d}" for i in range(len(clean_df))]
    )
    if "customerID" in clean_df.columns:
        clean_df = clean_df.drop(columns=["customerID"])
    if "Churn" in clean_df.columns:
        actual_churn = clean_df["Churn"].copy()
        clean_df = clean_df.drop(columns=["Churn"])
    else:
        actual_churn = None

    processed_df = preprocess_input(clean_df, encoders, feature_names)
    probs = model.predict_proba(processed_df)[:, 1]
    preds = (probs >= threshold).astype(int)

    result_df = df.copy()
    if "customerID" not in result_df.columns:
        result_df.insert(0, "customerID", customer_ids)
    result_df["Churn_Probability"] = np.round(probs * 100, 1)
    result_df["Predicted_Class"] = np.where(preds == 1, "Churn", "Stay")
    result_df["Risk_Tier"] = [get_risk_tier(p) for p in probs]
    if actual_churn is not None:
        result_df["Actual_Churn"] = actual_churn

    return result_df.sort_values(by="Churn_Probability", ascending=False)


def count_model_parameters(model: Any) -> int:
    """Learned parameters of a fitted model, used to compare model size across paradigms.

    Neural / quantum models: trainable weights. Gradient-boosted trees: every tree node (each
    split stores a threshold, each leaf a value). Linear models: coefficients + intercepts.
    Random forests: tree nodes.
    """
    if hasattr(model, "n_parameters"):
        return int(model.n_parameters)
    if hasattr(model, "get_booster"):
        return int(len(model.get_booster().trees_to_dataframe()))
    if hasattr(model, "coef_"):
        return int(model.coef_.size + model.intercept_.size)
    if hasattr(model, "estimators_"):
        return int(sum(tree.tree_.node_count for tree in model.estimators_))
    raise TypeError(f"Cannot count parameters of {type(model).__name__}.")


def load_paradigm_model(model_path: str, feature_names: List[str]) -> Dict[str, Any]:
    """Loads a DML / QML model saved by 02_paradigm_benchmark.py.

    Returns the saved payload: model, display_name, test_metrics and trainable_parameters.
    Raises:
        FileNotFoundError: if the file does not exist (run 02_paradigm_benchmark.py).
        ImportError: if torch / pennylane are not installed (requirements-research.txt).
        ValueError: if the file is not a model payload or expects other input features.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at: '{model_path}'")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(model_path, "rb") as f:
            payload = pickle.load(f)

    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"'{model_path}' is not a saved model payload.")
    if payload.get("features_names") != feature_names:
        raise ValueError(f"'{model_path}' was trained on different input features than the production model.")
    return payload


def predict_all_paradigms(
    processed_df: pd.DataFrame,
    models: Dict[str, Any],
    threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """Scores one preprocessed customer with every available model.

    Args:
        processed_df: Output of preprocess_input (one row, production feature order).
        models: Paradigm label ("ML", "DML", "QML") -> fitted model, in display order.
        threshold: Decision threshold applied to every model.

    Returns:
        One dict per model with paradigm, churn_probability, prediction (1 = churn),
        label ("Churn" / "Stay") and risk_tier.
    """
    results = []
    for paradigm, model in models.items():
        prediction, churn_prob, _, risk_tier = predict_churn(model, processed_df, threshold=threshold)
        results.append(
            {
                "paradigm": paradigm,
                "churn_probability": churn_prob,
                "prediction": prediction,
                "label": "Churn" if prediction == 1 else "Stay",
                "risk_tier": risk_tier,
            }
        )
    return results
