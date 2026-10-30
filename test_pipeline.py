"""Unit and Integration Tests for Hackforge Customer Churn Prediction Pipeline.

Covers:
- Module imports
- Model & encoder artifact loading
- Valid customer input preprocessing and prediction
- Invalid numerical input handling & validation errors
- Unknown categorical values detection
- Feature ordering & column integrity
- Predict proba shape and output consistency
- Graceful handling of missing model / encoder files
- Mock testing for missing artifacts
"""
# pylint: disable=redefined-outer-name  # pytest injects fixtures by parameter name

import pytest
import pandas as pd
import numpy as np

import customer_presets as cp
import prediction_pipeline as pp


@pytest.fixture
def sample_valid_input():
    """First record of the Telco dataset, which passes every validation rule."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


@pytest.fixture
def loaded_artifacts():
    """Saved model, feature order and encoders."""
    return pp.load_artifacts()


def test_imports():
    """Verify core modules and dependencies import cleanly."""
    assert hasattr(pp, "load_artifacts")
    assert hasattr(pp, "validate_customer_input")
    assert hasattr(pp, "preprocess_input")
    assert hasattr(pp, "predict_churn")
    assert hasattr(pp, "evaluate_model_on_dataset")


def test_load_artifacts(loaded_artifacts):
    """Test that model, feature names, and encoders load properly."""
    model, feature_names, encoders = loaded_artifacts
    assert model is not None
    assert isinstance(feature_names, list)
    assert len(feature_names) == 19
    assert isinstance(encoders, dict)
    assert "Contract" in encoders
    assert "InternetService" in encoders
    assert "PaymentMethod" in encoders


def test_missing_model_file_handling(tmp_path):
    """Test that load_artifacts raises FileNotFoundError when model path does not exist."""
    fake_model = str(tmp_path / "nonexistent_model.pkl")
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        pp.load_artifacts(model_path=fake_model)


def test_missing_encoder_file_handling(tmp_path):
    """Test that load_artifacts raises FileNotFoundError when encoder path does not exist."""
    fake_encoder = str(tmp_path / "nonexistent_encoder.pkl")
    # Provide the real model path but fake encoder path
    real_model = pp.DEFAULT_MODEL_FILE
    with pytest.raises(FileNotFoundError, match="Encoder artifact not found"):
        pp.load_artifacts(model_path=real_model, encoder_path=fake_encoder)


def test_valid_customer_input_validation(sample_valid_input):
    """Test validator passes with a well-formed customer input dictionary."""
    is_valid, errors = pp.validate_customer_input(sample_valid_input)
    assert is_valid is True
    assert len(errors) == 0


def test_invalid_numerical_input_negative_tenure(sample_valid_input):
    """Test validator catches negative tenure."""
    bad_input = sample_valid_input.copy()
    bad_input["tenure"] = -5
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("Tenure" in err for err in errors)


def test_invalid_numerical_input_excessive_charges(sample_valid_input):
    """Test validator catches out-of-bounds charges."""
    bad_input = sample_valid_input.copy()
    bad_input["MonthlyCharges"] = 9999.0
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("MonthlyCharges" in err for err in errors)


def test_invalid_senior_citizen_value(sample_valid_input):
    """Test validator catches invalid SeniorCitizen indicator."""
    bad_input = sample_valid_input.copy()
    bad_input["SeniorCitizen"] = 5
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("SeniorCitizen" in err for err in errors)


def test_missing_required_fields(sample_valid_input):
    """Test validator catches missing required fields."""
    bad_input = sample_valid_input.copy()
    del bad_input["Contract"]
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("Contract" in err for err in errors)


def test_feature_ordering_and_preprocessing(sample_valid_input, loaded_artifacts):
    """Test preprocessing produces the exact feature column ordering expected by the model."""
    _, feature_names, encoders = loaded_artifacts
    processed_df = pp.preprocess_input(sample_valid_input, encoders, feature_names)

    assert isinstance(processed_df, pd.DataFrame)
    assert list(processed_df.columns) == feature_names
    assert len(processed_df) == 1
    # Check that categoricals became integers
    assert np.issubdtype(processed_df["gender"].dtype, np.integer)
    assert np.issubdtype(processed_df["Contract"].dtype, np.integer)
    # Check that numerics are float/int
    assert np.issubdtype(processed_df["TotalCharges"].dtype, np.floating)


def test_unknown_categorical_value_handling(sample_valid_input, loaded_artifacts):
    """Test preprocessing raises an explicit ValueError when an unseen category is supplied."""
    _, feature_names, encoders = loaded_artifacts
    bad_input = sample_valid_input.copy()
    bad_input["Contract"] = "Lifetime-Contract"

    with pytest.raises(ValueError, match="Unseen category value"):
        pp.preprocess_input(bad_input, encoders, feature_names)


def test_successful_prediction_output(sample_valid_input, loaded_artifacts):
    """Test that end-to-end prediction returns expected class (0 or 1), probability, and risk tier."""
    model, feature_names, encoders = loaded_artifacts
    processed_df = pp.preprocess_input(sample_valid_input, encoders, feature_names)
    pred_class, churn_prob, prob_dict, risk_tier = pp.predict_churn(model, processed_df)

    assert pred_class in (0, 1)
    assert 0.0 <= churn_prob <= 1.0
    assert "Stay" in prob_dict and "Churn" in prob_dict
    assert pytest.approx(prob_dict["Stay"] + prob_dict["Churn"], rel=1e-4) == 1.0
    assert risk_tier in ("Low", "Moderate", "High")


def test_customer_id_is_dropped_if_present(sample_valid_input, loaded_artifacts):
    """Ensure customerID is stripped and not sent to the model."""
    _, feature_names, encoders = loaded_artifacts
    input_with_id = sample_valid_input.copy()
    input_with_id["customerID"] = "9999-ABCD"

    processed = pp.preprocess_input(input_with_id, encoders, feature_names)
    assert "customerID" not in processed.columns
    assert list(processed.columns) == feature_names


def test_predict_churn_custom_threshold(sample_valid_input, loaded_artifacts):
    """Test that decision threshold shifts classification sensitivity."""
    model, feature_names, encoders = loaded_artifacts
    processed_df = pp.preprocess_input(sample_valid_input, encoders, feature_names)

    # First inspect raw probability
    _, prob_churn, _, _ = pp.predict_churn(model, processed_df)

    # If threshold is lower than prob, must predict 1 (Churn)
    lower_threshold = max(0.01, prob_churn - 0.05)
    pred_low, _, _, _ = pp.predict_churn(model, processed_df, threshold=lower_threshold)
    assert pred_low == 1

    # If threshold is higher than prob, must predict 0 (Stay)
    higher_threshold = min(0.99, prob_churn + 0.05)
    pred_high, _, _, _ = pp.predict_churn(model, processed_df, threshold=higher_threshold)
    assert pred_high == 0


def test_threshold_boundary_cases(sample_valid_input, loaded_artifacts):
    """Test exact threshold boundary conditions (prob == threshold)."""
    model, feature_names, encoders = loaded_artifacts
    processed_df = pp.preprocess_input(sample_valid_input, encoders, feature_names)
    _, prob_churn, _, _ = pp.predict_churn(model, processed_df)

    # Exact equality: prob_churn >= threshold -> should be class 1
    pred_exact, _, _, _ = pp.predict_churn(model, processed_df, threshold=prob_churn)
    assert pred_exact == 1


def test_get_risk_tier_classification():
    """Verify get_risk_tier maps probability ranges to Low, Moderate, and High tiers."""
    # Low risk (< 0.35)
    assert pp.get_risk_tier(0.0) == "Low"
    assert pp.get_risk_tier(0.20) == "Low"
    assert pp.get_risk_tier(0.349) == "Low"

    # Moderate risk (0.35 to 0.65)
    assert pp.get_risk_tier(0.35) == "Moderate"
    assert pp.get_risk_tier(0.50) == "Moderate"
    assert pp.get_risk_tier(0.65) == "Moderate"

    # High risk (> 0.65)
    assert pp.get_risk_tier(0.651) == "High"
    assert pp.get_risk_tier(0.85) == "High"
    assert pp.get_risk_tier(1.0) == "High"


# ---------------------------------------------------------
# Input validation & boundary guards (issue #3)
# ---------------------------------------------------------

def test_valid_categories_match_fitted_encoders(loaded_artifacts):
    """VALID_CATEGORIES must stay in sync with the classes the encoders were fitted on."""
    _, _, encoders = loaded_artifacts
    for col, allowed in pp.VALID_CATEGORIES.items():
        assert sorted(allowed) == sorted(encoders[col].classes_), col


@pytest.mark.parametrize("blank", [None, "", "   ", float("nan")])
def test_blank_values_reported_as_missing(sample_valid_input, blank):
    bad_input = sample_valid_input.copy()
    bad_input["PaymentMethod"] = blank
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("Missing required field: 'PaymentMethod'" in err for err in errors)


def test_non_dict_input_is_rejected():
    is_valid, errors = pp.validate_customer_input(["not", "a", "dict"])
    assert is_valid is False
    assert len(errors) == 1


def test_invalid_category_value_is_rejected(sample_valid_input):
    bad_input = sample_valid_input.copy()
    bad_input["Contract"] = "Lifetime-Contract"
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("Invalid value 'Lifetime-Contract'" in err for err in errors)
    # the message should tell the user what is allowed
    assert any("Month-to-month" in err for err in errors)


def test_category_with_wrong_type_is_rejected(sample_valid_input):
    bad_input = sample_valid_input.copy()
    bad_input["gender"] = 1
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("Gender must be text" in err for err in errors)


@pytest.mark.parametrize("field", ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"])
@pytest.mark.parametrize("bad_value", ["abc", True, float("inf"), [1]])
def test_numeric_fields_reject_bad_types(sample_valid_input, field, bad_value):
    bad_input = sample_valid_input.copy()
    bad_input[field] = bad_value
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any(field in err for err in errors)


def test_clean_customer_input_normalises_values(sample_valid_input):
    raw = {k: (f"  {v}  " if isinstance(v, str) else str(v)) for k, v in sample_valid_input.items()}
    raw["customerID"] = "7590-VHVEG"
    cleaned = pp.clean_customer_input(raw)

    assert "customerID" not in cleaned
    assert cleaned["gender"] == "Female"
    assert cleaned["tenure"] == 1 and isinstance(cleaned["tenure"], int)
    assert cleaned["MonthlyCharges"] == pytest.approx(29.85)
    assert pp.validate_customer_input(cleaned) == (True, [])


def test_clean_customer_input_keeps_unconvertible_values(sample_valid_input):
    raw = sample_valid_input.copy()
    raw["TotalCharges"] = "not a number"
    cleaned = pp.clean_customer_input(raw)
    assert cleaned["TotalCharges"] == "not a number"
    assert pp.validate_customer_input(cleaned)[0] is False


@pytest.mark.parametrize(
    "field, value, ok",
    [
        ("tenure", 0, True),
        ("tenure", 120, True),
        ("tenure", 121, False),
        ("tenure", -1, False),
        ("tenure", 12.5, False),
        ("MonthlyCharges", 500.0, True),
        ("MonthlyCharges", 500.01, False),
        ("MonthlyCharges", -0.01, False),
        ("TotalCharges", 50000.01, False),
        ("TotalCharges", -1, False),
    ],
)
def test_numeric_range_boundaries(field, value, ok):
    """Edges of NUMERIC_RANGES are inclusive, anything outside is rejected."""
    record = {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "tenure": 10, "MonthlyCharges": 50.0, "TotalCharges": 500.0,
    }
    record[field] = value
    # keep TotalCharges consistent so only the range rule is being tested
    if field in ("tenure", "MonthlyCharges") and ok:
        record["TotalCharges"] = min(record["tenure"] * record["MonthlyCharges"], 50000.0)
    is_valid, errors = pp.validate_customer_input(record)
    assert is_valid is ok, errors


def test_total_charges_inconsistent_with_tenure(sample_valid_input):
    bad_input = sample_valid_input.copy()
    bad_input.update({"tenure": 72, "MonthlyCharges": 20.0, "TotalCharges": 40000.0})
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("does not match MonthlyCharges x tenure" in err for err in errors)


def test_new_customer_total_cannot_exceed_monthly(sample_valid_input):
    bad_input = sample_valid_input.copy()
    bad_input.update({"tenure": 0, "MonthlyCharges": 30.0, "TotalCharges": 300.0})
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("new customer" in err for err in errors)


def test_phone_service_yes_with_no_phone_service_lines(sample_valid_input):
    bad_input = sample_valid_input.copy()
    bad_input.update({"PhoneService": "Yes", "MultipleLines": "No phone service"})
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("MultipleLines" in err for err in errors)


def test_internet_addons_require_internet(sample_valid_input):
    bad_input = sample_valid_input.copy()
    bad_input.update({"InternetService": "No", "StreamingTV": "Yes"})
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("StreamingTV" in err for err in errors)


def test_internet_customer_cannot_have_no_internet_addons(sample_valid_input):
    bad_input = sample_valid_input.copy()
    bad_input.update({"InternetService": "Fiber optic", "TechSupport": "No internet service"})
    is_valid, errors = pp.validate_customer_input(bad_input)
    assert is_valid is False
    assert any("TechSupport" in err for err in errors)


def test_every_dataset_row_passes_validation():
    """The guards must never reject a real customer from the training data."""
    df = pd.read_csv(pp.DEFAULT_DATASET_FILE)
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)
    records = df.drop(columns=["customerID", "Churn"]).to_dict("records")
    rejected = [r for r in records if not pp.validate_customer_input(r)[0]]
    assert rejected == []


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("tenure", "abc", "Non-numeric"),
        ("tenure", 3.5, "whole numbers"),
        ("MonthlyCharges", True, "Boolean"),
        ("MonthlyCharges", float("inf"), "Non-numeric"),
        ("gender", None, "Missing value"),
    ],
)
def test_preprocess_input_guards(sample_valid_input, loaded_artifacts, field, value, message):
    """preprocess_input should fail with a clear ValueError even if validation was skipped."""
    _, feature_names, encoders = loaded_artifacts
    bad_input = sample_valid_input.copy()
    bad_input[field] = value
    with pytest.raises(ValueError, match=message):
        pp.preprocess_input(bad_input, encoders, feature_names)


# ---------------------------------------------------------
# Customer form presets (issue #1)
# ---------------------------------------------------------

@pytest.mark.parametrize("name", list(cp.CUSTOMER_PRESETS))
def test_preset_is_complete_and_valid(name):
    values = cp.get_preset_values(name)
    assert set(values) == set(pp.EXPECTED_FEATURE_COLUMNS)
    assert pp.validate_customer_input(values) == (True, [])
    assert cp.CUSTOMER_PRESETS[name]["description"].strip()


@pytest.mark.parametrize("name", list(cp.CUSTOMER_PRESETS))
def test_preset_prediction_matches_its_label(name, loaded_artifacts):
    """A 'High risk' preset should be predicted to churn and a 'Low risk' one to stay."""
    model, feature_names, encoders = loaded_artifacts
    processed = pp.preprocess_input(cp.get_preset_values(name), encoders, feature_names)
    _, churn_prob, _, _ = pp.predict_churn(model, processed)
    if name.startswith("High risk"):
        assert churn_prob > 0.5
    elif name.startswith("Low risk"):
        assert churn_prob < 0.5
    else:
        # borderline preset should sit somewhere around the threshold
        assert 0.3 <= churn_prob <= 0.7


def test_default_preset_exists():
    assert cp.DEFAULT_PRESET in cp.CUSTOMER_PRESETS


def test_get_preset_values_returns_a_copy():
    name = cp.DEFAULT_PRESET
    values = cp.get_preset_values(name)
    values["tenure"] = 999
    assert cp.get_preset_values(name)["tenure"] != 999


def test_unknown_preset_raises_key_error():
    with pytest.raises(KeyError, match="Unknown preset"):
        cp.get_preset_values("Does not exist")


# ---------------------------------------------------------
# Model evaluation (issue #4)
# ---------------------------------------------------------

@pytest.fixture(scope="module")
def live_results():
    """Live evaluation of the saved model on the held-out split."""
    return pp.evaluate_model_on_dataset()


def test_holdout_split_is_the_training_test_split(loaded_artifacts):
    """Held-out set = same stratified 20% split as the training script (1409 rows, ~26.5% churn)."""
    _, feature_names, encoders = loaded_artifacts
    X_test, y_test = pp.get_holdout_split(pp.load_dataset(), encoders, feature_names)
    assert len(X_test) == pp.SAVED_BENCHMARK_METRICS["test_samples"]
    assert list(X_test.columns) == feature_names
    assert int(y_test.sum()) == 374


def test_live_evaluation_reports_all_metrics(live_results):
    for key in ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc"):
        assert 0.0 <= live_results[key] <= 1.0, key
    assert live_results["total_test_samples"] == 1409
    assert live_results["test_churn_count"] + live_results["test_stay_count"] == 1409


def test_live_evaluation_matches_saved_benchmark(live_results):
    all_match, table = pp.compare_with_benchmark(live_results)
    assert all_match, table
    assert len(table) == 5


def test_confusion_matrix_is_consistent(live_results):
    cm = live_results["confusion_matrix"]
    cells = pp.confusion_matrix_breakdown(cm)
    assert sum(cells.values()) == live_results["total_test_samples"]
    assert cells["true_positives"] + cells["false_negatives"] == live_results["test_churn_count"]
    frame = pp.confusion_matrix_frame(cm)
    assert list(frame.columns) == ["Pred: Stay", "Pred: Churn"]
    assert list(frame.index) == ["Actual: Stay", "Actual: Churn"]


def test_compute_classification_metrics_on_known_values():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 1, 1, 1]  # TN=3 FP=1 FN=1 TP=3
    m = pp.compute_classification_metrics(y_true, y_pred, y_prob=[0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    assert m["confusion_matrix"] == [[3, 1], [1, 3]]
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(0.75)
    assert m["roc_auc"] == pytest.approx(0.9375)
    assert {"Stay", "Churn"} <= set(m["classification_report"])


def test_metrics_when_model_never_predicts_churn():
    m = pp.compute_classification_metrics([0, 1, 0, 1], [0, 0, 0, 0])
    assert m["precision"] == 0.0 and m["recall"] == 0.0
    assert m["confusion_matrix"] == [[2, 0], [2, 0]]


def test_metrics_input_errors():
    with pytest.raises(ValueError, match="empty"):
        pp.compute_classification_metrics([], [])
    with pytest.raises(ValueError, match="rows"):
        pp.compute_classification_metrics([0, 1], [0])
    with pytest.raises(ValueError, match="2x2"):
        pp.confusion_matrix_breakdown([[1, 2, 3]])


def test_lower_threshold_catches_more_churners(live_results):
    low = pp.evaluate_model_on_dataset(threshold=0.3)
    assert low["recall"] > live_results["recall"]
    assert low["threshold"] == 0.3
    assert pp.compare_with_benchmark(low)[0] is False
    with pytest.raises(ValueError, match="threshold"):
        pp.evaluate_model_on_dataset(threshold=1.5)


def test_missing_dataset_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        pp.evaluate_model_on_dataset(dataset_path=str(tmp_path / "nope.csv"))


def test_dataset_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.read_csv(pp.DEFAULT_DATASET_FILE).drop(columns=["Contract"]).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        pp.load_dataset(str(bad_csv))
