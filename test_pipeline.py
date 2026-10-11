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

import os
import pytest
import pandas as pd
import numpy as np

import prediction_pipeline as pp


@pytest.fixture
def sample_valid_input():
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
    """Test that end-to-end prediction returns expected class (0 or 1) and valid probability."""
    model, feature_names, encoders = loaded_artifacts
    processed_df = pp.preprocess_input(sample_valid_input, encoders, feature_names)
    pred_class, churn_prob, prob_dict = pp.predict_churn(model, processed_df)

    assert pred_class in (0, 1)
    assert 0.0 <= churn_prob <= 1.0
    assert "Stay" in prob_dict and "Churn" in prob_dict
    assert pytest.approx(prob_dict["Stay"] + prob_dict["Churn"], rel=1e-4) == 1.0


def test_customer_id_is_dropped_if_present(sample_valid_input, loaded_artifacts):
    """Ensure customerID is stripped and not sent to the model."""
    _, feature_names, encoders = loaded_artifacts
    input_with_id = sample_valid_input.copy()
    input_with_id["customerID"] = "9999-ABCD"

    processed = pp.preprocess_input(input_with_id, encoders, feature_names)
    assert "customerID" not in processed.columns
    assert list(processed.columns) == feature_names
