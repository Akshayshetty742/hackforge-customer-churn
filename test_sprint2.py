"""Tests for Sprint 2 Deliverables: What-If Simulator & Batch Customer Scoring."""

import numpy as np
import pandas as pd
import pytest

import prediction_pipeline as pp

@pytest.fixture(scope="module")
def artifacts():
    return pp.load_artifacts()


def test_score_customer_batch_holdout_sample(artifacts):
    """Verifies vectorized batch customer scoring on real holdout records."""
    model, feature_names, encoders = artifacts
    df = pp.load_dataset().head(25)
    
    scored_df = pp.score_customer_batch(df, model, encoders, feature_names, threshold=0.50)
    
    assert len(scored_df) == 25
    assert "Churn_Probability" in scored_df.columns
    assert "Predicted_Class" in scored_df.columns
    assert "Risk_Tier" in scored_df.columns
    assert set(scored_df["Predicted_Class"].unique()) <= {"Churn", "Stay"}
    assert set(scored_df["Risk_Tier"].unique()) <= {"High", "Moderate", "Low"}
    # Probabilities should be sorted descending
    probs = scored_df["Churn_Probability"].tolist()
    assert probs == sorted(probs, reverse=True)


def test_score_customer_batch_custom_threshold(artifacts):
    """Verifies that batch scoring properly uses custom decision threshold."""
    model, feature_names, encoders = artifacts
    df = pp.load_dataset().head(30)
    
    scored_low_thresh = pp.score_customer_batch(df, model, encoders, feature_names, threshold=0.20)
    scored_high_thresh = pp.score_customer_batch(df, model, encoders, feature_names, threshold=0.80)
    
    churn_low = (scored_low_thresh["Predicted_Class"] == "Churn").sum()
    churn_high = (scored_high_thresh["Predicted_Class"] == "Churn").sum()
    
    # Lower threshold must flag at least as many or more churners
    assert churn_low >= churn_high


def test_explain_customer_risk_attribution():
    """Verifies risk accelerators and protective anchors for known profiles."""
    high_risk_customer = {
        "Contract": "Month-to-month",
        "tenure": 1,
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "MonthlyCharges": 95.50,
    }
    
    explanation = pp.explain_customer_risk(high_risk_customer, 0.85)
    
    assert "risk_drivers" in explanation
    assert "protective_factors" in explanation
    assert len(explanation["risk_drivers"]) >= 3
    factors = [d["factor"] for d in explanation["risk_drivers"]]
    assert "Contract Type" in factors
    assert "Account Tenure" in factors


def test_what_if_counterfactual_risk_reduction(artifacts):
    """Verifies that counterfactual contract extension decreases churn probability."""
    model, feature_names, encoders = artifacts
    
    baseline_high_risk = {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 95.50,
    }
    
    processed_base = pp.preprocess_input(baseline_high_risk, encoders, feature_names)
    _, base_prob, _, _ = pp.predict_churn(model, processed_base, threshold=0.50)
    
    # Counterfactual intervention: upgrade contract to Two year and add TechSupport
    simulated_profile = dict(baseline_high_risk)
    simulated_profile["Contract"] = "Two year"
    simulated_profile["TechSupport"] = "Yes"
    simulated_profile["OnlineSecurity"] = "Yes"
    
    processed_sim = pp.preprocess_input(simulated_profile, encoders, feature_names)
    _, sim_prob, _, _ = pp.predict_churn(model, processed_sim, threshold=0.50)
    
    # Simulating long-term contract and security must significantly reduce churn probability
    assert sim_prob < base_prob
    assert (base_prob - sim_prob) > 0.30  # At least 30 percentage point risk reduction
