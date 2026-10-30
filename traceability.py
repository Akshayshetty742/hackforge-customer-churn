"""Test case catalogue: Test Plan v1.1 test cases, the SRS requirements they verify, and the tests that run them.

Single source of truth for the requirement -> test case -> automated test chain. conftest.py tags every
collected test with its test case ID, and tools/generate_test_reports.py builds the System Validation
Report from the same tables, so the documents and the test run cannot drift apart.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple


class TestCase(NamedTuple):
    """One Test Plan test case."""

    title: str
    requirements: List[str]
    level: str  # Unit, Integration, System or Non-functional


TEST_CASES: Dict[str, TestCase] = {
    # Test Plan v1.0 (Deliverable 1): training pipeline
    "TC-01": TestCase("Load valid dataset", ["FR-01", "SW-01"], "Integration"),
    "TC-02": TestCase("Missing file / missing target handled", ["FR-02", "NFR-02", "NFR-10", "SR-05", "SO-03"], "Unit"),
    "TC-03": TestCase("Target separation and encoding", ["FR-03", "FR-04"], "Unit"),
    "TC-04": TestCase("XGBoost training with class weighting", ["FR-05", "FR-10"], "Integration"),
    "TC-05": TestCase("Grid search with stratified cross validation", ["FR-06", "FR-11"], "Integration"),
    "TC-06": TestCase("Evaluation metrics and reports", ["FR-07", "FR-08", "FR-12", "NFR-04", "UI-01"], "System"),
    "TC-07": TestCase("Class imbalance handling, test set untouched", ["FR-10"], "Integration"),
    "TC-08": TestCase("Model and encoders saved", ["FR-09", "SW-02"], "System"),
    "TC-09": TestCase("Saved model and encoders reload as a pair", ["FR-14", "NFR-07", "SR-03", "SO-02"],
                      "Integration"),
    "TC-10": TestCase("Fresh environment and training runtime", ["NFR-01", "NFR-06", "HW-01"], "Non-functional"),
    "TC-11": TestCase("Privacy and repository hygiene", ["NFR-09", "SR-01", "SR-02", "SR-04", "SO-01", "COM-01"],
                      "Non-functional"),
    "TC-12": TestCase("Documentation completeness", ["FR-13", "NFR-05", "NFR-08"], "Non-functional"),
    "TC-13": TestCase("Repeatability", ["NFR-03"], "Non-functional"),
    # Test Plan v1.1: features added in Sprints 1-3
    "TC-14": TestCase("Customer profile form and presets", ["FR-15", "UI-02"], "System"),
    "TC-15": TestCase("Input validation and boundary guards", ["FR-16", "NFR-08", "SR-05"], "Unit"),
    "TC-16": TestCase("Single customer prediction with risk tier", ["FR-17"], "Integration"),
    "TC-17": TestCase("Live evaluation and decision threshold tuning", ["FR-18"], "System"),
    "TC-18": TestCase("Batch CSV scoring", ["FR-19"], "Integration"),
    "TC-19": TestCase("Churn driver explanation and what-if simulator", ["FR-20"], "Integration"),
    "TC-20": TestCase("ML vs DML vs QML side-by-side prediction", ["FR-21", "SR-07"], "System"),
    "TC-21": TestCase("Deep learning model (DML)", ["FR-22"], "Unit"),
    "TC-22": TestCase("Quantum machine learning model (QML)", ["FR-23"], "Unit"),
    "TC-23": TestCase("Paradigm benchmark and experiment tracking", ["FR-24", "NFR-12", "SR-06", "SW-03"], "System"),
    "TC-24": TestCase("Prediction latency", ["NFR-11"], "Non-functional"),
    "TC-25": TestCase("Production model and encoder integrity", ["NFR-07", "SO-02"], "Integration"),
    "TC-26": TestCase("Static code quality", ["NFR-13"], "Non-functional"),
}

# Tests written before the Test Plan v1.1 IDs existed, mapped by test function name.
# Tests in test_system_validation.py carry an explicit @pytest.mark.tc marker instead.
EXISTING_TEST_MAP: Dict[str, List[str]] = {
    # test_app.py (Streamlit UI, run headless with AppTest)
    "test_app_starts_with_default_preset": ["TC-14"],
    "test_every_input_has_label_and_help": ["TC-14"],
    "test_loaded_preset_is_used_for_prediction": ["TC-16"],
    "test_form_values_survive_page_switch": ["TC-14"],
    "test_cleared_form_reports_missing_fields": ["TC-15"],
    "test_invalid_combination_shows_error_instead_of_prediction": ["TC-15"],
    "test_estimated_total_charges": ["TC-14"],
    "test_live_evaluation_matches_benchmark": ["TC-17"],
    "test_live_evaluation_custom_threshold": ["TC-17"],
    "test_live_results_survive_page_switch": ["TC-17"],
    "test_prediction_shows_all_three_paradigms": ["TC-20"],
    "test_no_paradigm_comparison_for_invalid_input": ["TC-20"],
    # test_sprint2.py
    "test_score_customer_batch_holdout_sample": ["TC-18"],
    "test_score_customer_batch_custom_threshold": ["TC-18"],
    "test_explain_customer_risk_attribution": ["TC-19"],
    "test_what_if_counterfactual_risk_reduction": ["TC-19"],
    # test_paradigms.py
    "test_predict_proba_is_a_valid_distribution": ["TC-21", "TC-22"],
    "test_pickle_round_trip_keeps_predictions": ["TC-21", "TC-22"],
    "test_early_stopping_restores_best_epoch": ["TC-21", "TC-22"],
    "test_clone_keeps_hyperparameters": ["TC-21"],
    "test_mlp_parameter_count_matches_architecture": ["TC-21"],
    "test_vqc_parameter_split": ["TC-22"],
    "test_vqc_features_are_rotation_angles": ["TC-22"],
    "test_single_qubit_circuit_reduces_to_cosine": ["TC-22"],
    "test_benchmark_rejects_unknown_paradigm": ["TC-23"],
    "test_benchmark_plan_tags_each_paradigm": ["TC-23"],
    "test_classical_parameter_count_for_linear_model": ["TC-23"],
    "test_parameter_count_of_torch_models": ["TC-21", "TC-22"],
    "test_saved_paradigm_models_carry_their_test_metrics": ["TC-20"],
    "test_predict_all_paradigms_scores_one_customer_with_every_model": ["TC-20"],
    "test_threshold_applies_to_every_paradigm": ["TC-20"],
    "test_load_paradigm_model_errors": ["TC-20"],
    # test_pipeline.py
    "test_imports": ["TC-10"],
    "test_load_artifacts": ["TC-09"],
    "test_missing_model_file_handling": ["TC-09"],
    "test_missing_encoder_file_handling": ["TC-09"],
    "test_valid_customer_input_validation": ["TC-15"],
    "test_invalid_numerical_input_negative_tenure": ["TC-15"],
    "test_invalid_numerical_input_excessive_charges": ["TC-15"],
    "test_invalid_senior_citizen_value": ["TC-15"],
    "test_missing_required_fields": ["TC-15"],
    "test_feature_ordering_and_preprocessing": ["TC-16"],
    "test_unknown_categorical_value_handling": ["TC-15"],
    "test_successful_prediction_output": ["TC-16"],
    "test_customer_id_is_dropped_if_present": ["TC-11"],
    "test_predict_churn_custom_threshold": ["TC-17"],
    "test_threshold_boundary_cases": ["TC-17"],
    "test_get_risk_tier_classification": ["TC-16"],
    "test_valid_categories_match_fitted_encoders": ["TC-25"],
    "test_blank_values_reported_as_missing": ["TC-15"],
    "test_non_dict_input_is_rejected": ["TC-15"],
    "test_invalid_category_value_is_rejected": ["TC-15"],
    "test_category_with_wrong_type_is_rejected": ["TC-15"],
    "test_numeric_fields_reject_bad_types": ["TC-15"],
    "test_clean_customer_input_normalises_values": ["TC-15"],
    "test_clean_customer_input_keeps_unconvertible_values": ["TC-15"],
    "test_numeric_range_boundaries": ["TC-15"],
    "test_total_charges_inconsistent_with_tenure": ["TC-15"],
    "test_new_customer_total_cannot_exceed_monthly": ["TC-15"],
    "test_phone_service_yes_with_no_phone_service_lines": ["TC-15"],
    "test_internet_addons_require_internet": ["TC-15"],
    "test_internet_customer_cannot_have_no_internet_addons": ["TC-15"],
    "test_every_dataset_row_passes_validation": ["TC-15"],
    "test_preprocess_input_guards": ["TC-15"],
    "test_preset_is_complete_and_valid": ["TC-14"],
    "test_preset_prediction_matches_its_label": ["TC-16"],
    "test_default_preset_exists": ["TC-14"],
    "test_get_preset_values_returns_a_copy": ["TC-14"],
    "test_unknown_preset_raises_key_error": ["TC-14"],
    "test_holdout_split_is_the_training_test_split": ["TC-07"],
    "test_live_evaluation_reports_all_metrics": ["TC-06"],
    "test_live_evaluation_matches_saved_benchmark": ["TC-17"],
    "test_confusion_matrix_is_consistent": ["TC-06"],
    "test_compute_classification_metrics_on_known_values": ["TC-06"],
    "test_metrics_when_model_never_predicts_churn": ["TC-06"],
    "test_metrics_input_errors": ["TC-06"],
    "test_lower_threshold_catches_more_churners": ["TC-17"],
    "test_missing_dataset_file": ["TC-02"],
    "test_dataset_missing_columns": ["TC-02"],
}


def requirement_index() -> Dict[str, List[str]]:
    """Requirement ID -> test case IDs that verify it, in test case order."""
    index: Dict[str, List[str]] = {}
    for case_id, case in TEST_CASES.items():
        for requirement in case.requirements:
            index.setdefault(requirement, []).append(case_id)
    return index
