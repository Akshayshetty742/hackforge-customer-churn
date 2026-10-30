"""UI tests for the Streamlit prediction form (app.py) using streamlit's AppTest.

These run the real app script headless, so no browser is needed.
"""
# pylint: disable=redefined-outer-name  # pytest injects fixtures by parameter name

import importlib.util
import os

import pytest
from streamlit.testing.v1 import AppTest

import customer_presets as cp
import prediction_pipeline as pp

APP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
HIGH_RISK = "High risk: new fiber customer"


@pytest.fixture
def app():
    """Fresh headless run of the Streamlit app."""
    return AppTest.from_file(APP_FILE, default_timeout=60).run()


def click(at, label):
    """Clicks the button with this label and reruns the app."""
    next(b for b in at.button if b.label == label).click()
    return at.run()


def submit(at):
    """Submits the prediction form."""
    next(b for b in at.button if b.label == "Run Churn Prediction").click()
    return at.run()


def risk_pills(at):
    """Text of the rendered risk result pills."""
    return [m.value for m in at.markdown if "RISK" in m.value]


def test_app_starts_with_default_preset(app):
    assert not app.exception
    for col, value in cp.get_preset_values(cp.DEFAULT_PRESET).items():
        assert app.session_state[f"form_{col}"] == value


def test_every_input_has_label_and_help(app):
    widgets = list(app.selectbox) + list(app.number_input)
    form_widgets = [w for w in widgets if w.key and w.key.startswith("form_")]
    assert len(form_widgets) == len(pp.EXPECTED_FEATURE_COLUMNS)
    for w in form_widgets:
        assert w.label.strip(), w.key
        assert w.help, w.key


def test_loaded_preset_is_used_for_prediction(app):
    """Regression test: the preset used to reset back to low risk when the form was submitted."""
    app.selectbox(key="preset_choice").select(HIGH_RISK).run()
    click(app, "Load preset")
    submit(app)
    assert not app.exception
    assert app.number_input(key="form_tenure").value == 1
    assert any("HIGH RISK" in pill for pill in risk_pills(app))


def test_form_values_survive_page_switch(app):
    app.selectbox(key="preset_choice").select(HIGH_RISK).run()
    click(app, "Load preset")
    app.sidebar.radio[0].set_value("About the Project").run()
    app.sidebar.radio[0].set_value("Customer Churn Prediction").run()
    assert app.number_input(key="form_tenure").value == 1


def test_cleared_form_reports_missing_fields(app):
    click(app, "Clear form")
    submit(app)
    assert not app.exception
    assert risk_pills(app) == []
    message = app.error[0].value
    assert f"Please fix {len(pp.EXPECTED_FEATURE_COLUMNS)} problem(s)" in message
    assert "Missing required field: 'Contract'" in message


def test_invalid_combination_shows_error_instead_of_prediction(app):
    app.selectbox(key="form_InternetService").select("No")
    submit(app)
    assert risk_pills(app) == []
    assert "When InternetService is 'No'" in app.error[0].value


def test_estimated_total_charges(app):
    app.number_input(key="form_TotalCharges").set_value(1.0)  # wrong on purpose
    app.checkbox(key="auto_total_charges").check()
    submit(app)
    assert not app.error
    assert "Total Charges estimated" in app.info[0].value
    assert risk_pills(app)


def open_evaluation(at):
    """Switches the sidebar to the Model Evaluation page."""
    at.sidebar.radio[0].set_value("Model Evaluation").run()
    return at


def run_live(at):
    """Runs the live held-out evaluation on the evaluation page."""
    next(b for b in at.button if b.label.startswith("Run Test Evaluation")).click()
    return at.run()


def test_live_evaluation_matches_benchmark(app):
    run_live(open_evaluation(app))
    assert not app.exception
    assert any("match the verified benchmark" in s.value for s in app.success)
    labels = {m.label for m in app.metric}
    assert {"Accuracy", "Recall (Churn)", "Precision (Churn)", "F1-Score (Churn)"} <= labels


def test_live_evaluation_custom_threshold(app):
    open_evaluation(app)
    app.slider[0].set_value(0.3)
    run_live(app)
    assert not app.exception
    assert any("custom threshold of 0.30" in i.value for i in app.info)


def test_live_results_survive_page_switch(app):
    run_live(open_evaluation(app))
    app.sidebar.radio[0].set_value("About the Project").run()
    open_evaluation(app)
    assert any("match the verified benchmark" in s.value for s in app.success)


def paradigm_cards(at):
    """Markdown of the rendered ML / DML / QML comparison cards."""
    return [m.value for m in at.markdown if 'class="paradigm-card"' in m.value]


def test_prediction_shows_all_three_paradigms(app):
    app.selectbox(key="preset_choice").select(HIGH_RISK).run()
    click(app, "Load preset")
    submit(app)
    assert not app.exception
    cards = paradigm_cards(app)
    research_installed = all(importlib.util.find_spec(m) for m in ("torch", "pennylane"))
    if research_installed:
        assert len(cards) == 3
        for tag in ("ML · Classical ML", "DML · Deep Learning", "QML · Quantum ML"):
            assert any(tag in card for card in cards), tag
        assert any("paradigms predict" in s.value or "paradigms disagree" in s.value
                   for s in list(app.success) + list(app.info))
    else:
        # Without the research extras the page still works and says why DML / QML are missing
        assert len(cards) == 1
        assert sum("requirements-research.txt" in w.value for w in app.warning) == 2


def test_no_paradigm_comparison_for_invalid_input(app):
    click(app, "Clear form")
    submit(app)
    assert paradigm_cards(app) == []
