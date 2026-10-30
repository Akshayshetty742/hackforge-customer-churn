"""Hackforge — Customer Churn Prediction Streamlit Web Application.

Deliverable Part 2: Interactive ML Inference, Model Evaluation, and Architecture Overview.
"""

from __future__ import annotations

import importlib
import os
import sys
import altair as alt
import pandas as pd
import streamlit as st

# Ensure local imports work reliably
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import customer_presets as cp  # pylint: disable=wrong-import-position  # needs the sys.path entry above
import prediction_pipeline as pp  # pylint: disable=wrong-import-position

# Pick up edits to the helper modules on Streamlit reruns; best effort, the imported versions still work
try:
    importlib.reload(pp)
    importlib.reload(cp)
except Exception:  # pylint: disable=broad-exception-caught
    pass

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hackforge — Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean, modern executive light-mode look
st.markdown(
    """
    <style>
    /* Global light mode surface */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    /* Clean executive cards */
    .bright-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    .bright-card-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Persona quick-cards */
    .persona-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .persona-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .persona-tag-high {
        font-size: 0.75rem;
        font-weight: 700;
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 2px 8px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .persona-tag-mod {
        font-size: 0.75rem;
        font-weight: 700;
        background-color: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
        border-radius: 12px;
        padding: 2px 8px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .persona-tag-low {
        font-size: 0.75rem;
        font-weight: 700;
        background-color: #f0fdf4;
        color: #16a34a;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 2px 8px;
        display: inline-block;
        margin-bottom: 6px;
    }
    .persona-desc {
        font-size: 0.8rem;
        color: #64748b;
        line-height: 1.35;
        margin-bottom: 8px;
    }
    
    /* Metrics */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4f46e5;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    /* Risk Pills */
    .churn-pill-high {
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #f87171;
        padding: 8px 18px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }
    .churn-pill-low {
        background-color: #f0fdf4;
        color: #16a34a;
        border: 1px solid #4ade80;
        padding: 8px 18px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }
    .churn-pill-moderate {
        background-color: #fffbeb;
        color: #d97706;
        border: 1px solid #fcd34d;
        padding: 8px 18px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }
    
    /* Recommendation & Alert Boxes */
    .recommendation-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 12px;
        color: #1e293b;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .alert-disclaimer {
        font-size: 0.85rem;
        color: #475569;
        background: #f1f5f9;
        padding: 14px 18px;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        margin-top: 16px;
    }
    
    /* Gauge Meter */
    .gauge-container {
        width: 100%;
        height: 18px;
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        position: relative;
        margin: 10px 0 8px 0;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* ML vs DML vs QML comparison cards */
    .paradigm-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .paradigm-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
    }
    .paradigm-model {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 8px;
    }
    .paradigm-prob {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 10px;
        font-variant-numeric: tabular-nums;
    }
    .paradigm-sub {
        font-size: 0.8rem;
        color: #64748b;
    }
    .paradigm-verdict {
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .paradigm-foot {
        font-size: 0.75rem;
        color: #64748b;
        border-top: 1px solid #f1f5f9;
        margin-top: 12px;
        padding-top: 8px;
        font-variant-numeric: tabular-nums;
    }

    /* Attribution Chips */
    .driver-chip-danger {
        background: #ffffff;
        border: 1px solid #fee2e2;
        border-left: 4px solid #dc2626;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.875rem;
        color: #1e293b;
    }
    .driver-chip-warning {
        background: #ffffff;
        border: 1px solid #fef3c7;
        border-left: 4px solid #d97706;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.875rem;
        color: #1e293b;
    }
    .driver-chip-success {
        background: #ffffff;
        border: 1px solid #dcfce7;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.875rem;
        color: #1e293b;
    }
    
    /* Sprint Badge */
    .sprint-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 700;
        background: #e0e7ff;
        color: #4338ca;
        border: 1px solid #c7d2fe;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Simulator & Counterfactual styling */
    .sim-card {
        background-color: #ffffff;
        border: 1px solid #c7d2fe;
        border-radius: 10px;
        padding: 18px;
        margin-top: 14px;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(79, 70, 229, 0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading machine learning artifacts...")
def get_cached_artifacts():
    """Load and cache the trained model, feature order, and LabelEncoders once."""
    return pp.load_artifacts()


@st.cache_resource(show_spinner="Loading deep learning and quantum models...")
def get_cached_paradigm_models(feature_names: list) -> dict:
    """DML / QML payloads by paradigm, or the reason a model could not be loaded (str)."""
    loaded = {}
    for paradigm, path in pp.PARADIGM_MODEL_FILES.items():
        try:
            loaded[paradigm] = pp.load_paradigm_model(path, feature_names)
        except ImportError:
            loaded[paradigm] = "needs PyTorch and PennyLane: pip install -r requirements-research.txt"
        except (FileNotFoundError, ValueError) as exc:
            loaded[paradigm] = f"{exc} Run python 02_paradigm_benchmark.py to create it."
    return loaded


@st.cache_resource(show_spinner=False)
def get_ml_parameter_count() -> int:
    """Tree nodes in the production XGBoost model."""
    return pp.count_model_parameters(get_cached_artifacts()[0])


@st.cache_data(show_spinner=False)
def run_live_evaluation(threshold: float) -> dict:
    """Cached so moving between pages / re-running doesn't re-score the whole test set."""
    return pp.evaluate_model_on_dataset(threshold=threshold)


# ---------------------------------------------------------
# Form state helpers
# ---------------------------------------------------------
# Every form widget has a key in st.session_state, so the values survive reruns
# (e.g. after pressing a preset button and then submitting the form).
def field_key(col: str) -> str:
    """Session-state key of the form widget for a feature column."""
    return f"form_{col}"


def load_preset(name: str) -> None:
    """Button callback: copy a preset's values into the form widgets."""
    for col, value in cp.get_preset_values(name).items():
        st.session_state[field_key(col)] = value
    st.session_state["loaded_preset"] = name


def estimate_total_charges(monthly, tenure):
    """Monthly x tenure (a new customer with tenure 0 has paid nothing yet). None if inputs are missing."""
    if monthly is None or tenure is None:
        return None
    return round(float(monthly) * int(tenure), 2)


def profile_summary(values: dict) -> pd.DataFrame:
    """Readable two-column table (Field / Value) of a customer record."""
    rows = []
    for col in pp.EXPECTED_FEATURE_COLUMNS:
        value = values.get(col)
        if col == "SeniorCitizen":
            value = "Yes" if value == 1 else "No"
        elif col in ("MonthlyCharges", "TotalCharges"):
            value = f"${float(value):,.2f}"
        rows.append({"Field": pp.FIELD_LABELS[col], "Value": str(value)})
    return pd.DataFrame(rows)


def clear_form() -> None:
    """Button callback: empty every field so a customer can be typed in from scratch."""
    for col in pp.EXPECTED_FEATURE_COLUMNS:
        st.session_state[field_key(col)] = None
    st.session_state["loaded_preset"] = None


def current_form_values() -> dict:
    """Current form values keyed by feature column."""
    return {col: st.session_state.get(field_key(col)) for col in pp.EXPECTED_FEATURE_COLUMNS}


# Tooltip text shown next to every form field
FIELD_HELP = {
    "gender": "Customer's gender as recorded on the account.",
    "SeniorCitizen": "Is the customer 65 or older?",
    "Partner": "Does the customer have a partner / spouse?",
    "Dependents": "Does the customer have dependents (children, parents, etc.)?",
    "tenure": "Number of whole months the customer has been with the company (0 = brand new).",
    "PhoneService": "Does the customer have a home phone service?",
    "MultipleLines": "More than one phone line? Use 'No phone service' if Phone Service is 'No'.",
    "InternetService": "Type of internet connection, or 'No' if the customer has no internet.",
    "OnlineSecurity": "Online security add-on. Use 'No internet service' if Internet Service is 'No'.",
    "OnlineBackup": "Online backup add-on. Use 'No internet service' if Internet Service is 'No'.",
    "DeviceProtection": "Device protection plan. Use 'No internet service' if Internet Service is 'No'.",
    "TechSupport": "Premium tech support. Use 'No internet service' if Internet Service is 'No'.",
    "StreamingTV": "Streams TV through the internet service.",
    "StreamingMovies": "Streams movies through the internet service.",
    "Contract": "Current contract length.",
    "PaperlessBilling": "Does the customer receive bills electronically?",
    "PaymentMethod": "How the customer pays their bill.",
    "MonthlyCharges": "Amount currently billed every month, in dollars.",
    "TotalCharges": "Total amount billed over the whole tenure, in dollars (roughly Monthly Charges x Tenure).",
}


def category_select(col: str) -> None:
    """Labelled selectbox for one categorical field, bound to session state."""
    st.selectbox(
        pp.FIELD_LABELS[col],
        options=pp.VALID_CATEGORIES[col],
        help=FIELD_HELP[col],
        placeholder="Select an option...",
        key=field_key(col),
    )


def numeric_input(col: str, step: float) -> None:
    """Labelled number input whose limits come from pp.NUMERIC_RANGES (same as the validator)."""
    low, high = pp.NUMERIC_RANGES[col]
    if col == "tenure":
        st.number_input(
            pp.FIELD_LABELS[col], min_value=int(low), max_value=int(high), step=int(step),
            help=FIELD_HELP[col], key=field_key(col),
        )
    else:
        st.number_input(
            pp.FIELD_LABELS[col], min_value=float(low), max_value=float(high), step=float(step),
            format="%.2f", help=f"{FIELD_HELP[col]} Allowed range: ${low:,.2f} - ${high:,.2f}.",
            key=field_key(col),
        )


if field_key("gender") not in st.session_state:
    load_preset(cp.DEFAULT_PRESET)

# Streamlit deletes a widget's state when the widget is not drawn on a run (e.g. while
# the user is on the "Model Evaluation" page). Re-assigning the keys every run keeps
# the form values when the user comes back to the prediction page.
for _key in [field_key(c) for c in pp.EXPECTED_FEATURE_COLUMNS] + ["preset_choice", "auto_total_charges"]:
    if _key in st.session_state:
        st.session_state[_key] = st.session_state[_key]


def explain_customer_risk(customer: dict, churn_prob: float) -> dict:
    """Risk accelerators and protective anchors for the attribution panel.

    The rules live in prediction_pipeline so the app, batch scoring and tests share one copy.
    """
    return pp.explain_customer_risk(customer, churn_prob)


PARADIGM_STYLE = {
    "ML": {"title": "Classical ML", "color": "#2563eb"},
    "DML": {"title": "Deep Learning", "color": "#7c3aed"},
    "QML": {"title": "Quantum ML", "color": "#0891b2"},
}


def paradigm_details(ml_model, feature_names: list) -> tuple:
    """(models, details, unavailable) for the three paradigms, in display order."""
    benchmark = pp.SAVED_BENCHMARK_METRICS
    models = {"ML": ml_model}
    details = {
        "ML": {
            "name": "XGBoost (tuned, production)",
            "roc_auc": benchmark["test_roc_auc"],
            "recall": benchmark["test_recall"],
            "parameters": get_ml_parameter_count(),
        }
    }
    unavailable = {}
    for paradigm, payload in get_cached_paradigm_models(feature_names).items():
        if isinstance(payload, str):
            unavailable[paradigm] = payload
            continue
        model_obj = payload["model"]
        name = "PyTorch MLP " + "-".join(str(units) for units in model_obj.hidden_layers) \
            if paradigm == "DML" else f"PennyLane VQC · {model_obj.n_qubits} qubits, {model_obj.n_layers} layers"
        models[paradigm] = model_obj
        details[paradigm] = {
            "name": name,
            "roc_auc": payload["test_metrics"]["roc_auc"],
            "recall": payload["test_metrics"]["churn_recall"],
            "parameters": payload["trainable_parameters"],
        }
    return models, details, unavailable


def paradigm_card(result: dict, detail: dict) -> str:
    """HTML card with one model's prediction for the customer."""
    style = PARADIGM_STYLE[result["paradigm"]]
    prob = result["churn_probability"]
    prob_color = "#dc2626" if prob >= 0.65 else "#d97706" if prob >= 0.35 else "#16a34a"
    verdict_color = "#dc2626" if result["prediction"] == 1 else "#16a34a"
    verdict = "Predicted to churn" if result["prediction"] == 1 else "Predicted to stay"
    return f"""
        <div class="paradigm-card" style="border-top: 4px solid {style['color']};">
            <span class="paradigm-tag" style="background: {style['color']}1a; color: {style['color']};">
                {result['paradigm']} · {style['title']}
            </span>
            <div class="paradigm-model">{detail['name']}</div>
            <div class="paradigm-prob" style="color: {prob_color};">{prob * 100:.1f}%</div>
            <div class="paradigm-sub">churn probability</div>
            <div class="gauge-container">
                <div class="gauge-fill" style="width: {min(100.0, prob * 100):.1f}%; background-color: {prob_color};"></div>
            </div>
            <div class="paradigm-verdict" style="color: {verdict_color};">{verdict}</div>
            <div class="paradigm-sub">Risk tier: {result['risk_tier']}</div>
            <div class="paradigm-foot">
                Test ROC-AUC {detail['roc_auc']:.3f} · Recall {detail['recall'] * 100:.1f}%<br>
                {detail['parameters']:,} learned parameters
            </div>
        </div>
    """


def paradigm_probability_chart(results: list, threshold: float) -> alt.LayerChart:
    """Churn probability per paradigm with the decision threshold marked."""
    data = pd.DataFrame(
        {
            "Paradigm": [f"{r['paradigm']} · {PARADIGM_STYLE[r['paradigm']]['title']}" for r in results],
            "Churn probability": [r["churn_probability"] for r in results],
            "color": [PARADIGM_STYLE[r["paradigm"]]["color"] for r in results],
        }
    )
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=22).encode(
        x=alt.X("Churn probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("Paradigm:N", sort=None, title=None),
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=["Paradigm", alt.Tooltip("Churn probability:Q", format=".1%")],
    )
    rule = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(
        color="#0f172a", strokeDash=[4, 4]
    ).encode(x="threshold:Q")
    return (bars + rule).properties(height=40 * len(results) + 30)


def paradigm_agreement(results: list, threshold: float) -> tuple:
    """(all_agree, message) summarising the paradigms' churn / stay calls."""
    labels = {r["label"] for r in results}
    probs = [r["churn_probability"] for r in results]
    spread = f"Probabilities span {min(probs) * 100:.1f}% to {max(probs) * 100:.1f}%."
    if len(labels) == 1:
        return True, f"All {len(results)} paradigms predict **{labels.pop()}**. {spread}"
    votes = ", ".join(f"{r['paradigm']}: **{r['label']}**" for r in results)
    return False, f"The paradigms disagree at the {threshold * 100:.0f}% threshold ({votes}). {spread}"


def render_paradigm_comparison(processed_df, ml_model, feature_names: list, threshold: float) -> None:
    """Scores the validated customer with the ML, DML and QML models and shows them side by side."""
    st.write("")
    st.subheader("🧠 ML vs DML vs QML: Same Customer, Three Paradigms")
    st.caption(
        "The same validated input is scored by a classical gradient-boosted model, a deep neural network "
        f"and a variational quantum circuit (simulated on CPU), all at the {threshold * 100:.0f}% decision threshold. "
        "Test metrics are from the same held-out 1,409 customers."
    )
    try:
        models, details, unavailable = paradigm_details(ml_model, feature_names)
        results = pp.predict_all_paradigms(processed_df, models, threshold=threshold)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # UI boundary: shown to the user
        st.warning(f"Could not run the paradigm comparison: {exc}")
        return

    by_paradigm = {r["paradigm"]: r for r in results}
    for column, paradigm in zip(st.columns(3), PARADIGM_STYLE):
        with column:
            if paradigm in by_paradigm:
                st.markdown(paradigm_card(by_paradigm[paradigm], details[paradigm]), unsafe_allow_html=True)
            else:
                st.warning(f"**{paradigm} · {PARADIGM_STYLE[paradigm]['title']}** unavailable: {unavailable[paradigm]}")

    if len(results) > 1:
        st.altair_chart(paradigm_probability_chart(results, threshold), use_container_width=True)
        agree, message = paradigm_agreement(results, threshold)
        (st.success if agree else st.info)(message)


def confusion_matrix_chart(cm) -> alt.LayerChart:
    """Heatmap of the confusion matrix with the count and row % written in each cell."""
    frame = pp.confusion_matrix_frame(cm)
    long_df = frame.reset_index(names="Actual").melt(id_vars="Actual", var_name="Predicted", value_name="Count")
    long_df["Actual"] = long_df["Actual"].str.replace("Actual: ", "")
    long_df["Predicted"] = long_df["Predicted"].str.replace("Pred: ", "")
    long_df["Row %"] = long_df["Count"] / long_df.groupby("Actual")["Count"].transform("sum")
    long_df["Label"] = long_df.apply(lambda r: f"{r['Count']:,}  ({r['Row %']:.0%})", axis=1)

    base = alt.Chart(long_df).encode(
        x=alt.X("Predicted:N", sort=["Stay", "Churn"], title="Predicted"),
        y=alt.Y("Actual:N", sort=["Stay", "Churn"], title="Actual"),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Actual", "Predicted", "Count", alt.Tooltip("Row %:Q", format=".1%")],
    )
    text = base.mark_text(fontSize=16, fontWeight="bold").encode(
        text="Label:N",
        # white text on the dark (high count) cells so it stays readable
        color=alt.condition(alt.datum.Count > long_df["Count"].max() / 2, alt.value("white"), alt.value("black")),
    )
    return (heat + text).properties(height=260)


def classification_report_frame(report: dict) -> pd.DataFrame:
    """sklearn's classification_report(output_dict=True) as a rounded table."""
    rows = {k: v for k, v in report.items() if isinstance(v, dict)}
    frame = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
    frame["support"] = frame["support"].astype(int)
    return frame.round(4)


def confusion_matrix_takeaways(cm) -> str:
    """Markdown bullet list explaining each confusion matrix cell."""
    cells = pp.confusion_matrix_breakdown(cm)
    tp, fn = cells["true_positives"], cells["false_negatives"]
    miss_rate = fn / (tp + fn) if (tp + fn) else 0.0
    return (
        f"- **True Positives ({tp:,}):** At-risk customers correctly targeted for retention campaigns.\n"
        f"- **False Negatives ({fn:,}):** Churners missed by the model (~{miss_rate:.1%} miss rate).\n"
        f"- **False Positives ({cells['false_positives']:,}):** Non-churning customers offered incentives "
        "(tolerable trade-off for retention).\n"
        f"- **True Negatives ({cells['true_negatives']:,}):** Loyal customers correctly identified."
    )


# ---------------------------------------------------------
# Sidebar Navigation & Model Selector
# ---------------------------------------------------------
st.sidebar.markdown("<div class='sprint-badge'>Sprint 2: Complete System Demo</div>", unsafe_allow_html=True)
st.sidebar.image(
    "https://img.icons8.com/fluency/96/combo-chart.png",
    width=56,
)
st.sidebar.title("Hackforge ML")
st.sidebar.caption("Software Engineering — Deliverable Part 2")

nav_choice = st.sidebar.radio(
    "Navigation",
    options=["Customer Churn Prediction", "Batch Customer Scoring", "Model Evaluation", "About the Project"],
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Loaded Model Info")

# Load artifacts safely
artifacts_loaded = False
load_error = None
try:
    model, feature_names, encoders = get_cached_artifacts()
    artifacts_loaded = True
    st.sidebar.success("Model: XGBoost Classifier (Tuned)")
    st.sidebar.caption(f"Feature inputs: {len(feature_names)}")
    st.sidebar.caption("Encoders: 15 categorical features")
except Exception as e:  # pylint: disable=broad-exception-caught  # UI boundary: shown to the user
    load_error = str(e)
    st.sidebar.error("Error loading artifacts")


# ---------------------------------------------------------
# SECTION A: Customer Churn Prediction
# ---------------------------------------------------------
if nav_choice == "Customer Churn Prediction":
    st.header("Predict Customer Churn")
    st.write(
        "Enter individual customer profile details below to assess risk of churn "
        "using the tuned **XGBoost Classifier** model pipeline."
    )

    if not artifacts_loaded:
        st.error(f"Cannot run predictions: {load_error}")
        st.info(
            "Please ensure `customer_churn_model_xgboost.pkl` and `encoders.pkl` "
            "are present in the project directory."
        )
    else:
        st.subheader("Interactive Persona Quick-Cards")
        st.caption("One-click demonstration profiles for quick evaluation of different risk tiers:")

        # 4 Interactive persona preset cards
        pq1, pq2, pq3, pq4 = st.columns(4)
        with pq1:
            st.markdown(
                """
                <div class="persona-card">
                    <div>
                        <div class="persona-title">🚨 High-Risk Tech</div>
                        <div class="persona-tag-high">High Risk (~82%)</div>
                        <div class="persona-desc">Senior • Mo-to-Mo • Fiber • Electronic check • 1 mo tenure</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load High-Risk", key="btn_card_high", use_container_width=True):
                load_preset("High risk: new fiber customer")
                st.session_state["preset_choice"] = "High risk: new fiber customer"
                st.rerun()

        with pq2:
            st.markdown(
                """
                <div class="persona-card">
                    <div>
                        <div class="persona-title">⚖️ Borderline At-Risk</div>
                        <div class="persona-tag-mod">Moderate Risk (~50%)</div>
                        <div class="persona-desc">DSL • Mo-to-Mo • Mailed check • 14 mos tenure • $64/mo</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load Borderline", key="btn_card_borderline", use_container_width=True):
                load_preset("Borderline: month-to-month DSL")
                st.session_state["preset_choice"] = "Borderline: month-to-month DSL"
                st.rerun()

        with pq3:
            st.markdown(
                """
                <div class="persona-card">
                    <div>
                        <div class="persona-title">🛡️ Long-Term Loyal</div>
                        <div class="persona-tag-low">Low Risk (~10%)</div>
                        <div class="persona-desc">Family • 2-Year Contract • Auto-Bank • Full Security • 48 mos</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load Loyal Family", key="btn_card_loyal", use_container_width=True):
                load_preset("Low risk: loyal family on 2-year contract")
                st.session_state["preset_choice"] = "Low risk: loyal family on 2-year contract"
                st.rerun()

        with pq4:
            st.markdown(
                """
                <div class="persona-card">
                    <div>
                        <div class="persona-title">📞 Budget Phone-Only</div>
                        <div class="persona-tag-low">Low Risk (~5%)</div>
                        <div class="persona-desc">No Internet • 2-Year Contract • Credit Card • 71 mos tenure</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load Phone-Only", key="btn_card_phone", use_container_width=True):
                load_preset("Low risk: phone-only customer")
                st.session_state["preset_choice"] = "Low risk: phone-only customer"
                st.rerun()

        st.write("")
        st.subheader("Sample Customer Preset & Form Controls")
        preset_col, button_col, clear_col = st.columns([3, 1, 1], vertical_alignment="bottom")
        with preset_col:
            chosen_preset = st.selectbox(
                "Sample customer preset",
                options=list(cp.CUSTOMER_PRESETS),
                key="preset_choice",
                help="Pick a sample profile and press 'Load preset' to fill in the form below.",
            )
        with button_col:
            st.button(
                "Load preset",
                use_container_width=True,
                on_click=load_preset,
                args=(chosen_preset,),
            )
        with clear_col:
            st.button(
                "Clear form",
                use_container_width=True,
                on_click=clear_form,
                help="Empty all fields to enter a new customer manually.",
            )
        st.caption(cp.CUSTOMER_PRESETS[chosen_preset]["description"])
        if st.session_state.get("loaded_preset"):
            st.caption(f"Form currently filled from: **{st.session_state['loaded_preset']}** (you can still edit any field).")

        with st.form("churn_prediction_form"):
            st.subheader("1. Customer Demographics")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                category_select("gender")
            with c2:
                st.selectbox(
                    pp.FIELD_LABELS["SeniorCitizen"],
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    help=FIELD_HELP["SeniorCitizen"],
                    placeholder="Select an option...",
                    key=field_key("SeniorCitizen"),
                )
            with c3:
                category_select("Partner")
            with c4:
                category_select("Dependents")

            st.subheader("2. Subscribed Services")
            s1, s2, s3 = st.columns(3)
            with s1:
                for col in ("PhoneService", "MultipleLines", "InternetService"):
                    category_select(col)
            with s2:
                for col in ("OnlineSecurity", "OnlineBackup", "DeviceProtection"):
                    category_select(col)
            with s3:
                for col in ("TechSupport", "StreamingTV", "StreamingMovies"):
                    category_select(col)

            st.subheader("3. Contract & Billing Details")
            b1, b2, b3 = st.columns(3)
            with b1:
                for col in ("Contract", "PaperlessBilling", "PaymentMethod"):
                    category_select(col)

            with b2:
                numeric_input("tenure", step=1)
                numeric_input("MonthlyCharges", step=0.5)

            with b3:
                numeric_input("TotalCharges", step=5.0)
                st.checkbox(
                    "Estimate Total Charges as Monthly Charges x Tenure",
                    key="auto_total_charges",
                    help="Tick this if you don't know the exact total. The value typed above is then ignored.",
                )
                st.caption(
                    "Note: For a new customer (tenure 0 or 1), TotalCharges is typically close to MonthlyCharges."
                )

            st.subheader("4. Decision Threshold & Intervention Settings")
            st.caption(
                "Adjust the classification decision boundary. Lowering the threshold catches more at-risk churners "
                "at the cost of additional false positives."
            )
            decision_threshold = st.slider(
                "Classification Decision Threshold",
                min_value=0.20,
                max_value=0.80,
                value=0.50,
                step=0.05,
                help="If churn probability is >= this threshold, customer is classified as Churn (Default: 0.50).",
            )

            submit_prediction = st.form_submit_button(
                "Run Churn Prediction", use_container_width=True, type="primary"
            )

        if submit_prediction:
            input_dict = current_form_values()
            if st.session_state.get("auto_total_charges"):
                estimated = estimate_total_charges(input_dict["MonthlyCharges"], input_dict["tenure"])
                if estimated is None:
                    st.warning("Fill in Monthly Charges and Tenure to estimate Total Charges.")
                else:
                    input_dict["TotalCharges"] = estimated
                    st.info(f"Total Charges estimated as **${estimated:,.2f}** (Monthly Charges x Tenure).")
            input_dict = pp.clean_customer_input(input_dict)
            is_valid, validation_errors = pp.validate_customer_input(input_dict)

            if not is_valid:
                st.error(
                    f"**Please fix {len(validation_errors)} problem(s) before running the prediction:**\n\n"
                    + "\n".join(f"- {err}" for err in validation_errors),
                    icon="🚫",
                )
            else:
                try:
                    processed_df = pp.preprocess_input(input_dict, encoders, feature_names)
                    pred_class, churn_prob, prob_dict, risk_tier = pp.predict_churn(
                        model, processed_df, threshold=decision_threshold
                    )

                    st.markdown("### Prediction Outcome")
                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        if pred_class == 1:
                            st.markdown(
                                "<div class='churn-pill-high'>⚠️ HIGH RISK — PREDICTED TO CHURN</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<div class='churn-pill-low'>✅ LOW RISK — PREDICTED TO STAY</div>",
                                unsafe_allow_html=True,
                            )

                        # Churn Probability Visual Gauge Meter
                        # pylint: disable-next=invalid-name  # Streamlit page code runs at module level
                        gauge_color = "#dc2626" if churn_prob >= 0.65 else "#d97706" if churn_prob >= 0.35 else "#16a34a"
                        gauge_pct = min(100.0, max(0.0, churn_prob * 100))
                        st.markdown(
                            f"""
                            <div style="margin-top: 12px; margin-bottom: 8px;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600; color: #64748b; margin-bottom: 4px;">
                                    <span>Low Risk (0%)</span>
                                    <span>Threshold ({decision_threshold * 100:.0f}%)</span>
                                    <span>High Risk (100%)</span>
                                </div>
                                <div class="gauge-container">
                                    <div class="gauge-fill" style="width: {gauge_pct:.1f}%; background-color: {gauge_color};"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.metric(
                            label="Predicted Churn Probability",
                            value=f"{churn_prob * 100:.1f}%",
                            delta=f"{(churn_prob - decision_threshold) * 100:+.1f}% vs threshold ({decision_threshold * 100:.0f}%)",
                            delta_color="inverse",
                        )

                        # Display 3-tier operational risk category
                        st.write(f"**Operational Risk Tier:** `{risk_tier} Risk`")
                        if risk_tier == "High":
                            st.markdown(
                                "<div class='churn-pill-high'>Critical Churn Hazard (>65%)</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                "<div class='recommendation-box'>"
                                "<strong>Recommended Action:</strong> Immediate outreach by Senior Account Executive. "
                                "Offer customized long-term contract discounts and priority support."
                                "</div>",
                                unsafe_allow_html=True,
                            )
                        elif risk_tier == "Moderate":
                            st.markdown(
                                "<div class='churn-pill-moderate'>Moderate Churn Risk (35%–65%)</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                "<div class='recommendation-box'>"
                                "<strong>Recommended Action:</strong> Automated customer success check-in. "
                                "Send targeted engagement emails or review bundle service utilization."
                                "</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<div class='churn-pill-low'>Low Churn Risk (<35%)</div>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                "<div class='recommendation-box'>"
                                "<strong>Recommended Action:</strong> Standard relationship maintenance. "
                                "Account is stable; candidate for loyalty rewards or subtle cross-sell."
                                "</div>",
                                unsafe_allow_html=True,
                            )

                    with res_col2:
                        st.write("**Probability Distribution:**")
                        prob_df = pd.DataFrame(
                            {
                                "Class": ["Stay (Loyal)", "Churn (Leaves)"],
                                "Probability": [prob_dict["Stay"], prob_dict["Churn"]],
                            }
                        )
                        st.bar_chart(prob_df.set_index("Class"), height=200)

                    render_paradigm_comparison(processed_df, model, feature_names, decision_threshold)

                    # Explainable Risk Attribution & Key Churn Drivers
                    st.write("")
                    st.subheader("🔍 Explainable Risk Attribution & Key Drivers")
                    attribution = explain_customer_risk(input_dict, churn_prob)
                    attr_col1, attr_col2 = st.columns(2)
                    with attr_col1:
                        st.markdown("**🚨 Primary Risk Accelerators:**")
                        if attribution["risk_drivers"]:
                            for driver in attribution["risk_drivers"]:
                                st.markdown(
                                    f"""
                                    <div class="driver-chip-danger">
                                        <strong>{driver['factor']}:</strong> {driver['detail']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("No significant risk accelerators detected for this profile.")

                    with attr_col2:
                        st.markdown("**🛡️ Primary Retention Anchors:**")
                        if attribution["protective_factors"]:
                            for anchor in attribution["protective_factors"]:
                                st.markdown(
                                    f"""
                                    <div class="driver-chip-success">
                                        <strong>{anchor['factor']}:</strong> {anchor['detail']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("No strong retention anchors identified; customer is uncommitted.")

                    # -------------------------------------------------
                    # SPRINT 2: "What-If" Retention Strategy Simulator
                    # -------------------------------------------------
                    st.write("")
                    st.subheader("🧪 'What-If' Retention Strategy Simulator")
                    st.caption(
                        "Test counterfactual interventions in real-time. "
                        "Simulate contract extensions, security/support bundles, or billing discounts "
                        "to see immediate impact on predicted churn probability and retained revenue."
                    )

                    with st.container():
                        sim_c1, sim_c2, sim_c3 = st.columns(3)
                        with sim_c1:
                            sim_contract = st.selectbox(
                                "Simulate Contract Upgrade",
                                options=["Keep Current", "One year", "Two year"],
                                index=1 if input_dict.get("Contract") == "Month-to-month" else 0,
                                help="Long-term contracts are the single most powerful driver of customer retention (39.1% global feature importance).",
                            )
                        with sim_c2:
                            has_internet = input_dict.get("InternetService") != "No"
                            sim_support = st.selectbox(
                                "Simulate Support & Security Bundle",
                                options=["Keep Current", "Add Tech Support & Security", "Remove Add-ons"],
                                index=1 if (has_internet and input_dict.get("TechSupport") == "No") else 0,
                                help="Adds Online Security and Tech Support add-ons to enhance service stickiness.",
                            )
                        with sim_c3:
                            sim_discount = st.slider(
                                "Simulate Monthly Promotional Discount ($)",
                                min_value=0.0,
                                max_value=30.0,
                                value=10.0 if pred_class == 1 else 0.0,
                                step=2.5,
                                help="Simulate reducing monthly charges to test price elasticity.",
                            )

                        # Build counterfactual record
                        cf_dict = dict(input_dict)
                        if sim_contract != "Keep Current":
                            cf_dict["Contract"] = sim_contract
                        if sim_support == "Add Tech Support & Security" and has_internet:
                            cf_dict["OnlineSecurity"] = "Yes"
                            cf_dict["TechSupport"] = "Yes"
                        elif sim_support == "Remove Add-ons" and has_internet:
                            cf_dict["OnlineSecurity"] = "No"
                            cf_dict["TechSupport"] = "No"

                        if sim_discount > 0:
                            orig_monthly = float(input_dict.get("MonthlyCharges", 0))
                            cf_dict["MonthlyCharges"] = max(18.25, round(orig_monthly - sim_discount, 2))
                            orig_tenure = max(1, int(input_dict.get("tenure", 1)))
                            cf_dict["TotalCharges"] = round(cf_dict["MonthlyCharges"] * orig_tenure, 2)

                        try:
                            cf_processed = pp.preprocess_input(cf_dict, encoders, feature_names)
                            cf_pred, cf_prob, cf_prob_dict, cf_tier = pp.predict_churn(
                                model, cf_processed, threshold=decision_threshold
                            )

                            risk_delta = cf_prob - churn_prob
                            risk_reduction_pct = (churn_prob - cf_prob) / churn_prob * 100 if churn_prob > 0 else 0
                            orig_annual_rev = float(input_dict.get("MonthlyCharges", 0)) * 12
                            sim_annual_rev = float(cf_dict.get("MonthlyCharges", 0)) * 12

                            st.markdown("##### 📊 Counterfactual Simulation Outcome")
                            sc1, sc2, sc3, sc4 = st.columns(4)
                            sc1.metric(
                                "Simulated Churn Risk",
                                f"{cf_prob * 100:.1f}%",
                                delta=f"{risk_delta * 100:+.1f}% vs Baseline",
                                delta_color="inverse",
                            )
                            sc2.metric(
                                "Relative Risk Drop",
                                f"{risk_reduction_pct:.1f}%",
                                delta=f"-{abs(risk_delta)*100:.1f}% Abs Drop" if risk_delta < 0 else None,
                            )
                            sc3.metric(
                                "Simulated Risk Tier",
                                f"{cf_tier} Risk",
                                delta="Retention Safety" if (pred_class == 1 and cf_pred == 0) else None,
                            )
                            sc4.metric(
                                "Annual Recurring Revenue",
                                f"${sim_annual_rev:,.0f} / yr",
                                delta=f"-${(orig_annual_rev - sim_annual_rev):,.0f} Discount" if sim_discount > 0 else "Full Retention Value",
                            )

                            if pred_class == 1 and cf_pred == 0:
                                st.success(
                                    f"🎉 **Retention Strategy Successful!** Applying these intervention levers drops the customer's churn risk "
                                    f"from **{churn_prob * 100:.1f}%** down to **{cf_prob * 100:.1f}%** ({risk_reduction_pct:.1f}% risk reduction). "
                                    f"The account is successfully converted from **High Risk** to **{cf_tier} Risk**, preserving **${sim_annual_rev:,.2f}/yr** in recurring revenue!",
                                    icon="✅",
                                )
                            elif risk_delta < 0:
                                st.info(
                                    f"📉 **Risk Reduced:** Churn probability reduced from **{churn_prob * 100:.1f}%** to **{cf_prob * 100:.1f}%** "
                                    f"({risk_reduction_pct:.1f}% reduction). Consider coupling with automated payment billing for greater retention.",
                                    icon="💡",
                                )
                            else:
                                st.caption("No retention levers applied yet. Adjust the options above to simulate interventions.")
                        except Exception as e:  # pylint: disable=broad-exception-caught  # UI boundary: shown to the user
                            st.warning(f"Could not compute counterfactual simulation: {e}")

                    st.markdown(
                        "<div class='alert-disclaimer'>"
                        "<strong>Notice:</strong> Machine learning predictions represent statistical estimates "
                        "trained on historical customer profiles. Risk categories (<35% Low, 35%–65% Moderate, >65% High) "
                        "are operational business rules for intervention prioritization, not statistically validated cutoffs. "
                        "Notice that <code>customerID</code> was excluded from all feature sets."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    with st.expander("Customer profile used for this prediction"):
                        st.dataframe(profile_summary(input_dict), use_container_width=True, hide_index=True)

                except ValueError as ex:
                    # raised by the preprocess_input boundary guards
                    st.error(f"Input could not be processed: {ex}", icon="🚫")
                except Exception as ex:  # pylint: disable=broad-exception-caught  # UI boundary: shown to the user
                    st.error(f"Inference error: {ex}")


# ---------------------------------------------------------
# SECTION B: Batch Customer Scoring (Sprint 2 Deliverable)
# ---------------------------------------------------------
elif nav_choice == "Batch Customer Scoring":
    st.header("Batch Customer Scoring & Retention Export")
    st.write(
        "Evaluate customer portfolios in bulk using the tuned **XGBoost Classifier** inference engine. "
        "Upload customer CSV records or evaluate holdout evaluation samples to generate prioritized retention campaign targets."
    )

    if not artifacts_loaded:
        st.error(f"Cannot run batch scoring: {load_error}")
    else:
        st.subheader("1. Data Source Selection")
        data_source = st.radio(
            "Select Input Source",
            options=["Sample Holdout Evaluation Batch (100 Telco Customers)", "Upload Custom Customer CSV"],
            horizontal=True,
        )

        batch_df = None
        if data_source == "Sample Holdout Evaluation Batch (100 Telco Customers)":
            try:
                raw_df = pp.load_dataset()
                batch_df = raw_df.sample(n=min(100, len(raw_df)), random_state=42).copy()
                st.caption(f"Loaded {len(batch_df)} verified customer accounts from holdout dataset.")
            except Exception as e:  # pylint: disable=broad-exception-caught  # UI boundary: shown to the user
                st.error(f"Could not load sample dataset: {e}")
        else:
            uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
            if uploaded_file is not None:
                try:
                    batch_df = pd.read_csv(uploaded_file)
                    st.success(f"Uploaded CSV with {len(batch_df)} records.")
                except Exception as e:  # pylint: disable=broad-exception-caught  # UI boundary: shown to the user
                    st.error(f"Error parsing uploaded CSV: {e}")

        if batch_df is not None:
            st.write("")
            st.subheader("2. Batch Scoring Configuration")
            b_thresh_col, b_btn_col = st.columns([3, 1], vertical_alignment="bottom")
            with b_thresh_col:
                batch_threshold = st.slider(
                    "Batch Decision Threshold (P(Churn) >= Threshold -> Churn)",
                    min_value=0.20,
                    max_value=0.80,
                    value=0.50,
                    step=0.05,
                    key="batch_scoring_threshold",
                )
            with b_btn_col:
                run_batch = st.button("Run Batch Scoring", type="primary", use_container_width=True)

            if run_batch or "last_batch_results" in st.session_state:
                if run_batch:
                    with st.spinner("Executing vectorized inference across customer portfolio..."):
                        try:
                            scored_results = pp.score_customer_batch(
                                batch_df, model, encoders, feature_names, threshold=batch_threshold
                            )
                            st.session_state["last_batch_results"] = scored_results
                        except Exception as e:  # pylint: disable=broad-exception-caught  # UI boundary: shown to the user
                            st.error(f"Batch inference failed: {e}")
                            st.session_state["last_batch_results"] = None

                scored_df = st.session_state.get("last_batch_results")
                if scored_df is not None and not scored_df.empty:
                    st.write("")
                    st.subheader("3. Portfolio Risk Summary & Prioritization")
                    total_customers = len(scored_df)
                    high_risk_count = int((scored_df["Risk_Tier"] == "High").sum())
                    mod_risk_count = int((scored_df["Risk_Tier"] == "Moderate").sum())
                    low_risk_count = int((scored_df["Risk_Tier"] == "Low").sum())
                    pred_churners = int((scored_df["Predicted_Class"] == "Churn").sum())

                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Scored", f"{total_customers:,}")
                    k2.metric("Critical High Risk (🔴)", f"{high_risk_count} ({high_risk_count/total_customers*100:.1f}%)")
                    k3.metric("Moderate Risk (🟡)", f"{mod_risk_count} ({mod_risk_count/total_customers*100:.1f}%)")
                    k4.metric("Low Risk / Loyal (🟢)", f"{low_risk_count} ({low_risk_count/total_customers*100:.1f}%)")

                    # Risk distribution chart
                    dist_col, action_col = st.columns([1, 1])
                    with dist_col:
                        st.write("**Risk Tier Distribution:**")
                        risk_summary = pd.DataFrame({
                            "Risk Tier": ["High Risk (>65%)", "Moderate Risk (35-65%)", "Low Risk (<35%)"],
                            "Customers": [high_risk_count, mod_risk_count, low_risk_count],
                        })
                        chart = alt.Chart(risk_summary).mark_bar(cornerRadius=6).encode(
                            x=alt.X("Risk Tier:N", sort=None, title=None),
                            y=alt.Y("Customers:Q", title="Account Count"),
                            color=alt.Color("Risk Tier:N", scale=alt.Scale(
                                domain=["High Risk (>65%)", "Moderate Risk (35-65%)", "Low Risk (<35%)"],
                                range=["#ef4444", "#f59e0b", "#10b981"]
                            ), legend=None)
                        ).properties(height=220)
                        st.altair_chart(chart, use_container_width=True)

                    with action_col:
                        st.write("**Executive Action Plan:**")
                        st.markdown(
                            f"""
                            - **Immediate Intervention Target:** `{high_risk_count}` accounts flagged as Critical Churn Hazard. Assign to Senior Account Execs for discount outreach.
                            - **Automated Check-in Pool:** `{mod_risk_count}` accounts displaying moderate drift. Enroll in automated re-engagement workflows.
                            - **Stable Base:** `{low_risk_count}` loyal accounts identified for cross-sell and loyalty rewards.
                            """
                        )
                        # Export Button
                        csv_data = scored_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Priority Retention Outreach CSV",
                            data=csv_data,
                            file_name="prioritized_retention_campaign_targets.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True,
                        )

                    st.write("")
                    st.write("**Customer Risk Evaluation Table (Sorted by Highest Churn Risk):**")
                    display_cols = ["customerID", "Churn_Probability", "Predicted_Class", "Risk_Tier", "Contract", "tenure", "MonthlyCharges", "PaymentMethod"]
                    if "Actual_Churn" in scored_df.columns:
                        display_cols.append("Actual_Churn")
                    available_cols = [c for c in display_cols if c in scored_df.columns]
                    st.dataframe(scored_df[available_cols], use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# SECTION C: Model Evaluation
# ---------------------------------------------------------
elif nav_choice == "Model Evaluation":
    st.header("Model Evaluation & Performance Benchmarks")
    st.write(
        "This section reviews the verified evaluation metrics of the serialized XGBoost classifier. "
        "Metrics reflect a stratified 20% holdout test split (`random_state=42`, 1,409 customers) "
        "with class weighting applied (`scale_pos_weight ≈ 2.77`)."
    )

    tab_saved, tab_live = st.tabs(["Saved Verified Metrics", "Run Live Verification"])

    with tab_saved:
        saved = pp.SAVED_BENCHMARK_METRICS
        st.subheader(f"Holdout Test Set Performance (N = {saved['test_samples']:,})")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{saved['test_accuracy']:.1%}", help="Overall correct predictions across both classes")
        m2.metric("Recall (Churn)", f"{saved['test_recall']:.1%}", help="Share of actual churners the model catches")
        m3.metric("Precision (Churn)", f"{saved['test_precision']:.1%}", help="Proportion of predicted churners who actually left")
        m4.metric("F1-Score (Churn)", f"{saved['test_f1']:.3f}", help="Harmonic mean of precision and recall")

        m5, m6, _, _ = st.columns(4)
        m5.metric("Specificity (Stay)", f"{saved['test_specificity']:.1%}", help="Share of loyal customers correctly predicted to stay")
        m6.metric("ROC-AUC", f"{saved['test_roc_auc']:.3f}", help="How well the churn probability ranks churners above loyal customers (0.5 = random)")

        st.divider()

        c_mat_col, c_summary_col = st.columns([1, 1])

        with c_mat_col:
            st.write("**Confusion Matrix:**")
            st.altair_chart(confusion_matrix_chart(saved["confusion_matrix"]), use_container_width=True)
            st.caption("Each cell shows the number of customers and the % of that actual class.")

        with c_summary_col:
            st.write("**Key Takeaways:**")
            st.markdown(confusion_matrix_takeaways(saved["confusion_matrix"]))

    with tab_live:
        st.subheader("Live Benchmark Runner")
        st.write(
            "Execute a live evaluation of the saved model pipeline against the local dataset "
            "(`WA_Fn-UseC_-Telco-Customer-Churn.csv`) without retraining."
        )

        threshold = st.slider(
            "Decision threshold (predict churn when P(churn) >= threshold)",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            help="0.50 is the model default. Lower values catch more churners (higher recall) "
            "but flag more loyal customers (lower precision).",
        )

        if st.button("Run Test Evaluation Against Dataset", type="primary"):
            st.session_state["live_eval_threshold"] = threshold

        # results are kept in session state so they don't disappear on the next rerun
        if "live_eval_threshold" in st.session_state:
            run_threshold = st.session_state["live_eval_threshold"]
            try:
                with st.spinner("Loading dataset, performing holdout split and computing metrics..."):
                    results = run_live_evaluation(run_threshold)
                is_default_threshold = abs(run_threshold - 0.5) < 1e-9
                all_match, comparison = pp.compare_with_benchmark(results)
                if not is_default_threshold:
                    st.info(
                        f"Evaluated with a custom threshold of {results['threshold']:.2f}. The saved benchmark "
                        "was measured at 0.50, so the comparison table below is expected to differ."
                    )
                elif all_match:
                    st.success("Evaluation complete! Live results match the verified benchmark.")
                else:
                    st.warning(
                        "Evaluation complete, but the live results do NOT match the saved benchmark. "
                        "The model file may have been retrained or replaced. See the comparison below."
                    )

                st.caption(
                    f"Held-out test set: {results['total_test_samples']:,} customers "
                    f"({results['test_churn_count']:,} churned, {results['test_stay_count']:,} stayed)."
                )
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
                e2.metric("Recall (Churn)", f"{results['recall'] * 100:.2f}%")
                e3.metric("Precision (Churn)", f"{results['precision'] * 100:.2f}%")
                e4.metric("F1-Score (Churn)", f"{results['f1']:.4f}")
                e5, e6, _, _ = st.columns(4)
                e5.metric("Specificity (Stay)", f"{results['specificity'] * 100:.2f}%")
                if "roc_auc" in results:
                    e6.metric("ROC-AUC", f"{results['roc_auc']:.4f}")

                live_cm_col, live_takeaway_col = st.columns([1, 1])
                with live_cm_col:
                    st.write("**Live Confusion Matrix:**")
                    st.altair_chart(confusion_matrix_chart(results["confusion_matrix"]), use_container_width=True)
                with live_takeaway_col:
                    st.write("**Key Takeaways:**")
                    st.markdown(confusion_matrix_takeaways(results["confusion_matrix"]))

                st.write("**Comparison with Saved Benchmark:**")
                st.dataframe(comparison, use_container_width=True, hide_index=True)

                st.write("**Classification Report:**")
                st.dataframe(classification_report_frame(results["classification_report"]), use_container_width=True)
            except Exception as ex:  # pylint: disable=broad-exception-caught  # UI boundary: shown to the user
                st.error(f"Live evaluation failed: {ex}")


# ---------------------------------------------------------
# SECTION C: About the Project
# ---------------------------------------------------------
elif nav_choice == "About the Project":
    st.header("About Hackforge — Customer Churn Prediction")
    st.write(
        "Software Engineering Mini-Project — Deliverable Part 2. "
        "A complete end-to-end Machine Learning pipeline and interactive decision support application."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Business Objective")
        st.markdown(
            """
            Customer churn is one of the most critical metrics in telecommunications and subscription services.
            Acquiring a new customer costs 5 to 7 times more than retaining an existing one.

            This application empowers customer success teams to:
            1. **Detect At-Risk Customers:** Identify churn indicators early in the customer lifecycle.
            2. **Target Interventions:** Offer tailored discounts, contract upgrades, or tech support.
            3. **Optimize Retention Budgets:** Focus marketing resources where churn probability is highest.
            """
        )

        st.subheader("Technical Architecture")
        st.markdown(
            """
            - **Model:** `XGBClassifier` (Extreme Gradient Boosting) optimized via `GridSearchCV` & `StratifiedKFold`.
            - **Data Handling:** Preserves 100% of samples via class weighting (`scale_pos_weight = 2.77`) instead of destructive downsampling.
            - **Feature Count:** 19 curated customer attributes (demographics, services, contract, billing).
            - **UI Framework:** Streamlit 1.x with responsive layout and caching.
            """
        )

    with col2:
        st.subheader("Software Engineering Deliverables")
        st.markdown(
            """
            | Component | Description |
            | :--- | :--- |
            | **Phase 1** | Model Training & Artifact Serialization (`01_churn_prediction_xgboost.py`) |
            | **Phase 2** | Interactive Streamlit Application (`app.py`) |
            | **Phase 3** | Reliable Modular Pipeline (`prediction_pipeline.py`) |
            | **Phase 4** | Automated Test Suite with `pytest` (`test_pipeline.py`, `test_app.py`) |
            | **Phase 5** | Continuous Integration CI Workflow via GitHub Actions (`.github/workflows/ci.yml`) |
            """
        )

        st.subheader("Limitations & Known Boundaries")
        st.markdown(
            """
            - **Categorical Encoders:** Uses LabelEncoding matched to initial categories; novel values must be mapped or flagged.
            - **No Database Dependency:** Operates as a stateless inference app without external database dependencies.
            - **Single Tenant Demo:** Designed for internal decision support and academic demonstration.
            """
        )
