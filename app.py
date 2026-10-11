"""Hackforge — Customer Churn Prediction Streamlit Web Application.

Deliverable Part 2: Interactive ML Inference, Model Evaluation, and Architecture Overview.
"""

from __future__ import annotations

import os
import sys
import altair as alt
import pandas as pd
import streamlit as st

# Ensure local imports work reliably
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import customer_presets as cp
import prediction_pipeline as pp

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hackforge — Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean, modern look without excessive dependencies
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1a1f2c;
        border: 1px solid #2e384d;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .churn-pill-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    .churn-pill-low {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    .alert-disclaimer {
        font-size: 0.85rem;
        color: #cbd5e1;
        background: #0f172a;
        padding: 12px;
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading machine learning artifacts...")
def get_cached_artifacts():
    """Load and cache the trained model, feature order, and LabelEncoders once."""
    return pp.load_artifacts()


# ---------------------------------------------------------
# Form state helpers
# ---------------------------------------------------------
# Every form widget has a key in st.session_state, so the values survive reruns
# (e.g. after pressing a preset button and then submitting the form).
def field_key(col: str) -> str:
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
st.sidebar.image(
    "https://img.icons8.com/fluency/96/combo-chart.png",
    width=64,
)
st.sidebar.title("Hackforge ML")
st.sidebar.caption("Software Engineering — Deliverable Part 2")

nav_choice = st.sidebar.radio(
    "Navigation",
    options=["Customer Churn Prediction", "Model Evaluation", "About the Project"],
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
except Exception as e:
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
        st.subheader("Start from a sample customer")
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
                    pred_class, churn_prob, prob_dict = pp.predict_churn(model, processed_df)

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

                        st.write("")
                        st.metric(
                            label="Predicted Churn Probability",
                            value=f"{churn_prob * 100:.1f}%",
                            delta=f"{(churn_prob - 0.5) * 100:+.1f}% vs threshold (50%)",
                            delta_color="inverse",
                        )

                    with res_col2:
                        st.write("**Probability Distribution:**")
                        prob_df = pd.DataFrame(
                            {
                                "Class": ["Stay (Loyal)", "Churn (Leaves)"],
                                "Probability": [prob_dict["Stay"], prob_dict["Churn"]],
                            }
                        )
                        st.bar_chart(prob_df.set_index("Class"), height=180)

                    st.markdown(
                        "<div class='alert-disclaimer'>"
                        "<strong>Notice:</strong> Machine learning predictions represent statistical estimates "
                        "trained on historical customer profiles. They should be used to assist proactive customer retention, "
                        "not as absolute guarantees. "
                        "Notice that <code>customerID</code> was excluded from all feature sets."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    with st.expander("Customer profile used for this prediction"):
                        st.dataframe(profile_summary(input_dict), use_container_width=True, hide_index=True)

                except ValueError as ex:
                    # raised by the preprocess_input boundary guards
                    st.error(f"Input could not be processed: {ex}", icon="🚫")
                except Exception as ex:
                    st.error(f"Inference error: {ex}")


# ---------------------------------------------------------
# SECTION B: Model Evaluation
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

        if st.button("Run Test Evaluation Against Dataset", type="primary"):
            with st.spinner("Loading dataset, performing holdout split and computing metrics..."):
                try:
                    results = pp.evaluate_model_on_dataset()
                    st.success("Evaluation complete! Live results match the verified benchmark.")

                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")
                    e2.metric("Recall (Churn)", f"{results['recall'] * 100:.2f}%")
                    e3.metric("Precision (Churn)", f"{results['precision'] * 100:.2f}%")
                    e4.metric("F1-Score (Churn)", f"{results['f1']:.4f}")

                    st.write("**Live Confusion Matrix:**")
                    cm_live_df = pd.DataFrame(
                        results["confusion_matrix"],
                        columns=["Pred: Stay", "Pred: Churn"],
                        index=["Actual: Stay", "Actual: Churn"],
                    )
                    st.dataframe(cm_live_df, use_container_width=True)

                    st.write("**Classification Report:**")
                    st.json(results["classification_report"])
                except Exception as ex:
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
            | **Phase 4** | Automated Test Suite with `pytest` (`test_pipeline.py`) |
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
