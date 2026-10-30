# Developer Guide: Customer Churn Prediction

This guide explains the codebase structure, the rationale behind technical decisions, and how to extend or maintain the project.

## 1. Project Architecture

The project is organised in layers (see `docs/SAD_v1.1_Addendum.md` for the diagrams):

| Layer | Files | Role |
| :--- | :--- | :--- |
| Presentation | `app.py`, `customer_presets.py` | Streamlit web application and sample customers |
| Service | `prediction_pipeline.py` | Validation, preprocessing, prediction, evaluation, batch scoring, explanation, paradigm model loading |
| Models | `classical_models.py`, `dml_model.py`, `qml_model.py` | Classical (XGBoost, Logistic Regression, Random Forest), deep learning (PyTorch MLP) and quantum (PennyLane VQC) models |
| Training | `training_common.py`, `01_churn_prediction_xgboost.py`, `02_paradigm_benchmark.py` | Shared data split, metrics, plots and MLflow logging; classical training; ML vs DML vs QML benchmark |
| Configuration | `params.yaml`, `dvc.yaml`, `pyproject.toml` | Pipeline settings, DVC stages, pylint and coverage settings |

The machine learning workflow is unchanged from the first release:
1.  **Data Ingestion:** Reads the CSV data (`prediction_pipeline.load_dataset`).
2.  **Preprocessing:** Cleans and label-encodes the data (`training_common.prepare_data`).
3.  **Model Training:** XGBoost with stratified cross-validated grid search (`classical_models.train_xgboost`).
4.  **Evaluation:** Metrics, plots and an MLflow run for every model (`training_common.evaluate_and_log_run`).
5.  **Serialization:** Saves the model and encoders together.

## 2. Key Decisions & Rationale

### Why XGBoost?
We migrated from Random Forest to **XGBoost** because:
-   **Better Generalization:** XGBoost uses boosting (correcting previous errors) rather than bagging (averaging independent trees), often leading to higher accuracy.
-   **Regularization:** Built-in parameters (`lambda`, `alpha`) help prevents overfitting, which was a major issue in previous iterations.

### Handling Imbalanced Data
The dataset contains roughly **73% Non-Churners** and **27% Churners**.
-   **Old Approach (SMOTE):** Created synthetic (fake) data. Resulted in overfitting (97% Train / 78% Test).
-   **Intermediate Approach (Downsampling):** Deleted majority class data. Resulted in data loss and lower accuracy (74%).
-   **Current Approach (Class Weights):** We use `scale_pos_weight` calculated as `Majority Count / Minority Count` (~2.77).
    -   *Why:* This forces the model to treat every missed churner as **2.77x more important** than a missed non-churner. It improves Recall without discarding real data.

### Preventing Overfitting
We explicitly limit the model's complexity to ensure it learns patterns, not noise:
-   **`max_depth=5`**: Restricts trees to 5 levels deep.
-   **`subsample=0.8`**: Trains each tree on only 80% of data (adds randomness).
-   **`colsample_bytree=0.8`**: Uses only 80% of features per tree.

## 3. Code Walkthrough

### Dependencies
Install from the `requirements*.txt` files described in the README. Each file includes the previous one:
-   `requirements.txt`: pandas, numpy, scikit-learn, XGBoost, Streamlit, Altair (the application).
-   `requirements-mlops.txt`: MLflow, DVC, PyYAML, matplotlib, seaborn (training and tracking).
-   `requirements-dev.txt`: pylint, pytest-cov (quality checks).
-   `requirements-research.txt`: PyTorch, PennyLane (deep and quantum models).

### Shared training code (`training_common.py`)
-   **`load_config`**: reads `params.yaml` over built-in defaults, so a missing or broken file still gives a working run.
-   **`prepare_data`**: cleans the CSV with the same rules as the application, label-encodes the text columns and makes the stratified split (`test_size=0.2`, `random_state=42`). Every model in both scripts uses this one split.
    -   *Crucial:* The encoders are saved to `encoders.pkl` with the model. You **MUST** use these same encoders when predicting on new data so that, for example, "Yes" is always mapped to the same number.
-   **`evaluate_and_log_run`**: scores a trained model on the held-out split, writes its plots and logs parameters, metrics and the model to MLflow.

### Classical training (`01_churn_prediction_xgboost.py`, `classical_models.py`)
-   `TRAINERS` lists the model families; `params.yaml` `model.type` chooses which to train.
-   XGBoost is tuned with `GridSearchCV` and 3-fold `StratifiedKFold` over `train.param_grid`, with `scale_pos_weight` set from the training data.
-   The best model by `model.champion_metric` (ROC-AUC) is exported with its encoders, together with `metrics.json`, `classification_report.json` and the plots tracked by DVC.

### Deep and quantum models (`dml_model.py`, `qml_model.py`)
-   `TorchTabularClassifier` gives PyTorch models the scikit-learn interface (`fit`, `predict_proba`, `predict`). It handles preprocessing, class-weighted loss, Adam, early stopping on a validation slice, and pickling.
-   `ChurnMLPClassifier` builds the deep network: Linear, BatchNorm, ReLU and Dropout per hidden layer.
-   `VariationalQuantumClassifier` reduces the inputs to one feature per qubit with PCA, scales them to rotation angles, and builds a PennyLane circuit (angle embedding re-uploaded before each entangling layer, Pauli-Z measurements, linear output).
-   On macOS, PyTorch and XGBoost ship separate OpenMP runtimes. `dml_model.py` imports XGBoost before PyTorch and runs PyTorch single-threaded (`single_threaded_torch`) so both can be used in one process.

### Paradigm benchmark (`02_paradigm_benchmark.py`)
-   `build_plan` reads `paradigms.include` from `params.yaml` and returns one trainer per paradigm.
-   Each model is evaluated with the same code, tagged `ML`, `DML` or `QML` in MLflow, and the DML and QML models are saved to `models/` for the application.
-   It never writes the production model.

### Tests and traceability
-   Test Plan test cases, the requirements they verify and the tests that run them are listed in `traceability.py`. `conftest.py` tags each test with its test case ID.
-   `test_system_validation.py` runs the training pipeline and the benchmark end to end with every output redirected to a temporary folder.
-   `python tools/generate_test_reports.py` runs the suite with coverage and pylint and writes the reports in `docs/test-reports/`.
-   When you add a feature, add or extend a test case in `traceability.py` and tag the new tests with `@pytest.mark.tc("TC-xx")`.

## 4. Deployment & Usage

### Files Required for Production
To use this model in an application (e.g., a web service), you need two artifacts:
1.  `customer_churn_model_xgboost.pkl`: The trained XGBoost model.
2.  `encoders.pkl`: The dictionary of LabelEncoders.

The deep learning and quantum models used by the application are in `models/dml_model.pkl` and `models/qml_model.pkl` (written by `02_paradigm_benchmark.py`).

### Example Prediction Code
```python
import customer_presets as cp
import prediction_pipeline as pp

# 1. Load the model, its feature order and its encoders (always as one pair)
model, features, encoders = pp.load_artifacts()

# 2. A customer record with all 19 fields (here: a sample customer)
customer = cp.get_preset_values("High risk: new fiber customer")

# 3. Encode in training column order and predict
X = pp.preprocess_input(customer, encoders, features)
label, churn_prob, _, risk_tier = pp.predict_churn(model, X, threshold=0.5)
print("Churn" if label == 1 else "Stay", f"{churn_prob:.1%}", risk_tier)

# 4. The same customer scored by all three paradigms
models = {"ML": model}
for paradigm, path in pp.PARADIGM_MODEL_FILES.items():
    models[paradigm] = pp.load_paradigm_model(path, features)["model"]
for result in pp.predict_all_paradigms(X, models):
    print(result["paradigm"], f"{result['churn_probability']:.1%}", result["label"], result["risk_tier"])
```

## 5. Input Validation & Boundary Guards

Every customer record goes through three steps in `prediction_pipeline.py` before it reaches the model:

1.  **`clean_customer_input(data)`**: strips whitespace from text, turns numeric strings into numbers and drops `customerID`. Values it can't convert are left alone so step 2 can report them.
2.  **`validate_customer_input(data)`**: returns `(is_valid, errors)`. It never raises, so the UI can show every problem at once.
3.  **`preprocess_input(...)`**: has its own guards and raises a `ValueError` with a readable message if bad data skipped step 2 (e.g. batch scoring from a notebook).

### Validation rules

| Rule | Details |
| :--- | :--- |
| Required fields | All 19 features must be present. `None`, `""`, whitespace-only and `NaN` count as missing. |
| Categories | Must be a string from `VALID_CATEGORIES` (the classes in `encoders.pkl`). The error lists the allowed values. |
| Numeric types | `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen` must be finite numbers. Booleans, `inf` and text like `"abc"` are rejected. |
| Ranges (`NUMERIC_RANGES`) | `tenure` 0–120 (whole months), `MonthlyCharges` $0–$500, `TotalCharges` $0–$50,000, `SeniorCitizen` 0 or 1. Bounds are inclusive. |
| Billing consistency | `TotalCharges` must be within 0.5x–2.0x of `MonthlyCharges × tenure` (`BILLING_RATIO_RANGE`). With tenure 0 it cannot exceed `MonthlyCharges`. Every row of the Telco CSV falls between 0.69x and 1.57x. |
| Phone services | `PhoneService = "No"` → `MultipleLines` must be `"No phone service"` (or `"No"`); `PhoneService = "Yes"` → `MultipleLines` cannot be `"No phone service"`. |
| Internet add-ons | `InternetService = "No"` → the 6 add-ons in `INTERNET_DEPENDENT_COLUMNS` must be `"No internet service"` (or `"No"`); with DSL / Fiber optic they cannot be `"No internet service"`. |

### Example

```python
import prediction_pipeline as pp

record = pp.clean_customer_input(raw_record)
ok, errors = pp.validate_customer_input(record)
if not ok:
    for err in errors:
        print("-", err)
else:
    model, features, encoders = pp.load_artifacts()
    X = pp.preprocess_input(record, encoders, features)
    label, churn_prob, _, risk_tier = pp.predict_churn(model, X)
```

If the model is ever retrained with new category values, update `VALID_CATEGORIES`. The test `test_valid_categories_match_fitted_encoders` fails until the two match.

## 6. Future Improvements
1.  **Feature Engineering:** Create new features like `TenureYears` (binning months) or `HasStreamingService` (combining TV/Movies).
2.  **Ensembling:** The three paradigms make different mistakes; averaging their probabilities could improve recall on borderline customers.
3.  **Quantum hardware:** Run the variational quantum classifier on a real quantum device through PennyLane.
4.  **Monitoring:** Track input drift and schedule retraining through the DVC pipeline.
