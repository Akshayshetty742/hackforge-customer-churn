# Customer Churn Prediction (Team Hackforge)

A machine learning decision support system that predicts which telecom customers are likely to leave, explains why, and helps a retention team decide what to do about it. The project also compares three machine learning paradigms on the same problem: classical machine learning (XGBoost), deep learning (a PyTorch neural network) and quantum machine learning (a PennyLane variational quantum classifier).

Software Engineering Mini Project, PES University.

| Team member | SRN |
|---|---|
| Akshaya B | PES1UG24AM026 |
| Ashwin C H | PES1UG24AM900 |
| Amar Patil | PES1UG24AM031 |

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Model Performance](#model-performance)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation Instructions](#installation-instructions)
- [How to Launch the Streamlit Application](#how-to-launch-the-streamlit-application)
- [How to Run the Training Pipeline and Benchmark](#how-to-run-the-training-pipeline-and-benchmark)
- [How to Run Automated Tests and Generate Test Reports](#how-to-run-automated-tests-and-generate-test-reports)
- [Repository Structure](#repository-structure)
- [Documentation](#documentation)
- [CI/CD Automation](#cicd-automation)
- [Limitations and Known Boundaries](#limitations-and-known-boundaries)
- [Deliverable Sprints](#deliverable-sprints)
- [Sprint Demonstration Videos](#sprint-demonstration-videos)

---

## Project Overview

Customer retention is central to subscription businesses: winning a new customer typically costs five to seven times more than keeping an existing one. This system analyses a customer's demographics, subscribed services, contract and billing details, and returns a churn probability in real time, so account managers can act before the customer cancels.

The model is trained on the public Telco Customer Churn dataset (7,043 customers, 27% churners). Class imbalance is handled with class weighting rather than discarding data, and every result is measured on the same stratified held-out test set of 1,409 customers.

---

## Key Features

1. **Customer churn prediction:** a guided form with 19 labelled fields and four one-click sample customers, giving the predicted class, churn probability, risk tier and a risk gauge.
2. **ML vs DML vs QML side by side:** every prediction also shows the results of the deep learning and quantum models for the same customer.
3. **Input validation:** ranges, allowed values and service consistency rules are checked before any prediction, and every problem is listed.
4. **Model evaluation:** live re-evaluation on the held-out test set with an adjustable decision threshold and confusion matrix.
5. **Batch scoring:** score a CSV of customers at once and download a risk-sorted outreach list.
6. **Risk explanation and what-if simulator:** see what drives a customer's risk and how retention actions would change it.
7. **Reproducible MLOps:** DVC pipeline, `params.yaml` configuration and MLflow experiment tracking.
8. **Quality assurance:** 152 automated tests, 89% statement coverage, pylint 9.8/10, all checked in CI.

---

## Model Performance

The production **XGBoost** model, evaluated on the stratified 20% held-out test set (N = 1,409 customers):

| Metric | Test value | Interpretation |
| :--- | :--- | :--- |
| Accuracy | 75.66% | Correct predictions across both classes |
| Recall (churn) | 75.13% | Three out of four customers who actually leave are caught |
| Precision (churn) | 52.92% | More than half of flagged customers really churn |
| F1-score (churn) | 0.621 | Balance of precision and recall |
| ROC-AUC | 0.841 | Ranking quality across all thresholds |
| Class weight ratio | 2.77 | Imbalance handled without dropping records (`scale_pos_weight`) |

Confusion matrix on the held-out test set:
- **True positives (281):** at-risk customers correctly flagged for retention.
- **False negatives (93):** churners the model missed.
- **False positives (250):** loyal customers offered a retention incentive.
- **True negatives (785):** loyal customers correctly recognised.

### ML vs DML vs QML

All models are trained and tested on the same split (`02_paradigm_benchmark.py`, results in `paradigm_metrics.json`):

| Paradigm | Model | Accuracy | Recall (churn) | F1 (churn) | ROC-AUC | Trainable parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ML | XGBoost | 75.66% | 79.41% | 0.634 | 0.846 | 5,628 tree nodes |
| DML | Deep MLP (PyTorch) | 73.24% | 79.68% | 0.613 | 0.841 | 5,249 |
| ML | Logistic Regression | 73.67% | 79.41% | 0.616 | 0.839 | 20 |
| QML | Quantum VQC (PennyLane, 6 qubits) | 73.74% | 75.40% | 0.604 | 0.829 | 61 |

The three paradigms reach very similar quality on this tabular problem. The quantum classifier comes within 0.02 ROC-AUC of XGBoost with only 61 trainable parameters, and the deep network has the highest churn recall.

---

## System Architecture

```text
  Telco customer CSV (DVC tracked)
            |
            v
  training_common.py ---- params.yaml (all settings)
  (clean, encode, one stratified split)
            |
     +------+------------------------------+
     |                                     |
     v                                     v
  01_churn_prediction_xgboost.py      02_paradigm_benchmark.py
  Logistic Regression, Random         ML (XGBoost, Logistic Regression)
  Forest, XGBoost                     DML (dml_model.py, PyTorch MLP)
     |                                QML (qml_model.py, PennyLane VQC)
     |                                     |
     v                                     v
  customer_churn_model_xgboost.pkl    models/dml_model.pkl, models/qml_model.pkl
  encoders.pkl                        paradigm_metrics.json
     |            \                  /     |
     |             MLflow (every run: parameters, metrics, plots, model)
     v                                     v
  prediction_pipeline.py  (validation, preprocessing, prediction, evaluation)
            |
            v
  app.py  (Streamlit web application, four pages)
```

The design is described in detail in [docs/SAD_v1.1_Addendum.md](docs/SAD_v1.1_Addendum.md).

---

## Prerequisites
- Python 3.10 to 3.13 (CI runs on 3.12)
- Git
- A web browser

---

## Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Akshayshetty742/hackforge-customer-churn.git
   cd hackforge-customer-churn
   ```

2. **Create and activate a virtual environment:**
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies.** Choose the set you need; each file includes the one above it.

   | File | Installs | Needed for |
   | :--- | :--- | :--- |
   | `requirements.txt` | pandas, scikit-learn, XGBoost, Streamlit | The web application and core tests |
   | `requirements-mlops.txt` | + MLflow, DVC, matplotlib, seaborn | Training and experiment tracking |
   | `requirements-dev.txt` | + pylint, pytest-cov | Full test suite, coverage and linting |
   | `requirements-research.txt` | + PyTorch, PennyLane | Deep learning and quantum models |

   For everything:
   ```bash
   pip install -r requirements-research.txt -r requirements-dev.txt
   ```

---

## How to Launch the Streamlit Application

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501` with four pages:

1. **Customer Churn Prediction**
   - Customer profile form with 19 fields in three sections (Demographics, Services, Contract and Billing). Every field has a label and a help tooltip.
   - **Sample presets:** pick one from the *Sample customer preset* dropdown and press **Load preset**. Loaded values stay in place when you submit the form or switch pages.

     | Preset | What it shows | ML / DML / QML churn probability |
     | :--- | :--- | :--- |
     | High risk: new fiber customer | 1-month senior customer, month-to-month fiber, electronic check | 96.7% / 92.4% / 91.2% |
     | Borderline: month-to-month DSL | 14 months, DSL, no long contract | 49.7% / 65.3% / 43.7% |
     | Low risk: loyal family on 2-year contract | 48 months, two-year contract, all add-ons | 4.2% / 3.4% / 7.5% |
     | Low risk: phone-only customer | 60 months, no internet service | 1.3% / 1.9% / 3.9% |

   - **Clear form** empties every field; fields left empty are listed as missing.
   - **Estimate Total Charges** fills in Total Charges as Monthly Charges x Tenure when the exact total is not known.
   - The result shows the churn classification, probability chart, risk gauge and key risk drivers, followed by the **ML vs DML vs QML comparison**: one card per paradigm with its churn probability, prediction, risk tier and model size. The borderline preset is a good example of the models disagreeing near the decision boundary.
   - The **what-if simulator** shows how retention actions (for example a contract upgrade or a discount) change the churn probability.
   - Invalid input (missing fields, out-of-range numbers, impossible service combinations) is shown as one error list and no prediction is made. See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#5-input-validation--boundary-guards) for all the rules.

   Presets live in `customer_presets.py`. The tests check that each preset passes validation and that its "High risk" / "Low risk" name matches the model's prediction.

2. **Batch Customer Scoring**
   - Score the 100-customer sample batch or upload your own CSV, choose a decision threshold, and download the risk-sorted results.

3. **Model Evaluation**
   - **Saved Verified Metrics:** accuracy, precision, recall, F1, specificity and ROC-AUC on the held-out test set, with a confusion matrix heatmap and a plain-English breakdown. All numbers come from `SAVED_BENCHMARK_METRICS` in `prediction_pipeline.py`.
   - **Run Live Verification:** re-scores the saved model on the same held-out split used in training (stratified 20%, `random_state=42`, 1,409 customers) and confirms it matches the saved benchmark.
   - **Decision threshold slider:** evaluate at any threshold from 0.05 to 0.95 to see the precision and recall trade-off (for example 0.30 gives recall 88% and precision 46%).

   The same evaluation can be run from Python:
   ```python
   import prediction_pipeline as pp
   results = pp.evaluate_model_on_dataset(threshold=0.5)
   all_match, table = pp.compare_with_benchmark(results)
   print(table)
   ```

4. **About the Project:** problem statement, engineering approach and sprint summary.

The deep learning and quantum cards need PyTorch and PennyLane (`requirements-research.txt`). Without them the application works as normal and the two cards show which package to install.

---

## How to Run the Training Pipeline and Benchmark

All settings (data split, model grid, network and circuit sizes) are in `params.yaml`. The pipeline is defined in `dvc.yaml`.

```bash
python 02_paradigm_benchmark.py       # ML vs DML vs QML on one split (about one minute on a laptop CPU)
dvc repro benchmark_paradigms         # the same, as a DVC stage
```

The benchmark writes the leaderboard (`paradigm_metrics.json`), the comparison chart (`plots/paradigm_comparison.png`) and the deep and quantum models used by the application (`models/`). It never changes the production model.

To view every run in MLflow (port 5001, because macOS uses port 5000 for AirPlay):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Then open `http://localhost:5001` and compare the runs in the **Customer-Churn-Paradigms** and **Customer-Churn-Prediction** experiments.

`python 01_churn_prediction_xgboost.py` (DVC stage `train`) trains Logistic Regression, Random Forest and XGBoost, logs each to MLflow and exports the best model by ROC-AUC with its encoders. It replaces `customer_churn_model_xgboost.pkl` and `encoders.pkl` with the new pair. The committed pair is the validated production model whose metrics are listed above; restore it at any time with:

```bash
git checkout -- customer_churn_model_xgboost.pkl encoders.pkl
```

---

## How to Run Automated Tests and Generate Test Reports

```bash
pytest                                  # full suite: 152 tests, about 90 seconds
python tools/generate_test_reports.py   # tests + coverage + pylint, writes docs/reports/test-reports/
```

| Test file | What it covers |
| :--- | :--- |
| `test_pipeline.py` | Artefact loading, input validation, preprocessing, prediction, presets, metrics |
| `test_app.py` | The Streamlit application, run headless with Streamlit AppTest |
| `test_sprint2.py` | Batch scoring, risk explanation, what-if simulator |
| `test_paradigms.py` | Deep learning and quantum models, paradigm comparison |
| `test_system_validation.py` | Test Plan cases TC-01 to TC-13 end to end, benchmark, latency, model integrity |

Every test is linked to a Test Plan v1.1 test case (`traceability.py`). The training pipeline and the benchmark are run inside the tests with all outputs redirected to a temporary folder, so the production model is never changed. Tests that need optional packages (MLflow, PyTorch, PennyLane) skip cleanly when those packages are not installed.

Latest results (30 October 2026): 152 of 152 tests passed, 26 of 26 test cases passed, 89.1% statement coverage, 79.4% branch coverage. See [docs/reports/test-reports/](docs/reports/test-reports/).

---

## Repository Structure

```text
hackforge-customer-churn/
├── .github/workflows/ci.yml          # CI: core tests, pylint, full suite with coverage, test reports
├── app.py                            # Streamlit web application
├── prediction_pipeline.py            # Validation, preprocessing, prediction, evaluation, batch scoring
├── customer_presets.py               # Sample customers for the form
├── training_common.py                # Shared config, data split, metrics, plots, MLflow logging
├── classical_models.py               # Logistic Regression, Random Forest and XGBoost trainers
├── dml_model.py                      # Deep learning model (PyTorch MLP)
├── qml_model.py                      # Quantum ML model (PennyLane variational quantum classifier)
├── 01_churn_prediction_xgboost.py    # Classical training pipeline (DVC stage: train)
├── 02_paradigm_benchmark.py          # ML vs DML vs QML benchmark (DVC stage: benchmark_paradigms)
├── traceability.py                   # Test cases, requirements and the tests that verify them
├── conftest.py                       # Tags every test with its test case ID
├── test_*.py                         # Automated tests
├── tools/generate_test_reports.py    # Generates the Part 3 test reports
├── customer_churn_model_xgboost.pkl  # Production XGBoost model
├── encoders.pkl                      # Encoders saved with the production model
├── models/                           # Deep learning and quantum models used by the app
├── plots/                            # Evaluation charts
├── params.yaml, dvc.yaml, dvc.lock   # Pipeline settings and definition
├── metrics.json, paradigm_metrics.json, classification_report.json
├── requirements*.txt                 # Dependencies (see Installation)
├── pyproject.toml, pytest.ini        # pylint, coverage and pytest settings
├── DEVELOPER_GUIDE.md                # Technical design and decisions
├── PROJECT_REPORT.md                 # Project report
└── docs/
    ├── SRS_v1.1_Addendum.md          # Requirements v1.1
    ├── SAD_v1.1_Addendum.md          # Architecture and design v1.1
    ├── TEST_PLAN_v1.1.md             # Test plan and test cases v1.1
    ├── DEVIATIONS_AND_ENHANCEMENTS.md
    ├── reports/                      # All submission reports (SRS/STP/SAD, backlog, sprint, test reports)
    ├── deliverables/                 # Deliverable 1 document and presentation
    └── evidence/                     # Screenshots and sprint videos
```

---

## Documentation

| Document | Contents |
| :--- | :--- |
| [docs/README.md](docs/README.md) | Index of all project documents |
| [SRS v1.1](docs/SRS_v1.1_Addendum.md) | Requirements added during the sprints |
| [SAD v1.1](docs/SAD_v1.1_Addendum.md) | Architecture, component and sequence diagrams, API design |
| [Test Plan v1.1](docs/TEST_PLAN_v1.1.md) | 26 test cases with status and traceability |
| [Reports](docs/reports/README.md) | All submission reports in one folder |
| [Product Backlog](docs/reports/Product_Backlog.md) | Backlog items with story points, mapped to requirements and test cases |
| [Deviations and Enhancements](docs/DEVIATIONS_AND_ENHANCEMENTS.md) | Changes from the submitted documents |
| [Test Reports](docs/reports/test-reports/README.md) | Unit test, system validation, code coverage and branch coverage reports |
| [Developer Guide](DEVELOPER_GUIDE.md) | Design decisions and code walkthrough |
| [Project Report](PROJECT_REPORT.md) | Full project report |

---

## CI/CD Automation

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every pull request and every push to `main`:

1. **Core Tests:** installs only `requirements.txt` and runs the test suite, which proves the application works with its minimum dependencies.
2. **Quality:** installs every dependency (CPU-only PyTorch), runs pylint (must score at least 8.5/10), runs the full suite with statement and branch coverage (must reach 85%), and uploads the generated test reports as a build artifact.

No secrets or credentials are needed, so the workflow runs on the public repository as is.

---

## Limitations and Known Boundaries

- **Categorical encodings:** categorical inputs use scikit-learn `LabelEncoder` for the tree models. Unseen categories are rejected with a clear message rather than silently corrupted.
- **Probabilistic estimates:** predictions are decision support estimates, not guarantees.
- **Quantum simulation:** the quantum model runs on a classical simulator, not on quantum hardware.
- **Single dataset:** results come from one public dataset and should be re-validated before use on other customer data.
- **Stateless and local:** the application stores no customer input and sends no data to any external service.

---

## Deliverable Sprints

- **Sprint 1 (functional core and user interface):**
  - Light, high-contrast interface (`.streamlit/config.toml`).
  - Prediction form with demographic, service and billing inputs, and one-click sample customers.
  - Real-time XGBoost probability gauge and churn risk driver breakdown.
  - Input validation and boundary guards.

- **Sprint 2 (complete product):**
  - What-if retention simulator with risk reduction and preserved revenue.
  - Decision threshold tuning with confusion matrix.
  - Batch CSV scoring with downloadable outreach list.
  - Automated tests and GitHub Actions CI.

- **Sprint 3 (ML vs DML vs QML, MLOps and testing):**
  - Deep learning (PyTorch) and quantum machine learning (PennyLane) models, benchmarked against XGBoost on the same split.
  - Side-by-side ML, DML and QML predictions in the application.
  - DVC pipeline, `params.yaml` configuration and MLflow experiment tracking.
  - All Test Plan cases automated (152 tests), coverage and branch coverage reports, pylint quality gate.
  - SRS, SAD and Test Plan v1.1.

---

## Sprint Demonstration Videos

| Sprint | Content | File |
| :--- | :--- | :--- |
| Sprint 1 | Working product: interface, sample customers, real-time probability gauge and risk drivers. | [docs/evidence/videos/sprint1.mp4](docs/evidence/videos/sprint1.mp4) |
| Sprint 2 | Complete product: what-if simulator, batch CSV scoring with download, threshold tuning and automated tests. | [docs/evidence/videos/sprint2.mp4](docs/evidence/videos/sprint2.mp4) |
