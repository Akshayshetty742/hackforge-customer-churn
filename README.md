# Hackforge — Customer Churn Prediction

[![Customer Churn CI Pipeline](https://github.com/Akshayshetty742/hackforge-customer-churn/actions/workflows/ci.yml/badge.svg)](https://github.com/Akshayshetty742/hackforge-customer-churn/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-green.svg)](https://xgboost.readthedocs.io/)

A complete, production-grade Machine Learning decision support solution to predict customer churn in telecommunications and subscription businesses.

This project delivers **Deliverable Part 2** of the Software Engineering coursework: an interactive **Streamlit web application**, modular inference pipeline, automated test suite, and continuous integration workflow.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Achievements & Model Performance](#key-achievements--model-performance)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Prerequisites](#prerequisites)
- [Installation Instructions](#installation-instructions)
- [How to Launch the Streamlit Application](#how-to-launch-the-streamlit-application)
- [How to Run the Training Script](#how-to-run-the-training-script)
- [How to Run Automated Tests](#how-to-run-automated-tests)
- [Repository Structure](#repository-structure)
- [CI/CD Automation](#cicd-automation)
- [Limitations & Known Boundaries](#limitations--known-boundaries)
- [Deliverable Sprints](#deliverable-sprints)

---

## Project Overview
Customer retention is crucial for subscription-based business sustainability. Acquiring new customers typically costs 5 to 7 times more than retaining existing subscribers. 

This application analyzes customer demographic profiles, subscribed services, contract commitments, and billing metrics to compute real-time churn probability, allowing account managers and customer success teams to initiate targeted retention campaigns before customers cancel.

---

## Key Achievements & Model Performance

The tuned **XGBoost Classifier** was evaluated on a stratified 20% holdout test set ($N = 1,409$ customers) from the Telco Customer Churn dataset:

| Metric | Verified Test Value | Business Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **75.66%** | Overall correct predictions across both churn and loyal classes |
| **Recall (Churn)** | **75.13%** | Captures ~75% of all customers who actually churn |
| **Precision (Churn)** | **52.92%** | Balanced trade-off ensuring high coverage of at-risk users |
| **F1-Score (Churn)** | **0.6210** | Strong harmonic mean between precision and recall |
| **Class Weight Ratio** | **2.77** | Handled imbalance without dropping records (`scale_pos_weight`) |

### Verified Confusion Matrix (Holdout Test Set)
- **True Positives (281):** At-risk customers successfully flagged for retention interventions.
- **False Negatives (93):** Churners missed by the model (~24.9% miss rate).
- **False Positives (250):** Loyal customers offered proactive incentives (tolerable retention cost).
- **True Negatives (785):** Retained customers correctly recognized.

---

## System Architecture & Workflow

```text
  [ Raw Customer Data ]
            │
            ▼
┌─────────────────────────┐
│   Data Preprocessing    │ ──> Drops customerID, formats TotalCharges to float
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│ Categorical LabelEncoder│ ──> Applied consistently via serialized encoders.pkl
└─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│  XGBoost Classifier     │ ──> Serialized model artifact (GridSearchCV tuned)
└─────────────────────────┘
            │
            ├──> Streamlit UI (app.py) ────> Interactive Form & Probabilities
            │
            └──> Automated Tests (pytest) ─> CI/CD Pipeline (.github/workflows)
```

---

## Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git
- Web browser (Chrome, Edge, Firefox, or Safari)

---

## Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Akshayshetty742/hackforge-customer-churn.git
   cd hackforge-customer-churn
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv churn_env
     .\churn_env\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv churn_env
     source churn_env/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Launch the Streamlit Application

Run the application using:
```bash
streamlit run app.py
```

The application opens at `http://localhost:8501` featuring 3 tabs:
1. **Customer Churn Prediction:**
   - Comprehensive customer profile input form (Demographics, Services, Contract, Billing).
   - "Load High-Risk Sample" and "Load Low-Risk Sample" preset buttons for live demonstration.
   - Real-time churn classification and probability distribution chart.
   - Validation checks on numerical ranges and logical service combinations.
2. **Model Evaluation:**
   - Saved benchmark metrics and confusion matrix breakdown.
   - Live evaluation runner against `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. **About the Project:**
   - Problem statement, engineering rationale, and sprint deliverable breakdown.

---

## How to Run the Training Script

To re-train the XGBoost model with `GridSearchCV` and re-export the serialized artifacts:
```bash
python 01_churn_prediction_xgboost.py
```

This generates:
- `customer_churn_model_xgboost.pkl`: Serialized dictionary containing the tuned model and exact feature names.
- `encoders.pkl`: Serialized dictionary of `LabelEncoder` objects for all 15 categorical features.

---

## How to Run Automated Tests

The project includes unit and integration tests covering artifact loading, input validation, numerical boundaries, unseen categorical handling, column ordering, and prediction outputs:

```bash
pytest test_pipeline.py -v
```

All 13 automated tests run in under 3 seconds without retraining the model.

---

## Repository Structure

```text
hackforge-customer-churn/
│
├── .github/
│   └── workflows/
│       └── ci.yml                     # Continuous Integration workflow
│
├── 01_churn_prediction_xgboost.py     # Original model training & hyperparameter tuning
├── app.py                             # Interactive Streamlit application
├── prediction_pipeline.py             # Reusable inference, validation & evaluation module
├── test_pipeline.py                   # Automated pytest test suite
│
├── customer_churn_model_xgboost.pkl   # Serialized tuned XGBoost model artifact
├── encoders.pkl                       # Serialized categorical LabelEncoders
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Telco churn dataset
│
├── DEVELOPER_GUIDE.md                 # Technical architecture & design decisions
├── PROJECT_REPORT.md                  # Comprehensive academic / engineering report
├── requirements.txt                   # Project package dependencies
├── .gitignore                         # Exclusions for virtual environments, secrets, caches
└── README.md                          # Main project documentation
```

---

## CI/CD Automation

A GitHub Actions workflow is provided in `.github/workflows/ci.yml`. On every push and pull request to the `main` branch, the workflow:
1. Checks out repository code.
2. Sets up Python 3.11 with pip caching.
3. Installs dependencies from `requirements.txt`.
4. Executes `pytest test_pipeline.py -v`.
5. Requires no private secrets or credentials, ensuring public repository compatibility.

---

## Limitations & Known Boundaries

- **Categorical Encodings:** Categorical attributes use scikit-learn `LabelEncoder`. Unseen categories are safely rejected by `prediction_pipeline.py` rather than silently corrupted.
- **Probabilistic Estimates:** Predictions are probabilistic decision-support estimates and not absolute guarantees.
- **Stateless Inference:** The application does not store customer profile inputs to external databases, keeping customer data private.

---

## Deliverable Sprints

- **Sprint 1 (Functional Core):**
  - Interactive prediction UI with demographic, services, and billing inputs.
  - Integration of serialized XGBoost model and fitted encoders.
  - Input validation and sample demonstration presets.
- **Sprint 2 (Quality & Reliability):**
  - Standalone `prediction_pipeline.py` architecture.
  - Comprehensive automated test suite (`test_pipeline.py`).
  - GitHub Actions CI workflow (`.github/workflows/ci.yml`).
  - Project documentation and developer guides.
  - 
  ## Deliverable 2 Verification

Automated tests are run through the Customer Churn CI Pipeline using GitHub Actions.
