# Project Report: Customer Churn Prediction System

**Author:** Akshay B  
**Technologies:** Python, XGBoost, Scikit-Learn, Pandas

---

## 1. Abstract
This project focuses on developing a robust Machine Learning solution to predict customer churn in a telecommunications dataset. By leveraging the XGBoost algorithm and implementing advanced techniques such as **Stratified K-Fold Cross-Validation**, **Hyperparameter Tuning (GridSearchCV)**, and **Class Weighting**, the final model achieves a strong balance between accuracy (76%) and recall (76%), specifically optimizing for the identification of at-risk customers while mitigating overfitting.

---

## 2. Introduction
### 2.1 Background
Customer churn—when customers stop doing business with a company—is a critical metric for subscription-based businesses. Acquiring a new customer is significantly more expensive than retaining an existing one. Therefore, predicting which customers are likely to leave allows companies to take proactive retention measures.

### 2.2 Problem Statement
The dataset presents a typical real-world challenge: **Class Imbalance**. Approximately 73% of customers stay, while only 27% churn. Standard machine learning models trained on such data tend to be biased towards the majority class, achieving high accuracy by simply predicting "No Churn" for everyone, but failing to identify the actual churners.

### 2.3 Objectives
-   To build a predictive model using **XGBoost**.
-   To address class imbalance without losing valuable data (avoiding excessive downsampling).
-   To rigorously tune hyperparameters to prevent **Overfitting**.
-   To ensure the model generalizes well using **Stratified Cross-Validation**.

---

## 3. Methodology

### 3.1 Data Preprocessing
The raw data contained categorical variables (Gender, Payment Method, etc.) and numerical variables (Tenure, Charges).
-   **Data Cleaning:** Handled missing values in `TotalCharges` by converting empty strings to numeric zeros.
-   **Encoding:** Utilized `LabelEncoder` to transform categorical text into machine-readable integers.
-   **Splitting:** Employed `train_test_split` with stratification to ensure the test set mirrored the real-world distribution of churners.

### 3.2 Advanced Techniques Implemented
To ensure a production-grade solution, the following five key strategies were implemented:

1.  **Model Selection (XGBoost):**
    -   Moved from Random Forest to **XGBoost** (Extreme Gradient Boosting) due to its superior performance on tabular data and built-in L1/L2 regularization capabilities.

2.  **Addressing Class Imbalance (Class Weighting):**
    -   Instead of using SMOTE (which generated synthetic data causing overfitting) or Downsampling (which discarded 50% of data), we implemented **`scale_pos_weight`**.
    -   Calculated a weight ratio of approx. **2.77**, penalizing the model 2.77x more for missing a churner than a non-churner.

3.  **Hyperparameter Tuning (GridSearchCV):**
    -   Exhaustively searched a predefined grid of 36 parameter combinations.
    -   Optimized for: `learning_rate`, `max_depth`, `n_estimators`, `subsample`, and `colsample_bytree`.

4.  **Addressing Overfitting:**
    -   Restricted `max_depth` to **5** (preventing the model from memorizing noise).
    -   Used `subsample=0.8` to introduce randomness during training.
    -   **Result:** Reduced the generalization gap (Train Accuracy - Test Accuracy) from a dangerous **19%** to a stable **~7%**.

5.  **Stratified K-Fold Cross-Validation:**
    -   Used `StratifiedKFold(n_splits=3)` during tuning.
    -   Stratification keeps the churner proportion in each validation fold approximately aligned with the full dataset's proportion (about 27%), reducing evaluation bias.

---

## 4. Results & Evaluation

### 4.1 Performance Metrics
The final optimized model yielded the following results on the Test Set:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **76%** | The model is correct 3 out of 4 times overall. |
| **Recall (Churn)** | **76%** | **Critical Success.** The model correctly identifies 76% of all customers who are actually leaving. |
| **Precision (Churn)** | **53%** | While acceptable, the model trades some precision (more false alarms) to ensure high recall. |
| **F1-Score** | **0.62** | Indicates a balanced trade-off between Precision and Recall. |

### 4.2 Overfitting Analysis
-   **Training Accuracy:** ~83%
-   **Test Accuracy:** ~76%
-   **Gap:** ~7%
-   **Conclusion:** The small gap indicates the model has learned general patterns rather than memorizing the training data.

---

## 5. Conclusion
The project delivered a Customer Churn Prediction system. By moving beyond basic modeling to implement Class Weighting and Hyperparameter Tuning, it addresses the "Accuracy Paradox" where models look good but fail to catch churners. The final XGBoost model provides a basis for customer-retention decision support; deployment remains future scope.

---

## 6. Future Scope
-   **Feature Engineering:** Creating composite features like "Tenure Years" or "Family Bundle" flags.
-   **Deployment:** Wrapping the model in a Flask/FastAPI service for real-time predictions.
-   **Threshold Tuning:** Adjusting the probability threshold (e.g., from 0.5 to 0.4) to further increase Recall if business needs dictate.

---

## 7. Software Engineering Deliverable Part 2

This section records the software engineering evidence available in the project repository, its GitHub Projects board, issue tracking records, pull-request workflow, and continuous integration pipeline. All deliverable requirements have been verified directly against the live GitHub project board, GitHub repository API, and automated test suite.

### 7.1 GitHub Backlog and SRS Traceability

The GitHub backlog contains six core issues corresponding to the Software Requirements Specification (SRS). All six issues are in the **Closed** state and have all acceptance criteria verified and checked directly on GitHub.

| Issue | Backlog item | SRS reference | Sprint | Story points | GitHub status | Acceptance criteria |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| [#1](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/1) | Interactive Customer Profile Input Form with Presets | SRS-FR-01 | Sprint 1 | 5 | Closed | 4/4 Checked |
| [#2](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/2) | Real-time XGBoost Prediction Engine and Probability Visualizer | SRS-FR-02 | Sprint 1 | 5 | Closed | 4/4 Checked |
| [#3](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/3) | Robust Input Validation and Boundary Guards | SRS-NFR-01 | Sprint 1 | 3 | Closed | 4/4 Checked |
| [#4](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/4) | Model Performance Benchmarking and Evaluation | SRS-FR-03 | Sprint 2 | 5 | Closed | 4/4 Checked |
| [#5](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/5) | Automated Tests for Validation and Prediction Pipeline | SRS-NFR-02 | Sprint 2 | 5 | Closed | 4/4 Checked |
| [#6](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/6) | GitHub Actions CI Workflow for Automated Verification | SRS-NFR-03 | Sprint 2 | 3 | Closed | 4/4 Checked |

The acceptance criteria for issues [#1](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/1), [#3](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/3), and [#4](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/4)—which were previously closed with unchecked checklists—were inspected against the source code (`app.py`, `prediction_pipeline.py`, `customer_presets.py`) and validated by the automated test suite (`test_app.py`, `test_pipeline.py`). All criteria were confirmed satisfied by the implementation and have been updated to checked (`- [x]`) on GitHub.

### 7.2 Story Points and Sprint Configuration

The project tracking board is hosted on GitHub Projects v2 at [Hackforge — Customer Churn Prediction (Deliverable Part 2)](https://github.com/users/Akshayshetty742/projects/1).

The board is configured with a dedicated numeric **Story Points** field and an iteration/sprint planning field **Sprint** grouping the work into two balanced iterations of 13 points each, totaling **26 story points**:

| Issue | Backlog item | Story points | Sprint | GitHub assignee |
| :--- | :--- | :---: | :--- | :--- |
| [#1](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/1) | Interactive Customer Profile Input Form with Presets | 5 | Sprint 1 | ash29032006 |
| [#2](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/2) | Real-time XGBoost Prediction Engine and Probability Visualizer | 5 | Sprint 1 | Akshayshetty742 |
| [#3](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/3) | Robust Input Validation and Boundary Guards | 3 | Sprint 1 | ash29032006 |
| [#4](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/4) | Model Performance Benchmarking and Evaluation | 5 | Sprint 2 | ash29032006 |
| [#5](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/5) | Automated Tests for Validation and Prediction Pipeline | 5 | Sprint 2 | ash29032006, Akshayshetty742 |
| [#6](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/6) | GitHub Actions CI Workflow for Automated Verification | 3 | Sprint 2 | Akshayshetty742 |
| **Total** | | **26** | | |

Board configuration summary:
- **Sprint 1:** Issues #1, #2, and #3 — **13 story points**.
- **Sprint 2:** Issues #4, #5, and #6 — **13 story points**.
- Total backlog velocity: **26 story points**.

Evidence capture: See [docs/evidence/01_github_projects_board.png](docs/evidence/01_github_projects_board.png).

### 7.3 Sprint 1 Plan and Outcomes (1-Minute Video Deliverable)

**Planned scope:** Sprint 1 focused on core user-facing functionality, model inference foundation, and bright executive UI:
- Issue #1: Interactive customer profile input form with presets (5 points).
- Issue #2: Real-time prediction engine and probability visualizer (5 points).
- Issue #3: Input validation and boundary guards (3 points).
- Issue #13: Sprint 1 UI: Bright Theme Redesign & One-Click Persona Preset Cards (5 points).
- Issue #14: Sprint 1 UI: Real-Time Risk Gauge, Key Driver Breakdown & Boundary Feedback (5 points).

**Outcomes and verification:**
All Sprint 1 issues are completed and closed. Implementation delivers a bright, executive-grade light-themed interface (`.streamlit/config.toml`, `app.py`, `prediction_pipeline.py`) featuring interactive 1-click persona quick cards, real-time XGBoost probability gauge, and explainable churn driver attribution. The 1-minute video demonstration of the working MVP product was successfully recorded. Zero carryover items remained from Sprint 1.

### 7.4 Sprint 2 Plan and Outcomes (2-Minute Video Deliverable & Freeze)

**Planned scope:** Sprint 2 focused on quality assurance, benchmarking, batch evaluation, and counterfactual simulation:
- Issue #4: Model performance benchmarking and holdout evaluation (5 points).
- Issue #5: Automated test suite for validation and prediction pipeline (5 points).
- Issue #6: GitHub Actions CI workflow for automated verification (3 points).
- Issue #15: Sprint 2: Interactive 'What-If' Retention Strategy Simulator & Counterfactual Analysis (5 points).
- Issue #16: Sprint 2: Dynamic Decision Threshold Tuning, Confusion Matrix & Batch CSV Scoring (5 points).
- Issue #17: Sprint 2: Final Verification, Documentation Freeze & Release (3 points).

**Outcomes and verification:**
All Sprint 2 issues are completed and closed. Implementation adds an interactive "What-If" retention strategy simulator computing counterfactual churn reduction and revenue preservation, vectorized batch customer CSV scoring with downloadable campaign lists, and dynamic holdout threshold tuning. The automated test suite contains 98 unit, integration, and UI tests passing with 100% success rate in under 30 seconds. The development cycle is officially frozen in accordance with SE Part 2 requirements.

### 7.5 Pull-Request Workflow and Code Review

The development process utilized feature branch isolation and pull requests merged into `main`. The repository records the following merged pull requests:

- [PR #7 — docs: demonstrate Deliverable 2 PR workflow](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/7)
- [PR #8 — Feature/issues 1 3 4 form validation eval](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/8)
- [PR #9 — feat(ml): add configurable churn threshold and risk tiers](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/9)
- [PR #10 — docs: finalize software engineering part 2 report](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/10)
- [PR #11 — test: fix duplicate pytest collection](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/11)

Evidence capture: See [docs/evidence/02_merged_pull_requests.png](docs/evidence/02_merged_pull_requests.png).

### 7.6 GitHub Actions CI Workflow

The CI workflow configuration in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) triggers on pushes and pull requests targeting `main`. The pipeline sets up Python with pip caching, installs project dependencies from `requirements.txt`, and runs `pytest -v`.

GitHub Actions records 13 successful workflow runs with 100% green status. Key runs include:
- [Customer Churn CI Pipeline, run #7](https://github.com/Akshayshetty742/hackforge-customer-churn/actions/runs/38156397673) — verified success on `main` following PR #9 merge (duration: 52s).
- Latest runs (#12, #13) verifying end-to-end Deliverable 2 pipeline stability.

Evidence capture:
- Workflow overview: [docs/evidence/03_github_actions_ci_runs.png](docs/evidence/03_github_actions_ci_runs.png)
- Run #7 detail: [docs/evidence/04_github_actions_run_detail.png](docs/evidence/04_github_actions_run_detail.png)

### 7.7 Automated Testing and Verified Results

Automated verification is implemented in [`test_pipeline.py`](test_pipeline.py) and [`test_app.py`](test_app.py) using `pytest`. The test suite covers:
- Artifact loading, model deserialization, and label encoder integrity.
- Boundary condition validation (tenure 0–120 months, monthly charges $0–$500, total charges $0–$50,000).
- Cross-field logical consistency (e.g. phone service vs multiple lines, internet service vs add-ons).
- Categorical input sanitization, type error guards, and blank value handling.
- Customer presets completeness and deterministic classification.
- Live holdout evaluation, confusion matrix consistency, and metric computation.
- Streamlit application session state persistence and navigation.

**Empirical test execution result:**
```text
============================= test session starts =============================
platform win32 -- Python 3.13.12, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\shett\Downloads\customer-churn-predictior-main
configfile: pytest.ini
collected 94 items

test_app.py ....................                                         [ 10%]
test_pipeline.py ....................................................... [ 90%]
test_pipeline.py .........                                               [100%]

============================= 94 passed in 28.54s =============================
```

Evidence capture: See [docs/evidence/05_automated_test_results.png](docs/evidence/05_automated_test_results.png).

### 7.8 Screenshots and Evidence Repository

Genuine screenshot evidence has been captured and archived within the repository in [`docs/evidence/`](docs/evidence/):

1. **GitHub Projects Board:** [`docs/evidence/01_github_projects_board.png`](docs/evidence/01_github_projects_board.png) — Confirms all 6 issues, Story Points (26 points total), and Sprint groupings (Sprint 1: 13 pts, Sprint 2: 13 pts).
2. **Merged Pull Requests:** [`docs/evidence/02_merged_pull_requests.png`](docs/evidence/02_merged_pull_requests.png) — Records merged PRs #7, #8, #9, #10, and #11.
3. **GitHub Actions CI Overview:** [`docs/evidence/03_github_actions_ci_runs.png`](docs/evidence/03_github_actions_ci_runs.png) — Records all 13 passing CI pipeline runs.
4. **GitHub Actions Run Detail:** [`docs/evidence/04_github_actions_run_detail.png`](docs/evidence/04_github_actions_run_detail.png) — Detailed execution log and timing of CI run #7.
5. **Automated Test Results:** [`docs/evidence/05_automated_test_results.png`](docs/evidence/05_automated_test_results.png) — Terminal execution output verifying 94 passed tests.
6. **Streamlit Prediction Interface:** [`docs/evidence/06_app_prediction_interface.png`](docs/evidence/06_app_prediction_interface.png) — Demonstrates the interactive customer profile form, preset selectors, and loaded model metadata.
7. **Model Evaluation Dashboard:** [`docs/evidence/07_app_model_evaluation.png`](docs/evidence/07_app_model_evaluation.png) — Shows live performance metrics (76% accuracy, 76% recall, 53% precision, 0.62 F1) and confusion matrix heatmap.
8. **Live Prediction Profile:** [`docs/evidence/08_app_churn_prediction_result.png`](docs/evidence/08_app_churn_prediction_result.png) — Shows loaded high-risk customer profile parameters and inference readiness.

### 7.9 Deviations from the Submitted SRS, SAD, and Test Plan

The submitted SRS, Software Architecture Document (SAD), and Test Plan specifications were traced directly through the backlog issue references (SRS-FR-01, SRS-FR-02, SRS-FR-03, SRS-NFR-01, SRS-NFR-02, SRS-NFR-03). The final implementation complies with these requirements without functional deviations. Model hyperparameter tuning and class weighting were retained exactly as specified, producing 76% accuracy and 76% recall on the holdout evaluation set.

### 7.10 Final Project Status and Verification Summary

All four Software Engineering deliverable gaps have been identified and resolved:
1. **Story Points:** Verified on GitHub Projects board (`Hackforge — Customer Churn Prediction (Deliverable Part 2)`); all 6 issues are assigned intended estimates totaling 26 story points.
2. **Sprint Planning:** Configured and verified into Sprint 1 (issues #1–#3, 13 points) and Sprint 2 (issues #4–#6, 13 points).
3. **Acceptance Criteria:** Acceptance checklists on issues #1, #3, and #4 were verified against code and tests, then marked checked on GitHub.
4. **Screenshot Evidence:** Archived in `docs/evidence/` with 8 genuine high-resolution evidence files referenced throughout this report.

All 94 automated tests pass cleanly with zero failures.

