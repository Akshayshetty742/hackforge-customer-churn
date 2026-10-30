# Project Report: Customer Churn Prediction System

**Author:** Akshay B  , Ashwin C H , Amarpatil
**Technologies:** Python, XGBoost, Scikit-Learn, Pandas

---

## 1. Abstract
This project focuses on developing a robust Machine Learning solution to predict customer churn in a telecommunications dataset. By leveraging the XGBoost algorithm and implementing advanced techniques such as **Stratified K-Fold Cross-Validation**, **Hyperparameter Tuning (GridSearchCV)**, and **Class Weighting**, the final model achieves a strong balance between accuracy (75.7%) and recall (75.1%), specifically optimizing for the identification of at-risk customers while mitigating overfitting. The project also compares this classical model with a deep learning model and a quantum machine learning model on the same data, and delivers all three predictions through an interactive web application.

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
| **Recall (Churn)** | **75%** | **Critical Success.** The model correctly identifies 75% of all customers who are actually leaving. |
| **Precision (Churn)** | **53%** | While acceptable, the model trades some precision (more false alarms) to ensure high recall. |
| **F1-Score** | **0.62** | Indicates a balanced trade-off between Precision and Recall. |

### 4.2 Overfitting Analysis
-   **Training Accuracy:** ~83%
-   **Test Accuracy:** ~76%
-   **Gap:** ~7%
-   **Conclusion:** The small gap indicates the model has learned general patterns rather than memorizing the training data.

---

## 5. Conclusion
The project delivered a Customer Churn Prediction system. By moving beyond basic modeling to implement Class Weighting and Hyperparameter Tuning, it addresses the "Accuracy Paradox" where models look good but fail to catch churners. The final XGBoost model is delivered through a Streamlit decision support application, alongside deep learning and quantum machine learning models for comparison.

---

## 6. Future Scope
-   **Feature Engineering:** Creating composite features like "Tenure Years" or "Family Bundle" flags.
-   **Deployment:** Packaging the application in a container and serving the model through a REST API.
-   **Monitoring:** Tracking data drift and scheduling retraining.
-   **Quantum hardware:** Running the variational quantum classifier on real quantum hardware instead of a simulator.

---

## 7. Software Engineering Deliverable Part 2

This section records the software engineering process evidence: the GitHub Projects board, the backlog with story points, the sprints, the pull-request workflow and continuous integration.

### 7.1 GitHub Backlog and SRS Traceability

The backlog was created as GitHub issues from the requirements in the SRS, each with acceptance criteria, an SRS reference, story points, a sprint and an assignee. All Sprint 1 and Sprint 2 issues are closed with every acceptance criterion checked. The mapping of each issue to the detailed requirement IDs of SRS v1.1 and to its test cases is in [docs/TRACEABILITY_MATRIX.md](docs/TRACEABILITY_MATRIX.md).

### 7.2 Story Points and Sprint Configuration

The project board is hosted on GitHub Projects at [Hackforge — Customer Churn Prediction (Deliverable Part 2)](https://github.com/users/Akshayshetty742/projects/1), with a numeric **Story Points** field and a **Sprint** field.

| Issue | Backlog item | Story points | Sprint |
| :--- | :--- | :---: | :--- |
| [#1](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/1) | Interactive Customer Profile Input Form with Presets | 5 | Sprint 1 |
| [#2](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/2) | Real-time XGBoost Prediction Engine and Probability Visualizer | 5 | Sprint 1 |
| [#3](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/3) | Robust Input Validation and Boundary Guards | 3 | Sprint 1 |
| [#13](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/13) | Bright Theme Redesign and One-Click Persona Preset Cards | 5 | Sprint 1 |
| [#14](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/14) | Real-Time Risk Gauge, Key Driver Breakdown and Boundary Feedback | 5 | Sprint 1 |
| [#4](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/4) | Model Performance Benchmarking and Evaluation | 5 | Sprint 2 |
| [#5](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/5) | Automated Tests for Validation and Prediction Pipeline | 5 | Sprint 2 |
| [#6](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/6) | GitHub Actions CI Workflow for Automated Verification | 3 | Sprint 2 |
| [#15](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/15) | Interactive What-If Retention Strategy Simulator | 5 | Sprint 2 |
| [#16](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/16) | Decision Threshold Tuning, Confusion Matrix and Batch CSV Scoring | 5 | Sprint 2 |
| [#17](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/17) | Final Verification, Documentation Freeze and Release | 3 | Sprint 2 |
| **Total** | | **49** | |

- **Sprint 1:** 5 items, 23 story points, all completed.
- **Sprint 2:** 6 items, 26 story points, all completed.
- **Sprint 3:** 10 items, 44 story points (see section 8).

Evidence: [docs/evidence/01_github_projects_board.png](docs/evidence/01_github_projects_board.png).

### 7.3 Sprint 1 Plan and Outcomes (1-Minute Video Deliverable)

**Planned scope:** core user-facing functionality, the model inference foundation and the user interface:
- Issue #1: Interactive customer profile input form with presets (5 points).
- Issue #2: Real-time prediction engine and probability visualiser (5 points).
- Issue #3: Input validation and boundary guards (3 points).
- Issue #13: Bright theme redesign and one-click persona preset cards (5 points).
- Issue #14: Real-time risk gauge, key driver breakdown and boundary feedback (5 points).

**Outcomes:** all Sprint 1 items were completed. The light-themed interface (`.streamlit/config.toml`, `app.py`, `prediction_pipeline.py`) provides one-click sample customers, a real-time XGBoost probability gauge and an explanation of the churn drivers. The 1-minute video of the working product is in `docs/evidence/videos/sprint1.mp4`.

### 7.4 Sprint 2 Plan and Outcomes (2-Minute Video Deliverable and Freeze)

**Planned scope:** quality assurance, evaluation, batch scoring and simulation:
- Issue #4: Model performance benchmarking and held-out evaluation (5 points).
- Issue #5: Automated test suite for validation and prediction (5 points).
- Issue #6: GitHub Actions CI workflow (3 points).
- Issue #15: What-if retention strategy simulator (5 points).
- Issue #16: Decision threshold tuning, confusion matrix and batch CSV scoring (5 points).
- Issue #17: Final verification, documentation freeze and release (3 points).

**Outcomes:** all Sprint 2 items were completed. The product gained the what-if retention simulator with counterfactual churn reduction and preserved revenue, batch CSV scoring with a downloadable outreach list, and decision threshold tuning. The automated test suite passed in full in CI. The video of the complete product is in `docs/evidence/videos/sprint2.mp4`.

### 7.5 Pull-Request Workflow

Development used feature branches merged into `main` through pull requests:

- [PR #7 — docs: demonstrate Deliverable 2 PR workflow](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/7)
- [PR #8 — Feature/issues 1 3 4 form validation eval](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/8)
- [PR #9 — feat(ml): add configurable churn threshold and risk tiers](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/9)
- [PR #10 — docs: finalize software engineering part 2 report](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/10)
- [PR #11 — test: fix duplicate pytest collection](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/11)
- [PR #12 — docs: complete Software Engineering deliverable evidence](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/12)
- [PR #18 — feat: Sprint 2 complete product](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/18) (closes #15, #16, #17)

Pull requests #9 to #18 ran the CI workflow before merging, and all 20 workflow runs recorded on GitHub have passed. Evidence: [docs/evidence/02_merged_pull_requests.png](docs/evidence/02_merged_pull_requests.png).

### 7.6 GitHub Actions CI Workflow

The workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every pull request and every push to `main`. All 20 recorded workflow runs have passed. In Sprint 3 the workflow was extended with two jobs:

1. **Core Tests:** installs `requirements.txt` only and runs the test suite.
2. **Quality:** installs all dependencies, runs pylint (minimum 8.5/10), runs the full suite with statement and branch coverage (minimum 85%), and uploads the test reports as an artifact.

Evidence: [docs/evidence/03_github_actions_ci_runs.png](docs/evidence/03_github_actions_ci_runs.png), [docs/evidence/04_github_actions_run_detail.png](docs/evidence/04_github_actions_run_detail.png).

### 7.7 Automated Testing

The suite has grown steadily: 94 tests in the Deliverable 2 evidence run, 98 at the end of Sprint 2, and 152 in Sprint 3. It covers:
- Artefact loading, model deserialisation and encoder integrity.
- Boundary validation (tenure 0 to 120 months, monthly charges 0 to 500, total charges 0 to 50,000) and cross-field consistency.
- Sample customers and deterministic classification.
- Held-out evaluation, confusion matrix consistency and metric computation.
- The Streamlit application, run headless with Streamlit AppTest.
- Batch scoring, risk explanation and the what-if simulator.
- The deep learning and quantum models and the paradigm comparison.
- The full training pipeline and benchmark, run end to end in a temporary folder (Test Plan TC-01 to TC-13).

Latest run (30 October 2026): **152 of 152 tests passed**. Details are in section 9.

### 7.8 Screenshots and Evidence

Evidence is archived in [`docs/evidence/`](docs/evidence/):

1. [`01_github_projects_board.png`](docs/evidence/01_github_projects_board.png): project board with the Story Points and Sprint fields.
2. [`02_merged_pull_requests.png`](docs/evidence/02_merged_pull_requests.png): merged pull requests.
3. [`03_github_actions_ci_runs.png`](docs/evidence/03_github_actions_ci_runs.png): passing CI runs.
4. [`04_github_actions_run_detail.png`](docs/evidence/04_github_actions_run_detail.png): detail of a CI run.
5. [`05_automated_test_results.png`](docs/evidence/05_automated_test_results.png): automated test results.
6. [`06_app_prediction_interface.png`](docs/evidence/06_app_prediction_interface.png): prediction form, presets and model information.
7. [`07_app_model_evaluation.png`](docs/evidence/07_app_model_evaluation.png): model evaluation page with metrics and confusion matrix.
8. [`08_app_churn_prediction_result.png`](docs/evidence/08_app_churn_prediction_result.png): prediction result for a high-risk customer.

### 7.9 Deviations from the Submitted SRS, SAD and Test Plan

The submitted documents described the first release: a command line training script. The team kept every one of those requirements and extended the product during the sprints. The main enhancements are:

1. A Streamlit web application, which SAD v1.0 listed as future work.
2. Real-time single-customer prediction with risk tiers, also listed as future work.
3. Field-level input validation, live evaluation with threshold tuning, batch scoring, and the what-if simulator.
4. In Sprint 3, deep learning and quantum machine learning models compared with XGBoost and shown side by side in the application.
5. DVC and MLflow for reproducible, tracked experiments.
6. Full test automation with coverage and a code quality gate in CI.

Each change is listed with its reason and benefit in [docs/DEVIATIONS_AND_ENHANCEMENTS.md](docs/DEVIATIONS_AND_ENHANCEMENTS.md). The new requirements are specified in [SRS v1.1](docs/SRS_v1.1_Addendum.md), designed in [SAD v1.1](docs/SAD_v1.1_Addendum.md) and verified by [Test Plan v1.1](docs/TEST_PLAN_v1.1.md). The production XGBoost model and its class weighting were kept exactly as specified.

### 7.10 Part 2 Status Summary

| Part 2 requirement | Status |
|---|---|
| Project created and linked to the repository | Done: GitHub Projects board |
| Backlog created from the SRS FRs and NFRs | Done: 11 issues for Sprints 1 and 2, traced in `docs/TRACEABILITY_MATRIX.md` |
| Story points assigned | Done: 49 points across Sprints 1 and 2 |
| Stories assigned to team members | Done |
| Two sprints planned and completed | Done |
| GitHub Actions enabled | Done |
| Pull-request workflow | Done |
| Sprint 1 and Sprint 2 videos uploaded | Done: `docs/evidence/videos/` |
| Deviations documented | Done: `docs/DEVIATIONS_AND_ENHANCEMENTS.md` |
| Documentation complete | Done |

---

## 8. Sprint 3: ML vs DML vs QML and MLOps

### 8.1 Goal

Compare three machine learning paradigms on the churn problem under identical conditions, show their predictions side by side in the application, and make every experiment reproducible and tracked.

### 8.2 Backlog

| ID | Backlog item | Story points |
|---|---|---|
| S3-1 | DVC pipeline and MLflow experiment tracking | 5 |
| S3-2 | Code quality refactor and pylint configuration | 3 |
| S3-3 | Deep learning (DML) churn model | 5 |
| S3-4 | Quantum machine learning (QML) churn model | 8 |
| S3-5 | ML vs DML vs QML benchmark and side-by-side predictions | 5 |
| S3-6 | Updated test cases and automation of TC-01 to TC-26 | 5 |
| S3-7 | Unit, system validation, code coverage and branch coverage reports | 3 |
| S3-8 | CI upgrade: lint, coverage gate and report artifacts | 3 |
| S3-9 | SRS, SAD and Test Plan v1.1, traceability and deviations | 5 |
| S3-10 | README, project report and repository tidy-up | 2 |
| **Total** | | **44** |

### 8.3 The three paradigms

| Paradigm | Model | Design |
|---|---|---|
| ML | XGBoost (and Logistic Regression as a baseline) | Gradient boosted trees with class weighting and stratified grid search |
| DML | Multi-layer perceptron (PyTorch) | One-hot and standardised inputs, two hidden layers (64 and 32) with batch normalisation and dropout, class-weighted loss, early stopping |
| QML | Variational quantum classifier (PennyLane) | Inputs reduced to 6 features with PCA and encoded as qubit rotation angles, 3 entangling layers with data re-uploading, Pauli-Z measurements and a linear output; 61 trainable parameters |

All three are trained on the same stratified split and corrected for class imbalance in the same way, and the deep and quantum models follow the scikit-learn interface, so the same evaluation and logging code measures all of them.

### 8.4 Results

| Paradigm | Model | Accuracy | Recall (churn) | F1 (churn) | ROC-AUC | Trainable parameters | Training time |
|---|---|---|---|---|---|---|---|
| ML | XGBoost | 75.66% | 79.41% | 0.634 | 0.846 | 5,628 tree nodes | 6.0 s |
| DML | Deep MLP | 73.24% | 79.68% | 0.613 | 0.841 | 5,249 | 2.4 s |
| ML | Logistic Regression | 73.67% | 79.41% | 0.616 | 0.839 | 20 | 0.2 s |
| QML | Quantum VQC | 73.74% | 75.40% | 0.604 | 0.829 | 61 | 44.4 s |

Findings:
1. On tabular churn data the three paradigms reach very similar quality (ROC-AUC between 0.83 and 0.85), which confirms XGBoost as a sound production choice.
2. The deep network has the highest churn recall (79.7%).
3. The quantum classifier comes within 0.02 ROC-AUC of XGBoost with only 61 trainable parameters, about one hundredth of the deep network, which shows how compact a quantum model can be.
4. Showing all three in the application makes uncertainty visible: for the borderline sample customer the models disagree (49.7%, 65.3% and 43.7%), which tells the analyst that this customer needs a closer look.

### 8.5 MLOps

- **DVC:** `dvc.yaml` defines two stages, `train` and `benchmark_paradigms`. All settings are in `params.yaml`, and `dvc.lock` records the exact inputs of each result.
- **MLflow:** each model run logs its parameters, metrics, plots, trainable parameter count, training time and the model itself, in the experiments Customer-Churn-Prediction and Customer-Churn-Paradigms (UI on port 5001).
- **Code quality:** the training code was split into shared modules (`training_common.py`, `classical_models.py`), and the repository scores 9.8/10 in pylint.

---

## 9. Testing and Quality (Deliverable Part 3)

### 9.1 Test case update

Test Plan v1.1 ([docs/TEST_PLAN_v1.1.md](docs/TEST_PLAN_v1.1.md)) keeps the 13 original test cases, automates all of them, and adds 13 new test cases for the features built in the sprints, for a total of 26. Every automated test is linked to its test case in `traceability.py`.

### 9.2 Results

| Measure | Target | Result |
|---|---|---|
| Automated tests passed | All | 152 of 152 |
| Test cases passed | All | 26 of 26 |
| Requirements validated | All | All requirements of SRS v1.0 and v1.1 |
| Statement coverage (application modules) | 85% | 89.1% |
| Branch coverage (application modules) | 75% | 79.4% |
| pylint score | 8.5/10 | 9.8/10 |
| Full training run time (NFR-01) | under 300 s | 15.3 s |
| Single prediction latency (NFR-11) | under 200 ms | 11.3 ms |

### 9.3 Test reports

The four reports required for Part 3 are generated from one run of the suite by `python tools/generate_test_reports.py` and stored in [docs/test-reports/](docs/test-reports/):

1. [Unit Test Report](docs/test-reports/UNIT_TEST_REPORT.md)
2. [System Validation Report](docs/test-reports/SYSTEM_VALIDATION_REPORT.md)
3. [Code Coverage Report](docs/test-reports/CODE_COVERAGE_REPORT.md)
4. [Branch Coverage Report](docs/test-reports/BRANCH_COVERAGE_REPORT.md)
