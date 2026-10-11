# Project Report: Customer Churn Prediction System

**Author:** Ashwin  
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

This section records the software engineering evidence available in the project repository and its GitHub issue, pull-request, and Actions records. Items that cannot be established from those records are marked **Pending confirmation** rather than inferred.

### 7.1 GitHub Backlog and SRS Traceability

The GitHub backlog contains six issues. The SRS references below are taken from the issue descriptions.

| Issue | Backlog item | SRS reference | Sprint | GitHub status |
| :--- | :--- | :--- | :--- | :--- |
| [#1](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/1) | Interactive Customer Profile Input Form with Presets | SRS-FR-01 | Sprint 1 | Closed |
| [#2](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/2) | Real-time XGBoost Prediction Engine and Probability Visualizer | SRS-FR-02 | Sprint 1 | Closed |
| [#3](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/3) | Robust Input Validation and Boundary Guards | SRS-NFR-01 | Sprint 1 | Closed |
| [#4](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/4) | Model Performance Benchmarking and Evaluation | SRS-FR-03 | Sprint 2 | Closed |
| [#5](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/5) | Automated Tests for Validation and Prediction Pipeline | SRS-NFR-02 | Sprint 2 | Closed |
| [#6](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/6) | GitHub Actions CI Workflow for Automated Verification | SRS-NFR-03 | Sprint 2 | Closed |

The recorded issue state is **Closed** for all six items. This is a GitHub status, not independent confirmation that every acceptance criterion was met: the visible acceptance checklists in issues #1, #3, and #4 were not checked when inspected. Evidence or confirmation of those criteria is **Pending confirmation**.

### 7.2 Story Points and Issue Assignments

Story-point estimates, sprint assignments, and assignees are transcribed from the issue records. “Unassigned” means no assignee was listed in the inspected GitHub record; it does not identify who performed the work.

| Issue | Story points | Sprint | GitHub assignee |
| :--- | :---: | :--- | :--- |
| [#1](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/1) | 5 | Sprint 1 | ash29032006 |
| [#2](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/2) | 5 | Sprint 1 | Akshayshetty742 |
| [#3](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/3) | 3 | Sprint 1 | Unassigned |
| [#4](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/4) | 5 | Sprint 2 | ash29032006 |
| [#5](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/5) | 5 | Sprint 2 | Unassigned |
| [#6](https://github.com/Akshayshetty742/hackforge-customer-churn/issues/6) | 3 | Sprint 2 | Akshayshetty742 |

Individual contribution records beyond issue assignment are **Pending confirmation**.

For issue #3, the current GitHub issue record lists no assignee; it is **Unassigned**. An earlier team plan reportedly lists Amar. Reconciling that discrepancy is **Pending confirmation**; neither record establishes who completed the work.

### 7.3 Sprint 1 Plan and Outcomes

**Recorded plan:** Issues #1–#3 were assigned to Sprint 1: the interactive input form and presets (5 points), prediction engine (5 points), and input validation (3 points).

**Recorded issue state:** All three issues are closed in GitHub. Issue #2's acceptance checklist is checked. The visible checklists for issues #1 and #3 are unchecked, so closure alone does not verify their acceptance criteria. Specific sprint completion evidence, actual completed scope, carryovers, and retrospective findings are **Pending confirmation**.

### 7.4 Sprint 2 Plan and Outcomes

**Recorded plan:** Issues #4–#6 were assigned to Sprint 2: model evaluation (5 points), automated tests (5 points), and CI workflow (3 points).

**Recorded issue state:** All three issues are closed in GitHub. The visible acceptance checklists for issues #5 and #6 are checked; issue #4's checklist is unchecked. The repository contains evaluation and test code, and a successful CI run is recorded (see sections 7.6–7.7). These artifacts do not by themselves verify every acceptance criterion or establish sprint-level outcomes. Specific completion evidence, carryovers, and retrospective findings are **Pending confirmation**.

### 7.5 Pull-Request Workflow and Code Review

The repository records the following merged pull requests:

- [PR #7 — docs: demonstrate Deliverable 2 PR workflow](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/7)
- [PR #8 — Feature/issues 1 3 4 form validation eval](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/8)
- [PR #9 — feat(ml): add configurable churn threshold and risk tiers](https://github.com/Akshayshetty742/hackforge-customer-churn/pull/9)

The PR records document branches merged into `main`. The review records inspected for these PRs contained no submitted reviews or review threads. Therefore, merge status is not presented as evidence of an approval or substantive code review. Reviewer participation, review comments, and any required approval policy are **Pending confirmation**.

### 7.6 GitHub Actions CI Workflow

The [CI workflow](.github/workflows/ci.yml) is configured for pushes and pull requests targeting `main`. It checks out the repository, sets up Python 3.11 with pip caching, installs dependencies from `requirements.txt`, and runs `pytest -v`.

GitHub Actions records a successful run: [Customer Churn CI Pipeline, run #7](https://github.com/Akshayshetty742/hackforge-customer-churn/actions/runs/38156397673). The run completed with a **success** conclusion on `main`, following the merge of PR #9. This records the workflow result; specific job logs or test counts should be included only if separately captured.

### 7.7 Automated Testing and Verified Results

Automated tests are present in [`test_pipeline.py`](test_pipeline.py) and [`test_app.py`](test_app.py). The CI workflow invokes the test suite with `pytest -v`, and Actions run #7 completed successfully as described above.

PR #9's description reports that 16 tests passed locally. This is a PR-reported local result, distinct from the successful Actions run; it is not used here to infer a test count for run #7. A detailed test report, including per-test results and CI job output, is **Pending confirmation**.

### 7.8 Screenshots and Evidence

The evidence available for this report includes the linked GitHub issues and pull requests, the CI workflow file, and the successful Actions run. No screenshot files were found in the inspected repository. Screenshots or other captured evidence of the application, backlog, sprint records, pull-request/review activity, and CI/test output are **Pending confirmation**.

### 7.9 Deviations from the Submitted SRS, SAD, and Test Plan

The submitted SRS, Software Architecture Document (SAD), and Test Plan were not available among the inspected project documentation. No comparison has therefore been made, and no deviation is claimed. A document-by-document comparison, with identified changes, rationale, and requirement/design/test references, is **Pending confirmation** until those baselines are supplied.

### 7.10 Final Project Status and Development Freeze

At the time of inspection, GitHub showed issues #1–#6 closed, PRs #7–#9 merged, and CI run #7 successful on `main`. These records describe repository state only and do not establish formal project acceptance or a development freeze.

Final sign-off, release/acceptance evidence, and whether development has been frozen are **Pending confirmation**. No freeze date is stated because none was verified.
