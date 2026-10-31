# Software Test Plan and Test Cases, Version 1.1

Project: Customer Churn Prediction Model
Team: Hackforge (Akshaya B, Ashwin C H, Amar Patil)
Course: Software Engineering Mini Project, PES University
Base document: Test Plan v1.0, Part C of `docs/deliverables/Hackforge_Customer_Churn_Deliverable1.docx`

## 1. Introduction

### 1.1 Test plan identifier

| Field | Value |
|---|---|
| Identifier | STP-CCP-1.1 |
| Project | Customer Churn Prediction Model |
| Version | 1.1 (updated after Sprints 1 to 3) |
| Related documents | SRS v1.1 Addendum, SAD v1.1 Addendum, test reports in `docs/reports/test-reports/` |
| Test environment | macOS laptop (Python 3.12) and GitHub Actions Ubuntu runner (Python 3.12) |

### 1.2 Changes from version 1.0

1. All 13 original test cases (TC-01 to TC-13) are now automated and have been executed. Their status changed from Not Run to Pass.
2. Thirteen test cases (TC-14 to TC-26) were added for the features built during the sprints: the web application, input validation, batch scoring, the what-if simulator, the three-paradigm comparison, the deep and quantum models, the benchmark, prediction latency, model integrity and code quality.
3. Every automated test is tagged with the test case it executes, so the test reports show results per test case and per requirement.
4. Expected results that depended on the old command line output were updated to check the saved files and the MLflow records instead. TC-05 now checks that every combination in the `params.yaml` grid is evaluated, because the grid was made configurable during Sprint 3.

### 1.3 Purpose and scope

This plan checks that the delivered system meets every requirement in SRS v1.0 and v1.1: the training pipeline, the web application, batch scoring, explanation and simulation, the ML vs DML vs QML comparison, and the non-functional, interface and security requirements.

## 2. Test Items

| Test item | Files |
|---|---|
| Prediction service | `prediction_pipeline.py`, `customer_presets.py` |
| Web application | `app.py` |
| Training pipeline | `01_churn_prediction_xgboost.py`, `training_common.py`, `classical_models.py` |
| Deep and quantum models | `dml_model.py`, `qml_model.py` |
| Paradigm benchmark | `02_paradigm_benchmark.py` |
| Configuration and pipeline | `params.yaml`, `dvc.yaml`, `requirements*.txt`, `.gitignore` |
| Documentation | `README.md`, `DEVELOPER_GUIDE.md`, `docs/` |

## 3. Features to be Tested

| Feature | Requirements | Test cases |
|---|---|---|
| Loading the dataset and checking input | FR-01, FR-02, NFR-02, NFR-10, SW-01, SO-03 | TC-01, TC-02 |
| Target separation and encoding | FR-03, FR-04 | TC-03 |
| Training and class imbalance | FR-05, FR-10 | TC-04, TC-07 |
| Grid search and stratified cross validation | FR-06, FR-11 | TC-05 |
| Metrics output | FR-07, FR-08, FR-12, NFR-04, UI-01 | TC-06 |
| Saving and reloading files | FR-09, FR-14, NFR-07, SW-02, SR-03, SO-02 | TC-08, TC-09, TC-25 |
| Portability and runtime | NFR-01, NFR-06, HW-01 | TC-10 |
| Privacy and repository hygiene | NFR-09, SR-01, SR-02, SR-04, SO-01, COM-01 | TC-11 |
| Documentation | FR-13, NFR-05, NFR-08 | TC-12 |
| Repeatability | NFR-03 | TC-13 |
| Web form and presets | FR-15, UI-02 | TC-14 |
| Input validation | FR-16, NFR-08, SR-05 | TC-15 |
| Single prediction | FR-17 | TC-16 |
| Live evaluation and threshold | FR-18 | TC-17 |
| Batch scoring | FR-19 | TC-18 |
| Explanation and what-if simulation | FR-20 | TC-19 |
| Three-paradigm comparison | FR-21, SR-07 | TC-20 |
| Deep learning model | FR-22 | TC-21 |
| Quantum model | FR-23 | TC-22 |
| Benchmark and experiment tracking | FR-24, NFR-12, SR-06, SW-03 | TC-23 |
| Prediction latency | NFR-11 | TC-24 |
| Code quality | NFR-13 | TC-26 |

## 4. Features Not to be Tested

1. Whether a flagged customer will really churn in the future.
2. Retention campaigns and CRM connections, which are outside the system scope.
3. The internal correctness of third-party libraries (scikit-learn, XGBoost, PyTorch, PennyLane, Streamlit, MLflow, DVC). We test that we use them correctly.
4. Execution of the quantum model on real quantum hardware.
5. Formal security certification or legal compliance.

## 5. Approach

### 5.1 Security validation

| Check | How | Passes if | Test |
|---|---|---|---|
| Data minimisation | Inspect the model input columns after data preparation. | No identifier such as customer ID, name, email, phone number or address is a model input. | TC-11 |
| Repository hygiene | Check `.gitignore` and the list of tracked files. | Virtual environments, `.env` and `secrets.toml` are ignored and no key or secret file is tracked. | TC-11 |
| Local processing | Scan the source code for network client libraries. | No module imports a network client; MLflow uses a local store. | TC-11, TC-23 |
| Input handling | Use a missing file, a CSV without the target, invalid form values and unknown categories. | Each case stops with a clear error and no result is reported. | TC-02, TC-15 |
| Model file integrity | Reload the saved model and encoders; load paradigm models with wrong content. | Pairs load and predict; wrong files are rejected with a message. | TC-09, TC-20, TC-25 |

### 5.2 Test levels

| Level | What is tested | Tools |
|---|---|---|
| Unit | Validation rules, metric calculations, risk tiers, model classes | pytest |
| Integration | Data preparation with training, saving and reloading, preprocessing with prediction | pytest |
| System | Full training run, full benchmark run, the web application run headless | pytest, Streamlit AppTest |
| Non-functional | Runtime, latency, repeatability, portability, privacy, documentation, code quality | pytest, pylint, coverage.py |

### 5.3 Automation and continuous integration

All test cases are automated. `pytest` runs the whole suite. The training pipeline and the benchmark are run end to end inside the tests with every output file and the MLflow store redirected to a temporary folder, so the production model is never changed. GitHub Actions runs pylint and the full suite with coverage on every pull request and on every push to `main`.

### 5.4 Model evaluation steps

The steps of Test Plan v1.0 section 5.3 are unchanged. The production model is evaluated on the same stratified 20% split (random state 42, 1,409 customers) and its results are compared with the saved benchmark within a tolerance of 0.005.

### 5.5 Test data and environment

The project dataset is used for end-to-end tests. Small temporary CSV files are created for invalid input tests. The four sample customers in `customer_presets.py` are used for prediction tests. Library versions are recorded in the Unit Test Report.

## 6. Item Pass/Fail Criteria

A test case passes when all of its automated tests pass. The release passes when:

1. Every test case passes.
2. Combined statement and branch coverage of the application modules is at least 85%.
3. The pylint score is at least 8.5/10.

The severity levels of Test Plan v1.0 section 6 are unchanged.

## 7. Entry, Exit, Suspension and Resumption Criteria

| Entry criteria | Exit criteria |
|---|---|
| Code is merged into a feature branch, dependencies are installed, the dataset and production model are present. | All pass/fail criteria in section 6 are met and the test reports are generated. |

Testing is suspended if the dataset or production model is missing or if a privacy problem is found. After a fix, the full suite is run again.

## 8. Test Deliverables

| Deliverable | Location |
|---|---|
| Unit Test Report | `docs/reports/test-reports/UNIT_TEST_REPORT.md` |
| System Validation Report (includes the test execution record) | `docs/reports/test-reports/SYSTEM_VALIDATION_REPORT.md` |
| Code Coverage Report | `docs/reports/test-reports/CODE_COVERAGE_REPORT.md` |
| Branch Coverage Report | `docs/reports/test-reports/BRANCH_COVERAGE_REPORT.md` |
| Raw evidence (JUnit XML, Cobertura XML, coverage JSON, pytest and pylint output) | `docs/reports/test-reports/raw/` |

All reports are regenerated with `python tools/generate_test_reports.py`.

## 9. Responsibilities and Schedule

| Role | Responsibility |
|---|---|
| Developer | Write and automate test cases for each backlog item, keep the suite green. |
| Reviewer | Review pull requests, check that each new feature has a linked test case. |
| Instructor | Review the documents, reports and demonstration. |

| Activity | Dates |
|---|---|
| Test case update (Part 3, item 1) | Completed 30 October 2026 |
| Test execution and report generation | First complete run 30 October 2026; repeated on every pull request through the testing week of 2 to 6 November 2026 |

## 10. Risks and Contingencies

| Risk | Impact | Response |
|---|---|---|
| Library update changes model results | Saved benchmark no longer matches | TC-17 and TC-25 detect it; regenerate the model and encoders together. |
| Deep and quantum packages unavailable | DML and QML tests cannot run | Those tests skip cleanly and the core suite still runs; CI installs them in the quality job. |
| Training in tests overwrites the production model | Validated model lost | All test outputs go to temporary folders (checked by comparing file checksums before and after). |
| Documentation drifts from the code | Reports disagree with the system | Test cases and requirement links live in `traceability.py`, which drives the reports. |

## 11. Test Cases

### 11.1 Test case list

| ID | Name | Requirements | Steps | Expected result | Automated test(s) | Status |
|---|---|---|---|---|---|---|
| TC-01 | Load valid dataset | FR-01, SW-01 | Prepare the data with the project CSV. | 7,043 rows load; 1,409 go to the test set; all 19 feature columns are present. | `test_tc01_*` | Pass |
| TC-02 | Missing file / missing target handled | FR-02, NFR-02, NFR-10, SR-05, SO-03 | Use a missing path, a CSV without `Churn`, a bad config and a broken `params.yaml`; run the CLI with a missing file. | A clear error names the problem; the CLI exits with "Error:" and writes no model or metrics; a broken config falls back to defaults. | `test_tc02_*`, `test_missing_dataset_file`, `test_dataset_missing_columns` | Pass |
| TC-03 | Target separation and encoding | FR-03, FR-04 | Inspect the prepared features and encoders. | `Churn` is not an input; every input is numeric; an encoder exists for every text column. | `test_tc03_*` | Pass |
| TC-04 | XGBoost training with class weighting | FR-05, FR-10 | Train XGBoost on the training split. | Training finishes; `scale_pos_weight` equals the class ratio (about 2.77). | `test_tc04_*` | Pass |
| TC-05 | Grid search with stratified CV | FR-06, FR-11 | Build and run the grid search; read the best parameters from MLflow. | Stratified folds with shuffling are used; every grid combination is evaluated; the best value of each parameter is recorded. | `test_tc05_*` | Pass |
| TC-06 | Evaluation metrics and reports | FR-07, FR-08, FR-12, NFR-04, UI-01 | Run the full training pipeline. | Accuracy, precision, recall, F1 and ROC-AUC are saved; the confusion matrix sums to 1,409; the classification report and plots are written; a run is logged for each model. | `test_tc06_*`, metric tests in `test_pipeline.py` | Pass |
| TC-07 | Class imbalance handling | FR-10 | Compare class ratios; read the logged weights. | The test set keeps the natural churn ratio; class weighting is logged for each model. | `test_tc07_*`, `test_holdout_split_is_the_training_test_split` | Pass |
| TC-08 | Save files | FR-09, SW-02 | Run the full pipeline. | The model and encoder files exist and are not empty. | `test_tc08_*` | Pass |
| TC-09 | Reload saved files | FR-14, NFR-07, SR-03, SO-02 | Load the saved pair and predict for a sample customer; load with a missing file. | The pair loads and gives a probability between 0 and 1; a missing file gives a clear error. | `test_tc09_*`, `test_load_artifacts`, `test_missing_*_file_handling` | Pass |
| TC-10 | Fresh environment and runtime | NFR-01, NFR-06, HW-01 | Compare imports with the requirements files; time a full training run on CPU. | Every third-party import is declared; the run takes under 300 s (measured 15.3 s). | `test_tc10_*`, `test_imports` | Pass |
| TC-11 | Privacy and repository hygiene | NFR-09, SR-01, SR-02, SR-04, SO-01, COM-01 | Check model inputs, `.gitignore`, tracked files and imports. | No identifiers are inputs; secrets and environments are ignored; no secret files are tracked; no network clients are used. | `test_tc11_*`, `test_customer_id_is_dropped_if_present` | Pass |
| TC-12 | Documentation check | FR-13, NFR-05, NFR-08 | Check the README sections and the module docstrings. | The README covers installation, the web app, training, model performance, limitations and tests; every module documents its purpose. | `test_tc12_*` | Pass |
| TC-13 | Repeatability | NFR-03 | Prepare the data twice; train twice with the same seed. | The split and the predicted probabilities are identical. | `test_tc13_*` | Pass |
| TC-14 | Web form and presets | FR-15, UI-02 | Start the app headless; load presets; switch pages. | The app starts with the default preset; every field has a label and help; presets fill the form; values survive page switches. | 11 tests in `test_app.py` and `test_pipeline.py` | Pass |
| TC-15 | Input validation | FR-16, NFR-08, SR-05 | Submit blank, out-of-range, wrong-type and inconsistent values; validate every dataset row. | Each invalid case is reported with the field name and no prediction is made; all 7,043 real rows pass. | 54 tests in `test_pipeline.py` and `test_app.py` | Pass |
| TC-16 | Single prediction | FR-17 | Predict for valid customers and the four presets. | Class, probability and risk tier are returned; each preset's risk label matches the model. | 8 tests | Pass |
| TC-17 | Live evaluation and threshold | FR-18 | Re-evaluate the model in the app and in Python at several thresholds. | Live metrics match the saved benchmark; lowering the threshold raises recall. | 7 tests | Pass |
| TC-18 | Batch scoring | FR-19 | Score the sample batch at two thresholds. | Every customer gets a probability, class and risk tier, sorted by risk. | `test_score_customer_batch_*` | Pass |
| TC-19 | Explanation and what-if | FR-20 | Explain a high-risk customer; simulate a contract upgrade. | Risk drivers are listed; the simulated churn probability is lower. | `test_explain_customer_risk_attribution`, `test_what_if_counterfactual_risk_reduction` | Pass |
| TC-20 | Three-paradigm comparison | FR-21, SR-07 | Predict in the app; score one customer with all models; load invalid model files. | Three cards (ML, DML, QML) appear for valid input and none for invalid input; wrong files are rejected. | 6 tests in `test_app.py` and `test_paradigms.py` | Pass |
| TC-21 | Deep learning model | FR-22 | Train, pickle, reload and clone the MLP. | Probabilities are valid; reloaded predictions are identical; the best epoch is restored; the parameter count matches the architecture. | 9 tests in `test_paradigms.py` | Pass |
| TC-22 | Quantum model | FR-23 | Train, pickle and reload the VQC; check a one-qubit circuit against theory. | Probabilities are valid; reloaded predictions are identical; inputs are valid rotation angles; the circuit output equals the analytic cosine. | 10 tests in `test_paradigms.py` | Pass |
| TC-23 | Benchmark and tracking | FR-24, NFR-12, SR-06, SW-03 | Run the benchmark end to end with a local MLflow store; run it with a bad config. | One run per model tagged ML, DML or QML; leaderboard sorted by ROC-AUC; serving models reload; bad config stops with a message. | `test_tc23_*` and 3 tests in `test_paradigms.py` | Pass |
| TC-24 | Prediction latency | NFR-11 | Time 30 single predictions after a warm-up. | Median under 200 ms (measured 11.3 ms). | `test_tc24_*` | Pass |
| TC-25 | Production model integrity | NFR-07, SO-02 | Load the production model and encoders. | Feature order matches training; encoders cover every text column and match the allowed values. | `test_tc25_*`, `test_valid_categories_match_fitted_encoders` | Pass |
| TC-26 | Code quality | NFR-13 | Run pylint and coverage. | pylint at least 8.5/10 (measured 9.82); coverage at least 85%. | CI quality job, `tools/generate_test_reports.py` | Pass |

### 11.2 Test execution record

The full execution record, with the number of automated tests per test case, measured values and pass/fail status, is in `docs/reports/test-reports/SYSTEM_VALIDATION_REPORT.md`. Summary of the run on 30 October 2026:

| Test cases | Passed | Failed | Automated tests | Passed | Failed |
|---|---|---|---|---|---|
| 26 | 26 | 0 | 152 | 152 | 0 |

## 12. Traceability Matrix (SRS to Test Cases)

| Requirement | Test case(s) | Requirement | Test case(s) |
|---|---|---|---|
| FR-01 | TC-01 | NFR-01 | TC-10 |
| FR-02 | TC-02 | NFR-02 | TC-02 |
| FR-03 | TC-03 | NFR-03 | TC-13 |
| FR-04 | TC-03 | NFR-04 | TC-06 |
| FR-05 | TC-04 | NFR-05 | TC-12 |
| FR-06 | TC-05 | NFR-06 | TC-10 |
| FR-07 | TC-06 | NFR-07 | TC-09, TC-25 |
| FR-08 | TC-06 | NFR-08 | TC-12, TC-15 |
| FR-09 | TC-08 | NFR-09 | TC-11 |
| FR-10 | TC-04, TC-07 | NFR-10 | TC-02 |
| FR-11 | TC-05 | NFR-11 | TC-24 |
| FR-12 | TC-06 | NFR-12 | TC-23 |
| FR-13 | TC-12 | NFR-13 | TC-26 |
| FR-14 | TC-09 | UI-01 | TC-06 |
| FR-15 | TC-14 | UI-02 | TC-14 |
| FR-16 | TC-15 | SW-01 | TC-01 |
| FR-17 | TC-16 | SW-02 | TC-08 |
| FR-18 | TC-17 | SW-03 | TC-23 |
| FR-19 | TC-18 | HW-01 | TC-10 |
| FR-20 | TC-19 | COM-01 | TC-11 |
| FR-21 | TC-20 | SO-01 | TC-11 |
| FR-22 | TC-21 | SO-02 | TC-09, TC-25 |
| FR-23 | TC-22 | SO-03 | TC-02 |
| FR-24 | TC-23 | SR-01 to SR-04 | TC-11, TC-09 (SR-03) |
| | | SR-05 | TC-02, TC-15 |
| | | SR-06 | TC-23 |
| | | SR-07 | TC-20 |

The same matrix, with the individual automated tests for each test case, is generated in the System Validation Report from `traceability.py`.
