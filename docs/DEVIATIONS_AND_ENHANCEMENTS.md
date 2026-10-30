# Deviations from the Submitted Documents

Project: Customer Churn Prediction Model
Team: Hackforge (Akshaya B, Ashwin C H, Amar Patil)
Documents compared: SRS v1.0, SAD v1.0 and Test Plan v1.0 (Deliverable 1, 12 October 2026)

## 1. Summary

The submitted documents described the first release of the project: a command line script that trains, evaluates and saves an XGBoost churn model. During the three sprints the team kept every one of those requirements and built on top of them, turning the script into a complete decision support system. Every change below adds capability; nothing in the original scope was removed or weakened.

All of the changes are now specified in SRS v1.1, designed in SAD v1.1, and verified by Test Plan v1.1 (26 test cases, all passed).

| Area | Submitted documents | Delivered system |
|---|---|---|
| User interface | Terminal output only | Streamlit web application with four pages |
| Prediction | Planned as future work (SAD 3.9) | Real-time single and batch prediction with risk tiers |
| Models | XGBoost | XGBoost, plus a deep learning model and a quantum machine learning model compared side by side |
| Reproducibility | Fixed seeds | Fixed seeds, DVC pipeline, MLflow experiment tracking |
| Testing | 13 manual test cases, Not Run | 26 automated test cases, 152 automated tests, all passing, with coverage reports |
| Quality checks | None planned | pylint 8.5 threshold and 85% coverage threshold enforced in CI |

## 2. Product Enhancements

| No. | Submitted document said | What was delivered | Reason and benefit | Sprint | Requirements |
|---|---|---|---|---|---|
| E1 | "It is not a web app." Dashboard listed as future work (SRS 2.1, 2.6; SAD 3.9). | Streamlit web application with a guided customer form, four one-click sample customers and four pages. | The intended users (retention analysts) do not use a terminal. A web interface made the model usable and demonstrable. | 1 | FR-15, UI-02 |
| E2 | Prediction feature listed as future work (SAD 3.9). | Real-time prediction with churn probability, risk tier (Low, Moderate, High) and a risk gauge. | Turns the trained model into a decision support tool, which was the original aim of the project. | 1 | FR-17 |
| E3 | Input checks limited to the CSV file (FR-02). | Field-level validation of every input: ranges, allowed values and service consistency rules, with every error listed. | Prevents meaningless predictions from impossible customer profiles. | 1 | FR-16 |
| E4 | Fixed 0.5 threshold, metrics printed once. | Live re-evaluation on the held-out set with an adjustable decision threshold and confusion matrix. | Lets the business choose its own balance between catching churners and false alarms. | 2 | FR-18 |
| E5 | Not planned. | Batch scoring of a CSV file with a downloadable, risk-sorted outreach list. | Retention teams work with lists of customers, not one at a time. | 2 | FR-19 |
| E6 | "Explaining exactly why they did" was out of scope (SRS 2.6). | Plain-language risk drivers and a what-if simulator for retention actions. | Shows the analyst which action is most likely to keep a customer. Presented as guidance, not as a guarantee. | 2 | FR-20 |
| E7 | XGBoost is the only model (SRS 3.6). | Classical ML, deep learning (PyTorch MLP) and quantum ML (PennyLane variational quantum classifier) trained on the same split; all three predictions shown side by side for each customer. | Gives an evidence-based answer to which paradigm suits tabular churn data. XGBoost remains the production model. | 3 | FR-21 to FR-24 |
| E8 | Grid search with 36 combinations (SRS 4.3). | The grid is defined in `params.yaml` and can be changed without code; the default grid has 8 combinations and a full training run takes under 20 seconds. | Faster experiments and settings kept under version control. The tuned production model is unchanged. | 3 | FR-06, NFR-12 |

## 3. Engineering Enhancements

| No. | Submitted document said | What was delivered | Benefit | Requirements |
|---|---|---|---|---|
| E9 | Results recorded by hand in an experiment template (Test Plan Appendix A). | MLflow records the parameters, metrics, plots and model for every run automatically. | No transcription errors; any result can be traced to its settings. | NFR-12, SW-03 |
| E10 | Reproducibility through fixed seeds (NFR-03). | Seeds plus a DVC pipeline (`dvc.yaml`, `params.yaml`, `dvc.lock`) and a DVC-tracked dataset. | One command (`dvc repro`) rebuilds the results. | NFR-03, NFR-12 |
| E11 | One training script (SAD 2.1). | Shared modules for data preparation, metrics and logging, used by both training scripts and the tests. | Every model is measured in exactly the same way; less duplicated code. | NFR-05 |
| E12 | All tests manual and Not Run (Test Plan 11). | Every test case automated and run on every pull request through GitHub Actions. | Regressions are caught before merging. | NFR-13 |
| E13 | No code quality target. | pylint score of 9.8/10 and coverage above 85%, both enforced in CI. | Measurable maintainability. | NFR-13 |
| E14 | NFR-01 and NFR-08 had no numeric target. | Targets added in SRS v1.1: training under 300 s, every field labelled and every error naming its field. Both are measured. | Makes the requirements testable, as the IEEE format asks. | NFR-01, NFR-08 |

## 4. Changes to the Test Plan

1. TC-01 to TC-13 keep their purpose. Steps that relied on reading terminal output now check the saved files and MLflow records, which is stricter.
2. TC-05 expected "36 combinations are tried". It now checks that every combination of the configured grid is evaluated, which stays correct if the grid changes.
3. TC-14 to TC-26 were added for the new features and the new non-functional requirements.
4. All test cases now have the status Pass, recorded in `docs/test-reports/SYSTEM_VALIDATION_REPORT.md`.

## 5. Design Decisions Kept from Deliverable 1

1. XGBoost with class weighting (`scale_pos_weight` about 2.77) is still the production model.
2. The stratified 20% test split with random state 42 (1,409 customers) is still used for every evaluation, so all results remain comparable.
3. The model and its encoders are still saved and loaded as one pair (NFR-07), and this is now tested automatically.
4. All processing is still local; no customer data leaves the machine (COM-01, SR-04).
