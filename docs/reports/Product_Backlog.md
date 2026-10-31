# Product Backlog and Traceability Matrix

Project: Customer Churn Prediction Model
Team: Hackforge (Akshaya B, Ashwin C H, Amar Patil)

This document links each backlog item to the SRS requirements it delivers and the test cases that verify it. The requirement and test case IDs are those of SRS v1.1 and Test Plan v1.1. The links from test cases to individual automated tests are kept in `traceability.py` and printed in `docs/reports/test-reports/SYSTEM_VALIDATION_REPORT.md`.

## 1. Backlog to Requirements to Test Cases

The backlog items for Sprints 1 and 2 were written with short SRS references (for example SRS-FR-01). The table maps each item to the detailed requirement IDs of SRS v1.1.

| Backlog item | Title | Sprint | Story points | SRS v1.1 requirements | Test cases |
|---|---|---|---|---|---|
| #1 | Interactive Customer Profile Input Form with Presets | 1 | 5 | FR-15, UI-02 | TC-14 |
| #2 | Real-time XGBoost Prediction Engine and Probability Visualizer | 1 | 5 | FR-14, FR-17 | TC-09, TC-16 |
| #3 | Robust Input Validation and Boundary Guards | 1 | 3 | FR-16, NFR-08, SR-05 | TC-15 |
| #13 | Bright Theme Redesign and One-Click Persona Preset Cards | 1 | 5 | FR-15 | TC-14 |
| #14 | Real-Time Risk Gauge, Key Driver Breakdown and Boundary Feedback | 1 | 5 | FR-17, FR-20 | TC-16, TC-19 |
| #4 | Model Performance Benchmarking and Evaluation | 2 | 5 | FR-07, FR-08, FR-18 | TC-06, TC-17 |
| #5 | Automated Tests for Validation and Prediction Pipeline | 2 | 5 | NFR-02, NFR-13 | TC-02, TC-15, TC-26 |
| #6 | GitHub Actions CI Workflow for Automated Verification | 2 | 3 | NFR-06, NFR-13 | TC-10, TC-26 |
| #15 | Interactive What-If Retention Strategy Simulator | 2 | 5 | FR-20 | TC-19 |
| #16 | Decision Threshold Tuning, Confusion Matrix and Batch CSV Scoring | 2 | 5 | FR-18, FR-19 | TC-17, TC-18 |
| #17 | Final Verification, Documentation Freeze and Release | 2 | 3 | FR-13, NFR-05 | TC-12 |
| S3-1 | DVC pipeline and MLflow experiment tracking | 3 | 5 | NFR-12, SW-03, SR-06 | TC-23 |
| S3-2 | Code quality refactor and pylint configuration | 3 | 3 | NFR-05, NFR-13 | TC-12, TC-26 |
| S3-3 | Deep learning (DML) churn model | 3 | 5 | FR-22 | TC-21 |
| S3-4 | Quantum machine learning (QML) churn model | 3 | 8 | FR-23 | TC-22 |
| S3-5 | ML vs DML vs QML benchmark and side-by-side predictions | 3 | 5 | FR-21, FR-24, SR-07 | TC-20, TC-23 |
| S3-6 | Updated test cases and automation of TC-01 to TC-26 | 3 | 5 | All | TC-01 to TC-26 |
| S3-7 | Unit, system validation, code coverage and branch coverage reports | 3 | 3 | NFR-13 | TC-26 |
| S3-8 | CI upgrade: lint, coverage gate and report artifacts | 3 | 3 | NFR-06, NFR-13 | TC-10, TC-26 |
| S3-9 | SRS, SAD and Test Plan v1.1, traceability and deviations | 3 | 5 | FR-13 | TC-12 |
| S3-10 | README, project report and repository tidy-up | 3 | 2 | FR-13, SR-02 | TC-11, TC-12 |

## 2. Story Points by Sprint

| Sprint | Dates | Backlog items | Story points |
|---|---|---|---|
| Sprint 1 | 19 to 23 October 2026 | #1, #2, #3, #13, #14 | 23 |
| Sprint 2 | 26 to 30 October 2026 | #4, #5, #6, #15, #16, #17 | 26 |
| Sprint 3 | 2 to 6 November 2026 | S3-1 to S3-10 | 44 |
| Total | | 21 items | 93 |

## 3. Requirements Coverage

Every requirement in SRS v1.0 and v1.1 is verified by at least one test case:

| Requirement group | Requirements | Verified |
|---|---|---|
| Functional | FR-01 to FR-24 | 24 of 24 |
| Non-functional | NFR-01 to NFR-13 | 13 of 13 |
| Interface | UI-01, UI-02, SW-01 to SW-03, HW-01, COM-01 | 7 of 7 |
| Security objectives | SO-01 to SO-03 | 3 of 3 |
| Security requirements | SR-01 to SR-07 | 7 of 7 |

The requirement-by-requirement matrix is in section 12 of `docs/TEST_PLAN_v1.1.md` and section 4 of the System Validation Report.
