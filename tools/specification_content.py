"""Content of the specification set: requirement statements and the sequence diagrams of Part B.

Kept separate from tools/build_specification.py so the wording can be reviewed and edited on its own.
"""

from typing import Dict, Tuple

# ----- requirements (statement, IEEE 830 rank) -------------------------------------------------
REQUIREMENTS: Dict[str, Tuple[str, str]] = {
    "UI-01": ("Every evaluation output shall label each metric by name and identify churn as the positive class.",
              "Essential"),
    "UI-02": ("The system shall provide a browser-based user interface at http://localhost:8501 with four pages: "
              "Customer Churn Prediction, Batch Customer Scoring, Model Evaluation and About the Project.",
              "Essential"),
    "HW-01": ("The system shall run on a standard laptop or desktop CPU and shall not require a GPU.", "Conditional"),
    "SW-01": ("The system shall read the customer dataset from a CSV file and shall report a descriptive error "
              "when the file is missing or cannot be read.", "Essential"),
    "SW-02": ("The system shall persist the trained model and its encoders to the configured file paths.",
              "Essential"),
    "SW-03": ("The system shall record training and benchmark runs in a local MLflow tracking store that can be "
              "browsed through the MLflow user interface on port 5001.", "Conditional"),
    "COM-01": ("The system shall process all customer data locally and shall not transmit customer data to any "
               "external service.", "Essential"),
    "FR-01": ("The system shall load the Telco Customer Churn dataset from the configured CSV path.", "Essential"),
    "FR-02": ("If the dataset is missing, unreadable or lacks a required column, the system shall stop the operation "
              "with an error message that identifies the problem and shall not produce model, metric or report "
              "outputs.", "Essential"),
    "FR-03": ("The system shall separate the target attribute (Churn) from the input attributes before training.",
              "Essential"),
    "FR-04": ("The system shall encode every categorical attribute numerically with encoders that are fitted on the "
              "dataset and persisted with the model.", "Essential"),
    "FR-05": ("The system shall train a gradient-boosted decision tree classifier (XGBoost) on the training "
              "partition.", "Essential"),
    "FR-06": ("The system shall select hyperparameters by exhaustive grid search with stratified k-fold "
              "cross-validation (default k = 3) using the training partition only.", "Essential"),
    "FR-07": ("The system shall compute accuracy, precision, recall, F1-score and ROC-AUC for the churn class on the "
              "held-out test partition and record them in metrics.json and the experiment store.", "Essential"),
    "FR-08": ("The system shall produce a confusion matrix for the held-out test partition.", "Essential"),
    "FR-09": ("On successful completion of training, the system shall persist the selected model together with its "
              "encoders.", "Essential"),
    "FR-10": ("The system shall correct class imbalance by weighting the minority class during training and shall "
              "not resample the test partition.", "Essential"),
    "FR-11": ("The system shall report the best hyperparameter combination selected by the grid search.",
              "Conditional"),
    "FR-12": ("The system shall present training configuration and evaluation results as separate, labelled "
              "outputs.", "Essential"),
    "FR-13": ("The project documentation shall describe the dataset, the models, the evaluation metrics, "
              "installation and operation, and the known limitations.", "Essential"),
    "FR-14": ("The system shall reload a persisted model with its encoders and use them to score new customer "
              "records.", "Conditional"),
    "FR-15": ("The system shall provide a customer profile form containing all 19 model inputs, grouped into "
              "Demographics, Services and Contract and Billing, each with a label and help text, and at least four "
              "one-click sample customer profiles.", "Essential"),
    "FR-16": ("Before predicting, the system shall validate required fields, permitted categorical values, numeric "
              "ranges and service-consistency rules (Section 3.4). If any check fails, the system shall list every "
              "error and shall not display a prediction.", "Essential"),
    "FR-17": ("For a valid profile, the system shall display the predicted class, the churn probability as a "
              "percentage and the risk tier.", "Essential"),
    "FR-18": ("On request, the system shall re-evaluate the production model on the 1,409-customer held-out test "
              "partition and display accuracy, precision, recall, F1-score, specificity, ROC-AUC and the confusion "
              "matrix at a user-selected decision threshold between 0.05 and 0.95.", "Essential"),
    "FR-19": ("The system shall score a batch of customers (an uploaded CSV file or the 100-customer sample) in one "
              "action, return each customer's churn probability, predicted class and risk tier, and offer the "
              "result as a CSV download.", "Conditional"),
    "FR-20": ("The system shall identify the main factors that raise or lower a customer's churn risk and shall let "
              "the user simulate retention actions and observe the resulting churn probability.", "Conditional"),
    "FR-21": ("For every valid profile, the system shall display side by side the churn probability, predicted class "
              "and risk tier produced by the classical (XGBoost), deep learning (MLP) and quantum (VQC) models "
              "using the same decision threshold.", "Essential"),
    "FR-22": ("The system shall provide a deep learning churn model (multi-layer perceptron) trained with class "
              "weighting and early stopping that can be persisted and reloaded with identical predictions.",
              "Essential"),
    "FR-23": ("The system shall provide a quantum machine learning churn model (variational quantum classifier, "
              "6 qubits, 3 layers, executed on a simulator) that can be persisted and reloaded with identical "
              "predictions.", "Essential"),
    "FR-24": ("The system shall train the ML, DML and QML models on the same stratified split, log each run's "
              "parameters, metrics, trainable-parameter count and training time to the experiment store, and write "
              "a leaderboard (paradigm_metrics.json) and a comparison chart.", "Essential"),
    "NFR-01": ("A complete training run with the configured grid (Logistic Regression, Random Forest and XGBoost) "
               "shall finish in less than 300 seconds on a four-core laptop CPU without a GPU.", "Essential"),
    "NFR-02": ("If required data or columns are missing, the operation shall fail with a descriptive error and "
               "shall never report fabricated metrics.", "Essential"),
    "NFR-03": ("Given the same data, random seed and library versions, repeated runs shall produce an identical "
               "train/test split and identical predictions.", "Essential"),
    "NFR-04": ("Evaluation results shall report churn-class recall and precision alongside accuracy.", "Essential"),
    "NFR-05": ("Data loading, preprocessing, training, evaluation and persistence shall be implemented as separate "
               "functions or modules.", "Essential"),
    "NFR-06": ("The system shall install from the published requirements files into a new Python environment and "
               "run without any developer-specific environment.", "Essential"),
    "NFR-07": ("The model file and the encoder file used together shall always originate from the same training "
               "run.", "Essential"),
    "NFR-08": ("Every input field shall have a label and help text, and every validation message shall name the "
               "field it refers to.", "Essential"),
    "NFR-09": ("Personal identifiers (for example customer ID, name, e-mail address or telephone number) shall not "
               "be used as model inputs.", "Essential"),
    "NFR-10": ("If a required library or file is unavailable, the system shall stop with a visible error instead of "
               "appearing to complete.", "Essential"),
    "NFR-11": ("The median time to validate, preprocess and predict one customer shall be less than 200 ms on a "
               "laptop CPU.", "Essential"),
    "NFR-12": ("Training and benchmarking shall be defined as DVC pipeline stages with all settings in params.yaml, "
               "and every run shall be recorded in MLflow with its parameters and metrics.", "Essential"),
    "NFR-13": ("The code base shall achieve a pylint score of at least 8.5/10 and at least 85% combined statement and "
               "branch coverage of the application modules; both shall be enforced by continuous integration.",
               "Essential"),
    "SO-01": ("Confidentiality: customer data shall remain on the user's machine and personal identifiers shall not "
              "be collected or modelled unless required.", "Essential"),
    "SO-02": ("Integrity: model and encoder artifacts shall not be mixed between training runs or replaced "
              "unintentionally.", "Essential"),
    "SO-03": ("Safe execution: input files shall be treated strictly as data; their content shall never be executed "
              "as code.", "Essential"),
    "SR-01": ("Data minimisation: personal identifiers and unused columns shall be removed from the model inputs.",
              "Essential"),
    "SR-02": ("Repository hygiene: credentials, private data and virtual environments shall not be committed to the "
              "public repository.", "Essential"),
    "SR-03": ("Artifact integrity: model and encoder files shall be regenerated and versioned together whenever the "
              "pipeline changes.", "Essential"),
    "SR-04": ("Local processing: the training and prediction pipelines shall not send data or predictions to any "
              "external service.", "Essential"),
    "SR-05": ("Input validation: missing files, missing target columns, invalid formats and invalid field values "
              "shall result in a controlled error.", "Essential"),
    "SR-06": ("Experiment tracking data (MLflow database and artifacts) shall be stored on the local machine only.",
              "Essential"),
    "SR-07": ("The application shall load deep and quantum models only from the project models/ folder and shall "
              "reject any model file whose format or input features differ from the production model.",
              "Essential"),
}

SEQ_TRAIN = """sequenceDiagram
actor S as Data scientist
participant T as Training script
participant C as training_common
participant M as Classical trainers
participant F as MLflow
participant D as File system
S->>T: run 01_churn_prediction_xgboost.py
T->>C: load_config(), prepare_data()
alt dataset missing or invalid
C-->>T: FileNotFoundError / ValueError
T-->>S: error message, no outputs written
else dataset valid
C-->>T: stratified train / test split
loop each model family
T->>M: train (grid search, class weighting)
M-->>T: fitted model and parameters
T->>F: log parameters, metrics, plots, model
end
T->>T: select champion by ROC-AUC
T->>D: save model, encoders, metrics.json
T-->>S: results summary
end"""

SEQ_PREDICT = """sequenceDiagram
actor U as Analyst
participant APP as app.py
participant PP as prediction_pipeline
participant X as XGBoost
participant N as Deep MLP
participant Q as Quantum VQC
U->>APP: submit profile (form or preset)
APP->>PP: validate_customer_input(record)
alt input invalid
PP-->>APP: list of field errors
APP-->>U: show every error, no prediction
else input valid
APP->>PP: preprocess_input(record)
PP-->>APP: encoded row (training order)
APP->>PP: predict_churn(model, row, threshold)
PP->>X: predict_proba(row)
PP-->>APP: class, probability, risk tier
APP->>PP: predict_all_paradigms(row, models)
PP->>N: predict_proba(row)
PP->>Q: predict_proba(row)
PP-->>APP: one result per paradigm
APP-->>U: prediction, gauge, drivers, three model cards
end"""

SEQ_BATCH = """sequenceDiagram
actor U as Analyst
participant APP as app.py
participant PP as prediction_pipeline
participant X as XGBoost
U->>APP: upload CSV or choose sample, set threshold
APP->>PP: score_customer_batch(table, threshold)
alt unknown categories or missing columns
PP-->>APP: ValueError naming the problem
APP-->>U: error message, batch not scored
else table valid
PP->>PP: set customerID aside, encode
PP->>X: predict_proba(table)
PP-->>APP: probability, class, risk tier per row
APP-->>U: risk-sorted list and CSV download
end"""

SEQ_BENCH = """sequenceDiagram
actor S as Data scientist
participant B as 02_paradigm_benchmark
participant C as training_common
participant M as ML, DML, QML trainers
participant F as MLflow
participant D as models folder
S->>B: dvc repro benchmark_paradigms
B->>C: load_config(), prepare_data()
C-->>B: one stratified split for all models
loop each paradigm in params.yaml
B->>M: train(split)
M-->>B: fitted model and parameters
B->>C: evaluate_and_log_run(model, split)
C->>F: log parameters, metrics, plots, model
B->>D: save serving model (DML, QML)
end
B-->>S: leaderboard and comparison chart"""
