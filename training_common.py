"""Shared training utilities: configuration, data preparation, evaluation, plots and MLflow logging.

Used by 01_churn_prediction_xgboost.py (classical models) and 02_paradigm_benchmark.py
(ML vs DML vs QML) so every model is trained and scored on the identical stratified split,
and cleaned with the same rules the prediction pipeline applies at serving time.
"""

from __future__ import annotations

import copy
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import prediction_pipeline as pp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(BASE_DIR, "params.yaml")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
EXPERIMENT_NAME = "Customer-Churn-Prediction"

# Used when params.yaml is missing; each top-level section in params.yaml replaces the default one.
DEFAULT_CONFIG: Dict[str, Any] = {
    "prepare": {
        "test_size": 0.2,
        "random_state": 42,
        "target_column": "Churn",
    },
    "model": {
        "type": "all",  # "all", "xgboost", "random_forest", "logistic_regression"
        "champion_metric": "roc_auc",
    },
    "train": {
        "cv_splits": 3,
        "scoring": "accuracy",
        "eval_metric": "logloss",
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 4],
            "learning_rate": [0.02, 0.05],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        },
    },
}


def configure_runtime() -> None:
    """Process-wide settings for CLI training runs (call once from a script's main)."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
    warnings.filterwarnings("ignore")
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_config(path: str = PARAMS_FILE) -> Dict[str, Any]:
    """Returns the default configuration overlaid with params.yaml."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not os.path.exists(path):
        return config
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        print(f"Warning: Could not parse '{path}' ({exc}), using default parameters.")
        return config
    if loaded:
        config.update(loaded)
    print(f"Loaded configuration from '{path}'.")
    return config


@dataclass(frozen=True)
class DataSplit:
    """Label-encoded train / test data from the stratified split every model shares."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    @property
    def feature_names(self) -> List[str]:
        """Model input columns in training order."""
        return self.X_train.columns.tolist()

    @property
    def scale_pos_weight(self) -> float:
        """Majority / minority ratio of the training labels (~2.77 for Telco)."""
        minority = int(self.y_train.sum())
        return (len(self.y_train) - minority) / minority if minority else 1.0


def prepare_data(
    config: Dict[str, Any],
    dataset_path: str = pp.DEFAULT_DATASET_FILE,
) -> Tuple[DataSplit, Dict[str, LabelEncoder]]:
    """Loads and cleans the dataset, label-encodes the text columns and splits it.

    Raises:
        FileNotFoundError: if the dataset is missing.
        ValueError: if required columns are missing.
    """
    prep = config["prepare"]
    target = str(prep.get("target_column", "Churn"))

    df = pp.load_dataset(dataset_path)
    encoders: Dict[str, LabelEncoder] = {}
    for col in pp.CATEGORICAL_COLUMNS + [target]:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(prep.get("test_size", 0.2)),
        random_state=int(prep.get("random_state", 42)),
        stratify=y,
    )
    return DataSplit(X_train, X_test, y_train, y_test), encoders


def log_sklearn_model(model: Any, input_example: pd.DataFrame) -> None:
    """Logs any scikit-learn compatible estimator as the run's MLflow model."""
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example,
        serialization_format="pickle",
    )


def log_xgboost_model(model: Any, input_example: pd.DataFrame) -> None:
    """Logs an XGBoost estimator with MLflow's native XGBoost flavor."""
    mlflow.xgboost.log_model(xgb_model=model, name="model", input_example=input_example)


@dataclass
class TrainedModel:
    """A fitted model plus everything needed to log it as one MLflow run."""

    display_name: str
    run_name: str
    model: Any
    params: Dict[str, Any]
    log_model: Callable[[Any, pd.DataFrame], None] = log_sklearn_model
    tags: Dict[str, str] = field(default_factory=dict)
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        """File-name friendly model name, e.g. 'random_forest'."""
        return self.display_name.lower().replace(" ", "_").replace("(", "").replace(")", "")


@dataclass(frozen=True)
class Predictions:
    """Test-set predictions of one model (churn probabilities only if the model provides them)."""

    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]


def predict_test(model: Any, split: DataSplit) -> Predictions:
    """Scores the held-out split once so metrics and plots share the same predictions."""
    y_proba = model.predict_proba(split.X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return Predictions(np.asarray(model.predict(split.X_test)), y_proba)


def compute_run_metrics(model: Any, split: DataSplit, preds: Predictions) -> Dict[str, Any]:
    """Train / test metrics in the layout of metrics.json (churn is the positive class)."""
    train_acc = float(accuracy_score(split.y_train, model.predict(split.X_train)))
    test = pp.compute_classification_metrics(split.y_test, preds.y_pred, preds.y_proba)
    (tn, fp), (fn, tp) = test["confusion_matrix"]
    return {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test["accuracy"], 4),
        "generalization_gap": round(train_acc - test["accuracy"], 4),
        "churn_recall": round(test["recall"], 4),
        "churn_precision": round(test["precision"], 4),
        "churn_f1_score": round(test["f1"], 4),
        "roc_auc": round(test.get("roc_auc", 0.0), 4),
        "confusion_matrix_tp": int(tp),
        "confusion_matrix_fp": int(fp),
        "confusion_matrix_tn": int(tn),
        "confusion_matrix_fn": int(fn),
    }


def _plot_path(prefix: Optional[str], kind: str) -> str:
    return os.path.join(PLOTS_DIR, f"{prefix}_{kind}.png" if prefix else f"{kind}.png")


def save_evaluation_plots(
    model: Any,
    split: DataSplit,
    preds: Predictions,
    title: str,
    prefix: Optional[str] = None,
) -> List[str]:
    """Writes the confusion matrix, ROC curve and (tree models) feature importance plots.

    Files are named '<prefix>_<kind>.png', or '<kind>.png' without a prefix (the DVC plots).
    Returns the paths written.
    """
    paths: List[str] = []
    cm = confusion_matrix(split.y_test, preds.y_pred, labels=[0, 1])
    accuracy = float(accuracy_score(split.y_test, preds.y_pred))

    path = _plot_path(prefix, "confusion_matrix")
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn (0)", "Churn (1)"],
        yticklabels=["No Churn (0)", "Churn (1)"],
    )
    plt.title(f"{title} Confusion Matrix (Acc: {accuracy:.1%})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    paths.append(path)

    if preds.y_proba is not None:
        path = _plot_path(prefix, "roc_curve")
        fpr, tpr, _ = roc_curve(split.y_test, preds.y_proba)
        auc = float(roc_auc_score(split.y_test, preds.y_proba))
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title(f"{title} ROC Curve", fontsize=12, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)

    if hasattr(model, "feature_importances_"):
        path = _plot_path(prefix, "feature_importance")
        importances = pd.Series(model.feature_importances_, index=split.feature_names).sort_values()
        plt.figure(figsize=(8, 6))
        importances.tail(12).plot(kind="barh", color="#0284c7")
        plt.title(f"{title} Top Feature Importances", fontsize=12, fontweight="bold")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        paths.append(path)

    return paths


def evaluate_and_log_run(trained: TrainedModel, split: DataSplit) -> Dict[str, Any]:
    """Evaluates a trained model on the held-out split and logs it as one MLflow run."""
    print(f"\n---> Logging Run: {trained.run_name} ({trained.display_name})")

    with mlflow.start_run(run_name=trained.run_name):
        mlflow.log_params(trained.params)
        mlflow.log_param("model_family", trained.display_name)
        mlflow.set_tags({"algorithm": trained.display_name, "developer": "Ashwin", **trained.tags})

        preds = predict_test(trained.model, split)
        metrics = compute_run_metrics(trained.model, split, preds)
        metrics.update(trained.extra_metrics)
        mlflow.log_metrics(metrics)

        print(
            f"[{trained.display_name}] Test Acc: {metrics['test_accuracy']:.4f} | "
            f"Recall: {metrics['churn_recall']:.4f} | Precision: {metrics['churn_precision']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

        for path in save_evaluation_plots(trained.model, split, preds, trained.display_name, trained.slug):
            mlflow.log_artifact(path, artifact_path="plots")

        trained.log_model(trained.model, split.X_test.iloc[:2])

    return metrics
