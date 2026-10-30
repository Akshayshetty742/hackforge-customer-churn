"""Tri-paradigm benchmark: classical ML vs deep learning (DML) vs quantum ML (QML).

Every model is trained on the same stratified split, corrects the 73 / 27 class imbalance by
up-weighting churners by the same ratio, and is scored at the same 0.5 threshold by
training_common.evaluate_and_log_run. Each run is logged to the MLflow experiment
"Customer-Churn-Paradigms" with a `paradigm` tag (ML / DML / QML), its trainable parameter
count and its training time, and the leaderboard is written to paradigm_metrics.json.

The deep and quantum models are also saved to models/ (pp.PARADIGM_MODEL_FILES) so the app can
show their predictions next to the production XGBoost model. This script never touches the
production model (customer_churn_model_xgboost.pkl).

    python 02_paradigm_benchmark.py                   # local ./mlflow.db
    MLFLOW_TRACKING_URI=http://localhost:5001 python 02_paradigm_benchmark.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

import classical_models as cm
import prediction_pipeline as pp
import training_common as tc
from dml_model import ChurnMLPClassifier, TrainingConfig
from qml_model import VariationalQuantumClassifier

EXPERIMENT_NAME = "Customer-Churn-Paradigms"
LEADERBOARD_FILE = os.path.join(tc.BASE_DIR, "paradigm_metrics.json")
COMPARISON_PLOT = os.path.join(tc.PLOTS_DIR, "paradigm_comparison.png")
LEADERBOARD_METRICS = ["test_accuracy", "churn_recall", "churn_precision", "churn_f1_score", "roc_auc"]

DEFAULT_PARADIGMS: Dict[str, Any] = {
    "include": ["ml", "dml", "qml"],
    "dml": {
        "hidden_layers": [64, 32],
        "dropout": 0.3,
        "training": {"learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 128, "max_epochs": 200,
                     "patience": 15},
    },
    "qml": {
        "n_qubits": 6,
        "n_layers": 3,
        "training": {"learning_rate": 0.02, "weight_decay": 0.0, "batch_size": 64, "max_epochs": 40,
                     "patience": 8},
    },
}


def _training_config(settings: Dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(**settings.get("training", {}))


def _training_params(config: TrainingConfig) -> Dict[str, Any]:
    return {f"train_{k}": v for k, v in vars(config).items()}


def train_dml(split: tc.DataSplit, settings: Dict[str, Any], random_state: int) -> tc.TrainedModel:
    """PyTorch MLP with batch norm, dropout, Adam and early stopping."""
    print("\n--- Training Deep MLP (PyTorch) ---")
    training = _training_config(settings)
    model = ChurnMLPClassifier(
        hidden_layers=tuple(settings.get("hidden_layers", (64, 32))),
        dropout=float(settings.get("dropout", 0.3)),
        training=training,
        random_state=random_state,
    ).fit(split.X_train, split.y_train)
    return tc.TrainedModel(
        display_name="Deep MLP",
        run_name="dml-mlp-pytorch",
        model=model,
        params={
            "hidden_layers": str(list(model.hidden_layers)),
            "dropout": model.dropout,
            "input_features": model.n_features_,
            "epochs_run": model.n_epochs_,
            **_training_params(training),
        },
        extra_metrics={"trainable_parameters": model.n_parameters, "best_val_loss": round(model.best_val_loss_, 4)},
    )


def train_qml(split: tc.DataSplit, settings: Dict[str, Any], random_state: int) -> tc.TrainedModel:
    """Data re-uploading variational quantum classifier on a CPU state-vector simulator."""
    print("\n--- Training Variational Quantum Classifier (PennyLane) ---")
    training = _training_config(settings)
    model = VariationalQuantumClassifier(
        n_qubits=int(settings.get("n_qubits", 6)),
        n_layers=int(settings.get("n_layers", 3)),
        training=training,
        random_state=random_state,
    ).fit(split.X_train, split.y_train)
    return tc.TrainedModel(
        display_name="Quantum VQC",
        run_name="qml-vqc-pennylane",
        model=model,
        params={
            "n_qubits": model.n_qubits,
            "n_layers": model.n_layers,
            "embedding": "AngleEmbedding(RY), re-uploaded per layer",
            "ansatz": "StronglyEntanglingLayers (Rot + CNOT ring)",
            "measurement": "PauliZ expectation on every qubit -> linear head",
            "device": "default.qubit",
            "epochs_run": model.n_epochs_,
            **_training_params(training),
        },
        extra_metrics={
            "trainable_parameters": model.n_parameters,
            "quantum_parameters": model.n_quantum_parameters,
            "classical_parameters": model.n_classical_parameters,
            "best_val_loss": round(model.best_val_loss_, 4),
        },
    )


def _classical(trainer: Callable[..., tc.TrainedModel], config: Dict[str, Any]) -> Callable[..., tc.TrainedModel]:
    def train(split: tc.DataSplit) -> tc.TrainedModel:
        trained = trainer(split, config)
        trained.extra_metrics["trainable_parameters"] = pp.count_model_parameters(trained.model)
        return trained

    return train


def build_plan(config: Dict[str, Any]) -> List[Tuple[str, Callable[[tc.DataSplit], tc.TrainedModel]]]:
    """(paradigm tag, trainer) pairs for the paradigms listed in params.yaml paradigms.include."""
    paradigms = {**DEFAULT_PARADIGMS, **config.get("paradigms", {})}
    random_state = int(config["prepare"].get("random_state", 42))
    available = {
        "ml": [("ML", _classical(cm.train_xgboost, config)),
               ("ML", _classical(cm.train_logistic_regression, config))],
        "dml": [("DML", lambda split: train_dml(split, paradigms["dml"], random_state))],
        "qml": [("QML", lambda split: train_qml(split, paradigms["qml"], random_state))],
    }
    unknown = [name for name in paradigms["include"] if name not in available]
    if unknown:
        raise ValueError(f"Unknown paradigms {unknown} in paradigms.include. Use: {', '.join(available)}.")
    return [step for name in paradigms["include"] for step in available[name]]


def save_serving_model(paradigm: str, trained: tc.TrainedModel, split: tc.DataSplit, metrics: Dict[str, Any]) -> None:
    """Writes the model the Streamlit app loads for this paradigm (with its held-out metrics)."""
    path = pp.PARADIGM_MODEL_FILES[paradigm]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": trained.model,
        "features_names": split.feature_names,
        "paradigm": paradigm,
        "display_name": trained.display_name,
        "test_metrics": {name: metrics[name] for name in LEADERBOARD_METRICS},
        "trainable_parameters": int(metrics["trainable_parameters"]),
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    print(f"Serving model for {paradigm} saved to '{path}'.")


def plot_comparison(leaderboard: pd.DataFrame) -> None:
    """Grouped bar chart of the headline metrics per model."""
    ax = leaderboard.set_index("model")[LEADERBOARD_METRICS].T.plot(kind="bar", figsize=(10, 5), width=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score on held-out test set")
    ax.set_title("ML vs DML vs QML: Customer Churn", fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT, dpi=200)
    plt.close()


def run_benchmark(config: Dict[str, Any]) -> pd.DataFrame:
    """Trains, evaluates and logs every planned model; returns the leaderboard."""
    plan = build_plan(config)
    split, _ = tc.prepare_data(config)
    print(f"Train samples: {len(split.X_train)}, Test samples: {len(split.X_test)}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    rows = []
    for paradigm, train in plan:
        start = time.perf_counter()
        trained = train(split)
        seconds = round(time.perf_counter() - start, 2)
        trained.tags["paradigm"] = paradigm
        trained.extra_metrics["train_seconds"] = seconds
        metrics = tc.evaluate_and_log_run(trained, split)
        if paradigm in pp.PARADIGM_MODEL_FILES:
            save_serving_model(paradigm, trained, split, metrics)
        rows.append({"paradigm": paradigm, "model": trained.display_name, **metrics})

    leaderboard = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    with open(LEADERBOARD_FILE, "w", encoding="utf-8") as handle:
        json.dump({row["model"]: row for row in leaderboard.to_dict(orient="records")}, handle, indent=2)
    plot_comparison(leaderboard)
    return leaderboard


def main() -> None:
    """CLI entry point (python 02_paradigm_benchmark.py / dvc repro benchmark_paradigms)."""
    tc.configure_runtime()
    try:
        leaderboard = run_benchmark(tc.load_config())
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error: {exc}")

    columns = ["paradigm", "model", *LEADERBOARD_METRICS, "trainable_parameters", "train_seconds"]
    print("\n" + "=" * 100)
    print(" ML vs DML vs QML LEADERBOARD (sorted by ROC-AUC)")
    print("=" * 100)
    print(leaderboard[columns].to_string(index=False))
    print("=" * 100)
    print(f"• MLflow experiment '{EXPERIMENT_NAME}' at {mlflow.get_tracking_uri()}")
    print(f"• Leaderboard: {LEADERBOARD_FILE}")
    print(f"• Chart:       {COMPARISON_PLOT}")


if __name__ == "__main__":
    main()
