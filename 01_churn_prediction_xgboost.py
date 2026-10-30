"""End-to-End Multi-Model Customer Churn Pipeline with MLflow & DVC.

Supports training and comparing multiple model families:
1. Logistic Regression (Linear baseline with class weights)
2. Random Forest (Bagging ensemble with GridSearch)
3. XGBoost (Gradient boosting with scale_pos_weight & GridSearch)

Tracks all runs in MLflow for side-by-side leaderboard comparison, keeps the best
model by params.yaml model.champion_metric, and produces DVC metrics and plots for
reproducible pipelines. Set MLFLOW_TRACKING_URI to log to a tracking server.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Any, Callable, Dict, List, Tuple

import mlflow
from sklearn.metrics import classification_report

import classical_models as cm
import training_common as tc

METRICS_FILE = os.path.join(tc.BASE_DIR, "metrics.json")
REPORT_FILE = os.path.join(tc.BASE_DIR, "classification_report.json")
OUTPUT_MODEL_FILE = os.path.join(tc.BASE_DIR, "customer_churn_model_xgboost.pkl")
OUTPUT_ENCODERS_FILE = os.path.join(tc.BASE_DIR, "encoders.pkl")


Trainer = Callable[[tc.DataSplit, Dict[str, Any]], tc.TrainedModel]

# Training order for model.type "all"; keys are the accepted model.type values.
TRAINERS: Dict[str, Trainer] = {
    "logistic_regression": cm.train_logistic_regression,
    "random_forest": cm.train_random_forest,
    "xgboost": cm.train_xgboost,
}


def select_trainers(model_mode: str) -> List[Trainer]:
    """Trainers to run for params.yaml model.type."""
    if model_mode == "all":
        return list(TRAINERS.values())
    if model_mode not in TRAINERS:
        raise ValueError(f"Unknown model.type '{model_mode}'. Use 'all' or one of: {', '.join(TRAINERS)}.")
    return [TRAINERS[model_mode]]


def select_champion(
    results: List[Tuple[tc.TrainedModel, Dict[str, Any]]],
    metric: str,
) -> Tuple[tc.TrainedModel, Dict[str, Any]]:
    """Best (model, metrics) pair by the given metric; earlier models win ties."""
    if metric not in results[0][1]:
        raise ValueError(f"Unknown model.champion_metric '{metric}'. Use one of: {', '.join(results[0][1])}.")
    return max(results, key=lambda result: result[1][metric])


def export_champion(
    champion: tc.TrainedModel,
    metrics: Dict[str, Any],
    split: tc.DataSplit,
    encoders: Dict[str, Any],
) -> None:
    """Writes the DVC outputs: model + encoders (same run), metrics, report and standard plots."""
    with open(OUTPUT_MODEL_FILE, "wb") as handle:
        pickle.dump({"model": champion.model, "features_names": split.feature_names}, handle)
    with open(OUTPUT_ENCODERS_FILE, "wb") as handle:
        pickle.dump(encoders, handle)
    print(f"Production model ({champion.display_name}) saved to '{OUTPUT_MODEL_FILE}'.")
    print(f"Encoders saved to '{OUTPUT_ENCODERS_FILE}'.")

    preds = tc.predict_test(champion.model, split)
    tc.save_evaluation_plots(champion.model, split, preds, "Production Model")

    with open(METRICS_FILE, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(classification_report(split.y_test, preds.y_pred, output_dict=True), handle, indent=2)


def run_training_pipeline(config: Dict[str, Any]) -> None:
    """Prepares the data, trains and logs every selected model, then exports the champion."""
    model_mode = str(config["model"].get("type", "all")).lower()
    champion_metric = str(config["model"].get("champion_metric", "roc_auc"))
    trainers = select_trainers(model_mode)

    print("\n[1/4] Loading, cleaning and encoding the Telco Customer Churn dataset...")
    split, encoders = tc.prepare_data(config)
    print(f"Train samples: {len(split.X_train)}, Test samples: {len(split.X_test)}")
    print(f"Minority ratio (scale_pos_weight): {split.scale_pos_weight:.2f}")

    print(f"\n[2/4] Multi-Model Training (Mode: '{model_mode}')...")
    mlflow.set_experiment(tc.EXPERIMENT_NAME)
    results = []
    for trainer in trainers:
        trained = trainer(split, config)
        results.append((trained, tc.evaluate_and_log_run(trained, split)))

    print(f"\n[3/4] Selecting champion by '{champion_metric}'...")
    champion, champion_metrics = select_champion(results, champion_metric)
    print(f"Champion: {champion.display_name} ({champion_metric} = {champion_metrics[champion_metric]:.4f})")

    print("\n[4/4] Finalizing DVC artifacts...")
    export_champion(champion, champion_metrics, split, encoders)

    print("\n" + "=" * 65)
    print(" MULTI-MODEL BENCHMARK COMPLETE")
    print("=" * 65)
    print(f"• MLflow tracking URI:        {mlflow.get_tracking_uri()}")
    print("• DVC Metrics Table:          dvc metrics show")
    print("=" * 65)


def main() -> None:
    """CLI entry point (python 01_churn_prediction_xgboost.py / dvc repro)."""
    tc.configure_runtime()
    try:
        run_training_pipeline(tc.load_config())
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
