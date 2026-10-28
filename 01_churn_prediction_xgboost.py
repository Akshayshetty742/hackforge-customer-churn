"""End-to-End Multi-Model Customer Churn Pipeline with MLflow & DVC.

Supports training and comparing multiple model families:
1. Logistic Regression (Linear baseline with class weights)
2. Random Forest (Bagging ensemble with GridSearch)
3. XGBoost (Gradient boosting with scale_pos_weight & GridSearch)

Tracks all runs in MLflow for side-by-side leaderboard comparison,
and produces DVC metrics and plots for reproducible pipelines.
"""

from __future__ import annotations

import json
import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import pickle
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PARAMS_FILE = os.path.join(BASE_DIR, "params.yaml")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METRICS_FILE = os.path.join(BASE_DIR, "metrics.json")
REPORT_FILE = os.path.join(BASE_DIR, "classification_report.json")
OUTPUT_MODEL_FILE = os.path.join(BASE_DIR, "customer_churn_model_xgboost.pkl")
OUTPUT_ENCODERS_FILE = os.path.join(BASE_DIR, "encoders.pkl")

os.makedirs(PLOTS_DIR, exist_ok=True)

# Default parameter fallbacks
config: Dict[str, Any] = {
    "prepare": {
        "test_size": 0.2,
        "random_state": 42,
        "target_column": "Churn",
    },
    "model": {
        "type": "all",  # "all", "xgboost", "random_forest", "logistic_regression"
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

if os.path.exists(PARAMS_FILE):
    try:
        with open(PARAMS_FILE, "r", encoding="utf-8") as pf:
            loaded_params = yaml.safe_load(pf)
            if loaded_params:
                config.update(loaded_params)
        print(f"Loaded configuration from '{PARAMS_FILE}'.")
    except Exception as e:
        print(f"Warning: Could not parse '{PARAMS_FILE}' ({e}), using default parameters.")

prep_cfg = config.get("prepare", {})
train_cfg = config.get("train", {})
model_mode = str(config.get("model", {}).get("type", "all")).lower()

test_size = float(prep_cfg.get("test_size", 0.2))
random_state = int(prep_cfg.get("random_state", 42))
target_col = str(prep_cfg.get("target_column", "Churn"))
cv_splits = int(train_cfg.get("cv_splits", 3))
eval_metric = str(train_cfg.get("eval_metric", "logloss"))
scoring_metric = str(train_cfg.get("scoring", "accuracy"))
xgb_grid = train_cfg.get("param_grid", {})


def evaluate_and_log_run(
    model_name: str,
    run_name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params_logged: Dict[str, Any],
    is_xgb: bool = False,
) -> Dict[str, Any]:
    """Evaluates a trained model, generates visual plots, and logs run to MLflow."""
    print(f"\n---> Logging Run: {run_name} ({model_name})")

    with mlflow.start_run(run_name=run_name):
        # Log Hyperparameters & metadata
        mlflow.log_params(params_logged)
        mlflow.log_param("model_family", model_name)
        mlflow.set_tag("algorithm", model_name)
        mlflow.set_tag("developer", "Ashwin")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        train_acc = float(accuracy_score(y_train, y_train_pred))
        test_acc = float(accuracy_score(y_test, y_test_pred))
        churn_recall = float(recall_score(y_test, y_test_pred, pos_label=1))
        churn_prec = float(precision_score(y_test, y_test_pred, pos_label=1))
        churn_f1 = float(f1_score(y_test, y_test_pred, pos_label=1))
        roc_auc = float(roc_auc_score(y_test, y_test_proba)) if y_test_proba is not None else 0.0
        gen_gap = float(train_acc - test_acc)

        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        metrics_dict = {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "generalization_gap": round(gen_gap, 4),
            "churn_recall": round(churn_recall, 4),
            "churn_precision": round(churn_prec, 4),
            "churn_f1_score": round(churn_f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix_tp": tp,
            "confusion_matrix_fp": fp,
            "confusion_matrix_tn": tn,
            "confusion_matrix_fn": fn,
        }
        mlflow.log_metrics(metrics_dict)

        print(f"[{model_name}] Test Acc: {test_acc:.4f} | Recall: {churn_recall:.4f} | Precision: {churn_prec:.4f} | ROC-AUC: {roc_auc:.4f}")

        # Plot 1: Confusion Matrix
        prefix = model_name.lower().replace(" ", "_")
        cm_path = os.path.join(PLOTS_DIR, f"{prefix}_confusion_matrix.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn (0)", "Churn (1)"],
            yticklabels=["No Churn (0)", "Churn (1)"],
        )
        plt.title(f"{model_name} Confusion Matrix (Acc: {test_acc:.1%})", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=200)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Plot 2: ROC Curve
        if y_test_proba is not None:
            roc_path = os.path.join(PLOTS_DIR, f"{prefix}_roc_curve.png")
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate (1 - Specificity)")
            plt.ylabel("True Positive Rate (Recall)")
            plt.title(f"{model_name} ROC Curve", fontsize=12, fontweight="bold")
            plt.legend(loc="lower right")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(roc_path, dpi=200)
            plt.close()
            mlflow.log_artifact(roc_path, artifact_path="plots")

        # Plot 3: Feature Importances (for tree-based models)
        if hasattr(model, "feature_importances_"):
            feat_path = os.path.join(PLOTS_DIR, f"{prefix}_feature_importance.png")
            importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
            plt.figure(figsize=(8, 6))
            importances.tail(12).plot(kind="barh", color="#0284c7")
            plt.title(f"{model_name} Top Feature Importances", fontsize=12, fontweight="bold")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(feat_path, dpi=200)
            plt.close()
            mlflow.log_artifact(feat_path, artifact_path="plots")

        # Log Model Flavor to MLflow
        if is_xgb:
            mlflow.xgboost.log_model(xgb_model=model, artifact_path="model", input_example=X_test.iloc[:2])
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_test.iloc[:2],
                serialization_format="pickle",
            )

        return metrics_dict


def run_training_pipeline() -> None:
    # -----------------------------------------------------------------------
    # 1. Data Loading
    # -----------------------------------------------------------------------
    print("\n[1/5] Loading Telco Customer Churn dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at '{DATASET_PATH}'.")
        exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset shape: {df.shape}")

    # -----------------------------------------------------------------------
    # 2. Preprocessing
    # -----------------------------------------------------------------------
    print("\n[2/5] Cleaning and encoding data...")
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)

    object_columns = df.select_dtypes(include="object").columns
    encoders: Dict[str, LabelEncoder] = {}
    for col in object_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    with open(OUTPUT_ENCODERS_FILE, "wb") as f:
        pickle.dump(encoders, f)
    print(f"Encoders saved to '{OUTPUT_ENCODERS_FILE}'.")

    # -----------------------------------------------------------------------
    # 3. Train-Test Split
    # -----------------------------------------------------------------------
    print("\n[3/5] Stratified Train-Test Split...")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    count_minority = int(y_train.sum())
    count_majority = int(len(y_train) - count_minority)
    scale_pos_weight = count_majority / count_minority if count_minority > 0 else 1.0
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Minority ratio (scale_pos_weight): {scale_pos_weight:.2f}")

    # -----------------------------------------------------------------------
    # 4. Multi-Model Training & MLflow Tracking
    # -----------------------------------------------------------------------
    print(f"\n[4/5] Multi-Model Training (Mode: '{model_mode}')...")
    experiment_name = "Customer-Churn-Prediction"
    mlflow.set_experiment(experiment_name)

    champion_model: Any = None
    champion_metrics: Dict[str, Any] = {}

    # --- Model 1: Logistic Regression ---
    if model_mode in ["all", "logistic_regression"]:
        print("\n--- Training Logistic Regression Baseline ---")
        lr_model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )
        lr_model.fit(X_train, y_train)

        lr_metrics = evaluate_and_log_run(
            model_name="Logistic Regression",
            run_name="logistic-regression-baseline",
            model=lr_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params_logged={
                "class_weight": "balanced",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": random_state,
            },
            is_xgb=False,
        )
        champion_model = lr_model
        champion_metrics = lr_metrics

    # --- Model 2: Random Forest Classifier ---
    if model_mode in ["all", "random_forest"]:
        print("\n--- Training Random Forest Ensemble ---")
        rf_base = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
        )
        rf_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 8],
            "min_samples_split": [2, 5],
        }
        cv_rf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        rf_search = GridSearchCV(
            estimator=rf_base,
            param_grid=rf_grid,
            cv=cv_rf,
            n_jobs=-1,
            scoring=scoring_metric,
        )
        rf_search.fit(X_train, y_train)
        rf_best = rf_search.best_estimator_

        rf_params = {"cv_splits": cv_splits, "scoring": scoring_metric}
        for k, v in rf_search.best_params_.items():
            rf_params[f"best_{k}"] = v

        rf_metrics = evaluate_and_log_run(
            model_name="Random Forest",
            run_name="random-forest-ensemble",
            model=rf_best,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params_logged=rf_params,
            is_xgb=False,
        )
        champion_model = rf_best
        champion_metrics = rf_metrics

    # --- Model 3: XGBoost Classifier ---
    if model_mode in ["all", "xgboost"]:
        print("\n--- Training XGBoost with Hyperparameter Tuning ---")
        base_xgb = XGBClassifier(eval_metric=eval_metric, random_state=random_state)
        cv_xgb = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

        grid_params = {k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in xgb_grid.items()}
        grid_params["scale_pos_weight"] = [round(scale_pos_weight, 4)]

        xgb_search = GridSearchCV(
            estimator=base_xgb,
            param_grid=grid_params,
            cv=cv_xgb,
            n_jobs=-1,
            scoring=scoring_metric,
        )
        xgb_search.fit(X_train, y_train)
        xgb_best = xgb_search.best_estimator_

        xgb_params = {
            "cv_splits": cv_splits,
            "scoring": scoring_metric,
            "eval_metric": eval_metric,
            "scale_pos_weight": round(scale_pos_weight, 4),
        }
        for k, v in xgb_search.best_params_.items():
            xgb_params[f"best_{k}"] = v

        xgb_metrics = evaluate_and_log_run(
            model_name="XGBoost",
            run_name="xgboost-churn-classifier",
            model=xgb_best,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            params_logged=xgb_params,
            is_xgb=True,
        )
        champion_model = xgb_best
        champion_metrics = xgb_metrics

    # -----------------------------------------------------------------------
    # 5. Save Champion Model & DVC Outputs
    # -----------------------------------------------------------------------
    print("\n[5/5] Finalizing DVC artifacts...")
    model_payload = {
        "model": champion_model,
        "features_names": X.columns.tolist(),
    }
    with open(OUTPUT_MODEL_FILE, "wb") as f:
        pickle.dump(model_payload, f)
    print(f"Production model saved to '{OUTPUT_MODEL_FILE}'.")

    # Generate standard DVC plots using the champion model
    y_test_pred = champion_model.predict(X_test)
    y_test_proba = champion_model.predict_proba(X_test)[:, 1] if hasattr(champion_model, "predict_proba") else None
    cm = confusion_matrix(y_test, y_test_pred)

    # Standard confusion_matrix.png for DVC
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Production Model Confusion Matrix (Acc: {champion_metrics['test_accuracy']:.1%})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()

    # Standard roc_curve.png for DVC
    if y_test_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC curve (AUC = {champion_metrics['roc_auc']:.3f})")
        plt.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("Recall")
        plt.title("Production Model ROC Curve", fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=200)
        plt.close()

    # Standard feature_importance.png for DVC
    if hasattr(champion_model, "feature_importances_"):
        plt.figure(figsize=(8, 6))
        pd.Series(champion_model.feature_importances_, index=X.columns).sort_values().tail(12).plot(kind="barh", color="#0284c7")
        plt.title("Production Model Feature Importance", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=200)
        plt.close()

    # Write DVC metrics.json and classification_report.json
    with open(METRICS_FILE, "w", encoding="utf-8") as mf:
        json.dump(champion_metrics, mf, indent=2)

    cls_report = classification_report(y_test, y_test_pred, output_dict=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as rf:
        json.dump(cls_report, rf, indent=2)

    print("\n" + "=" * 65)
    print(" MULTI-MODEL BENCHMARK COMPLETE")
    print("=" * 65)
    print("• View Leaderboard in MLflow: http://127.0.0.1:5001")
    print("• DVC Metrics Table:          dvc metrics show")
    print("=" * 65)


if __name__ == "__main__":
    run_training_pipeline()
