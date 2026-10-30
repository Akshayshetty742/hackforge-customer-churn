"""Classical ML trainers (paradigm 1 of 3): Logistic Regression, Random Forest and XGBoost.

Each trainer fits on a training_common.DataSplit and returns a TrainedModel ready for
training_common.evaluate_and_log_run. Used by 01_churn_prediction_xgboost.py and
02_paradigm_benchmark.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

import training_common as tc


def _random_state(config: Dict[str, Any]) -> int:
    return int(config["prepare"].get("random_state", 42))


def _grid_search(estimator: Any, grid: Dict[str, List[Any]], config: Dict[str, Any]) -> GridSearchCV:
    """Stratified K-fold grid search configured from params.yaml train settings."""
    train_cfg = config["train"]
    cv = StratifiedKFold(
        n_splits=int(train_cfg.get("cv_splits", 3)),
        shuffle=True,
        random_state=_random_state(config),
    )
    return GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        cv=cv,
        n_jobs=-1,
        scoring=str(train_cfg.get("scoring", "accuracy")),
    )


def _search_params(search: GridSearchCV, config: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "cv_splits": int(config["train"].get("cv_splits", 3)),
        "scoring": str(config["train"].get("scoring", "accuracy")),
    }
    params.update({f"best_{k}": v for k, v in search.best_params_.items()})
    return params


def train_logistic_regression(split: tc.DataSplit, config: Dict[str, Any]) -> tc.TrainedModel:
    """Linear baseline with balanced class weights."""
    print("\n--- Training Logistic Regression Baseline ---")
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=_random_state(config))
    model.fit(split.X_train, split.y_train)
    return tc.TrainedModel(
        display_name="Logistic Regression",
        run_name="logistic-regression-baseline",
        model=model,
        params={
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": _random_state(config),
        },
    )


def train_random_forest(split: tc.DataSplit, config: Dict[str, Any]) -> tc.TrainedModel:
    """Bagging ensemble with balanced class weights, tuned by grid search."""
    print("\n--- Training Random Forest Ensemble ---")
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 8],
        "min_samples_split": [2, 5],
    }
    search = _grid_search(
        RandomForestClassifier(class_weight="balanced", random_state=_random_state(config)), grid, config
    )
    search.fit(split.X_train, split.y_train)
    return tc.TrainedModel(
        display_name="Random Forest",
        run_name="random-forest-ensemble",
        model=search.best_estimator_,
        params=_search_params(search, config),
    )


def train_xgboost(split: tc.DataSplit, config: Dict[str, Any]) -> tc.TrainedModel:
    """Gradient boosting with scale_pos_weight, tuned over params.yaml train.param_grid."""
    print("\n--- Training XGBoost with Hyperparameter Tuning ---")
    eval_metric = str(config["train"].get("eval_metric", "logloss"))
    scale_pos_weight = round(split.scale_pos_weight, 4)

    grid = {k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in config["train"]["param_grid"].items()}
    grid["scale_pos_weight"] = [scale_pos_weight]
    search = _grid_search(XGBClassifier(eval_metric=eval_metric, random_state=_random_state(config)), grid, config)
    search.fit(split.X_train, split.y_train)

    params = _search_params(search, config)
    params.update({"eval_metric": eval_metric, "scale_pos_weight": scale_pos_weight})
    return tc.TrainedModel(
        display_name="XGBoost",
        run_name="xgboost-churn-classifier",
        model=search.best_estimator_,
        params=params,
        log_model=tc.log_xgboost_model,
    )
