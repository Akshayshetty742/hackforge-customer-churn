"""System validation tests: Test Plan test cases TC-01 to TC-13 plus TC-23 to TC-25.

The training pipeline (01_churn_prediction_xgboost.py) and the paradigm benchmark
(02_paradigm_benchmark.py) are run end to end, but every output path and the MLflow store are
redirected to a temporary folder, so the committed production model is never touched.

Skipped when the MLOps requirements (requirements-mlops.txt) are not installed.
"""
# pylint: disable=redefined-outer-name  # pytest injects fixtures by parameter name

import ast
import copy
import importlib
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import customer_presets as cp
import prediction_pipeline as pp

if any(importlib.util.find_spec(name) is None for name in ("mlflow", "yaml", "matplotlib", "seaborn")):
    pytest.skip("MLOps dependencies not installed (requirements-mlops.txt)", allow_module_level=True)

# pylint: disable=wrong-import-position,wrong-import-order  # importable only once the dependencies are known
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import mlflow  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import classical_models as cm  # noqa: E402
import training_common as tc  # noqa: E402

trainer = importlib.import_module("01_churn_prediction_xgboost")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_MODULES = [
    "app.py",
    "prediction_pipeline.py",
    "customer_presets.py",
    "training_common.py",
    "classical_models.py",
    "01_churn_prediction_xgboost.py",
    "02_paradigm_benchmark.py",
    "dml_model.py",
    "qml_model.py",
]
RESEARCH_INSTALLED = all(importlib.util.find_spec(name) for name in ("torch", "pennylane"))
TRAINING_TIME_BUDGET_SECONDS = 300  # NFR-01
PREDICTION_LATENCY_BUDGET_MS = 200  # NFR-11
SMALL_GRID = {
    "n_estimators": [100],
    "max_depth": [3, 4],
    "learning_rate": [0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}


def _record(request, name, value):
    """Adds a measured value to the JUnit XML so the reports can quote it."""
    request.node.user_properties.append((name, value))


def _small_config():
    config = tc.load_config()
    config["train"] = {**config["train"], "param_grid": copy.deepcopy(SMALL_GRID)}
    return config


def _use_temporary_mlflow(directory, experiment_name):
    """Points MLflow at a SQLite store inside `directory`; returns the previous tracking URI."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{directory / 'mlflow.db'}")
    mlflow.create_experiment(experiment_name, artifact_location=(directory / "artifacts").as_uri())
    return previous


def _redirect_training_outputs(monkeypatch, directory):
    (directory / "plots").mkdir(exist_ok=True)
    monkeypatch.setattr(tc, "PLOTS_DIR", str(directory / "plots"))
    for name, file_name in [
        ("METRICS_FILE", "metrics.json"),
        ("REPORT_FILE", "classification_report.json"),
        ("OUTPUT_MODEL_FILE", "customer_churn_model_xgboost.pkl"),
        ("OUTPUT_ENCODERS_FILE", "encoders.pkl"),
    ]:
        monkeypatch.setattr(trainer, name, str(directory / file_name))


def _read(path):
    with open(os.path.join(BASE_DIR, path), encoding="utf-8") as handle:
        return handle.read()


def _imported_top_level_modules(path):
    tree = ast.parse(_read(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.fixture(scope="module")
def config():
    """params.yaml configuration (the documented default grid)."""
    return tc.load_config()


@pytest.fixture(scope="module")
def prepared(config):
    """(split, encoders) from the shared data preparation step."""
    return tc.prepare_data(config)


@pytest.fixture(scope="module")
def training_run(tmp_path_factory, config):
    """One full run of the training pipeline with the params.yaml grid, written to a temporary folder."""
    out = tmp_path_factory.mktemp("training_run")
    with pytest.MonkeyPatch.context() as monkeypatch:
        _redirect_training_outputs(monkeypatch, out)
        previous_uri = _use_temporary_mlflow(out, tc.EXPERIMENT_NAME)
        try:
            start = time.perf_counter()
            trainer.run_training_pipeline(copy.deepcopy(config))
            seconds = time.perf_counter() - start
            runs = mlflow.search_runs(experiment_names=[tc.EXPERIMENT_NAME])
        finally:
            mlflow.set_tracking_uri(previous_uri)
    return SimpleNamespace(dir=out, seconds=seconds, runs=runs)


# ----- TC-01 Load valid dataset ---------------------------------------------------------------
@pytest.mark.tc("TC-01")
def test_tc01_valid_dataset_loads_and_splits(prepared):
    split, _ = prepared
    assert len(split.X_train) + len(split.X_test) == 7043
    assert len(split.X_test) == 1409
    assert sorted(split.feature_names) == sorted(pp.EXPECTED_FEATURE_COLUMNS)


# ----- TC-02 Missing file / missing target handled ----------------------------------------------
@pytest.mark.tc("TC-02")
def test_tc02_missing_dataset_raises_clear_error(config, tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        tc.prepare_data(config, dataset_path=str(tmp_path / "missing.csv"))


@pytest.mark.tc("TC-02")
def test_tc02_dataset_without_target_raises_clear_error(config, tmp_path):
    path = tmp_path / "no_target.csv"
    pd.read_csv(pp.DEFAULT_DATASET_FILE).head(50).drop(columns=["Churn"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Churn"):
        tc.prepare_data(config, dataset_path=str(path))


@pytest.mark.tc("TC-02")
def test_tc02_cli_stops_without_writing_outputs(monkeypatch, tmp_path):
    _redirect_training_outputs(monkeypatch, tmp_path)
    prepare = tc.prepare_data
    monkeypatch.setattr(tc, "prepare_data", lambda cfg: prepare(cfg, dataset_path=str(tmp_path / "missing.csv")))
    with warnings.catch_warnings(), pytest.raises(SystemExit) as stopped:
        trainer.main()
    assert str(stopped.value.code).startswith("Error: Dataset file not found")
    assert not os.path.exists(trainer.METRICS_FILE)
    assert not os.path.exists(trainer.OUTPUT_MODEL_FILE)


@pytest.mark.tc("TC-02")
def test_tc02_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError, match="Unknown model.type"):
        trainer.select_trainers("svm")
    assert trainer.select_trainers("xgboost") == [cm.train_xgboost]
    with pytest.raises(ValueError, match="Unknown model.champion_metric"):
        trainer.select_champion([(None, {"roc_auc": 0.8})], "auc")
    assert set(config) >= {"prepare", "model", "train"}


@pytest.mark.tc("TC-02")
def test_tc02_missing_or_broken_params_file_falls_back_to_defaults(tmp_path):
    assert tc.load_config(str(tmp_path / "absent.yaml")) == tc.DEFAULT_CONFIG
    broken = tmp_path / "broken.yaml"
    broken.write_text("train: [unclosed", encoding="utf-8")
    assert tc.load_config(str(broken)) == tc.DEFAULT_CONFIG


# ----- TC-03 Target separation and encoding -----------------------------------------------------
@pytest.mark.tc("TC-03")
def test_tc03_target_separated_and_text_columns_encoded(prepared):
    split, encoders = prepared
    assert "Churn" not in split.X_train.columns
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in split.X_train.dtypes)
    assert set(encoders) == set(pp.CATEGORICAL_COLUMNS) | {"Churn"}
    assert list(encoders["Churn"].classes_) == ["No", "Yes"]
    assert set(split.y_train.unique()) == {0, 1}


# ----- TC-04 XGBoost training with class weighting ----------------------------------------------
@pytest.mark.tc("TC-04")
def test_tc04_xgboost_trains_with_class_weighting(prepared):
    split, _ = prepared
    trained = cm.train_xgboost(split, _small_config())
    assert isinstance(trained.model, XGBClassifier)
    assert trained.model.get_params()["scale_pos_weight"] == pytest.approx(split.scale_pos_weight, abs=1e-3)
    assert trained.model.predict_proba(split.X_test).shape == (1409, 2)


# ----- TC-05 Grid search with stratified cross validation ----------------------------------------
@pytest.mark.tc("TC-05")
def test_tc05_grid_search_uses_stratified_folds(config):
    search = cm._grid_search(XGBClassifier(), {"max_depth": [3]}, config)  # pylint: disable=protected-access
    assert isinstance(search.cv, StratifiedKFold)
    assert search.cv.n_splits == config["train"]["cv_splits"]
    assert search.cv.shuffle


@pytest.mark.tc("TC-05")
def test_tc05_every_grid_combination_is_evaluated(prepared, config):
    split, _ = prepared
    grid = {"max_depth": [2, 3], "n_estimators": [20, 40]}
    search = cm._grid_search(XGBClassifier(), grid, config)  # pylint: disable=protected-access
    search.fit(split.X_train.head(800), split.y_train.head(800))
    assert len(search.cv_results_["params"]) == 4


@pytest.mark.tc("TC-05")
def test_tc05_best_hyperparameters_are_reported(training_run, config):
    xgb_run = training_run.runs[training_run.runs["tags.mlflow.runName"] == "xgboost-churn-classifier"].iloc[0]
    for name, values in config["train"]["param_grid"].items():
        assert float(xgb_run[f"params.best_{name}"]) in [float(v) for v in values]


# ----- TC-06 Evaluation metrics and reports ----------------------------------------------------
@pytest.mark.tc("TC-06")
def test_tc06_metrics_and_reports_are_written(training_run):
    metrics = json.loads((training_run.dir / "metrics.json").read_text(encoding="utf-8"))
    for key in ["test_accuracy", "churn_precision", "churn_recall", "churn_f1_score", "roc_auc"]:
        assert 0.0 <= metrics[key] <= 1.0
    counts = [metrics[f"confusion_matrix_{k}"] for k in ("tp", "fp", "tn", "fn")]
    assert sum(counts) == 1409
    report = json.loads((training_run.dir / "classification_report.json").read_text(encoding="utf-8"))
    assert {"0", "1", "accuracy"} <= set(report)
    assert (training_run.dir / "plots" / "confusion_matrix.png").exists()
    assert (training_run.dir / "plots" / "roc_curve.png").exists()


@pytest.mark.tc("TC-06")
def test_tc06_every_model_run_is_logged(training_run):
    assert set(training_run.runs["tags.mlflow.runName"]) == {
        "logistic-regression-baseline",
        "random-forest-ensemble",
        "xgboost-churn-classifier",
    }
    assert training_run.runs["metrics.churn_recall"].between(0, 1).all()


# ----- TC-07 Class imbalance handling, test set untouched ---------------------------------------
@pytest.mark.tc("TC-07")
def test_tc07_test_set_keeps_the_natural_class_ratio(prepared):
    split, _ = prepared
    full_ratio = (pd.read_csv(pp.DEFAULT_DATASET_FILE)["Churn"] == "Yes").mean()
    assert len(split.X_test) == 1409
    assert split.y_test.mean() == pytest.approx(full_ratio, abs=0.005)
    assert 2.7 < split.scale_pos_weight < 2.85


@pytest.mark.tc("TC-07")
def test_tc07_class_weighting_is_logged_for_each_model(training_run):
    runs = training_run.runs.set_index("tags.mlflow.runName")
    assert runs.loc["logistic-regression-baseline", "params.class_weight"] == "balanced"
    assert float(runs.loc["xgboost-churn-classifier", "params.scale_pos_weight"]) == pytest.approx(2.77, abs=0.01)


# ----- TC-08 Model and encoders saved ------------------------------------------------------------
@pytest.mark.tc("TC-08")
def test_tc08_model_and_encoders_are_saved(training_run):
    for name in ("customer_churn_model_xgboost.pkl", "encoders.pkl"):
        path = training_run.dir / name
        assert path.exists() and path.stat().st_size > 0


# ----- TC-09 Saved model and encoders reload as a pair --------------------------------------------
@pytest.mark.tc("TC-09")
def test_tc09_saved_pair_reloads_and_predicts(training_run):
    model, features, encoders = pp.load_artifacts(
        str(training_run.dir / "customer_churn_model_xgboost.pkl"), str(training_run.dir / "encoders.pkl")
    )
    assert set(pp.CATEGORICAL_COLUMNS) <= set(encoders)
    processed = pp.preprocess_input(cp.get_preset_values(cp.DEFAULT_PRESET), encoders, features)
    _, probability, _, _ = pp.predict_churn(model, processed)
    assert 0.0 <= probability <= 1.0


# ----- TC-10 Fresh environment and training runtime ---------------------------------------------
@pytest.mark.tc("TC-10")
def test_tc10_requirements_declare_every_third_party_import():
    declared = set()
    for file_name in os.listdir(BASE_DIR):
        if re.fullmatch(r"requirements(-[a-z]+)?\.txt", file_name):
            for line in _read(file_name).splitlines():
                match = re.match(r"\s*([A-Za-z0-9_.\-]+)", line)
                if match and not line.lstrip().startswith(("#", "-r")):
                    declared.add(match.group(1).lower())
    distribution = {"sklearn": "scikit-learn", "yaml": "pyyaml"}
    local = {name[:-3] for name in os.listdir(BASE_DIR) if name.endswith(".py")}
    imported = set().union(*(_imported_top_level_modules(path) for path in SOURCE_MODULES))
    third_party = {name for name in imported if name not in sys.stdlib_module_names and name not in local}
    assert {distribution.get(name, name).lower() for name in third_party} <= declared


@pytest.mark.tc("TC-10")
def test_tc10_training_run_completes_within_budget(training_run, request):
    _record(request, "training_seconds", f"{training_run.seconds:.1f}")
    assert training_run.seconds < TRAINING_TIME_BUDGET_SECONDS


# ----- TC-11 Privacy and repository hygiene ------------------------------------------------------
@pytest.mark.tc("TC-11")
def test_tc11_personal_identifiers_are_not_model_inputs(prepared):
    split, _ = prepared
    assert "customerID" not in split.feature_names
    assert not [name for name in split.feature_names if re.search(r"name|email|phone_?number|address", name, re.I)]


@pytest.mark.tc("TC-11")
def test_tc11_secrets_and_environments_are_git_ignored():
    ignored = _read(".gitignore").split()
    assert {".venv/", ".env", "secrets.toml"} <= set(ignored)


@pytest.mark.tc("TC-11")
def test_tc11_no_network_clients_in_the_code():
    network = {"requests", "urllib", "http", "socket", "smtplib", "ftplib", "httpx", "aiohttp"}
    for path in SOURCE_MODULES:
        assert not _imported_top_level_modules(path) & network, path


@pytest.mark.tc("TC-11")
def test_tc11_no_secret_files_are_tracked():
    try:
        tracked = subprocess.run(
            ["git", "ls-files"], cwd=BASE_DIR, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    secret = re.compile(r"(^|/)(\.env(\..*)?|secrets\.toml|.*\.pem|.*\.key|id_rsa.*)$")
    assert not [path for path in tracked if secret.search(path)]


# ----- TC-12 Documentation completeness ----------------------------------------------------------
@pytest.mark.tc("TC-12")
def test_tc12_readme_covers_setup_usage_model_and_limitations():
    headings = [line.lower() for line in _read("README.md").splitlines() if line.startswith("#")]
    for topic in ["installation", "streamlit", "training", "model performance", "limitations", "tests"]:
        assert any(topic in heading for heading in headings), topic
    assert os.path.exists(os.path.join(BASE_DIR, "DEVELOPER_GUIDE.md"))


@pytest.mark.tc("TC-12")
def test_tc12_every_module_documents_its_purpose():
    for path in SOURCE_MODULES:
        assert ast.get_docstring(ast.parse(_read(path))), path


# ----- TC-13 Repeatability ------------------------------------------------------------------------
@pytest.mark.tc("TC-13")
def test_tc13_split_is_identical_across_runs(config, prepared):
    first, _ = prepared
    second, _ = tc.prepare_data(config)
    assert first.X_test.index.equals(second.X_test.index)
    assert first.y_train.equals(second.y_train)


@pytest.mark.tc("TC-13")
def test_tc13_training_is_deterministic(prepared, config):
    split, _ = prepared
    first = cm.train_logistic_regression(split, config).model.predict_proba(split.X_test)
    second = cm.train_logistic_regression(split, config).model.predict_proba(split.X_test)
    assert np.array_equal(first, second)
    xgb = [
        XGBClassifier(n_estimators=50, subsample=0.8, colsample_bytree=0.8, random_state=42)
        .fit(split.X_train, split.y_train)
        .predict_proba(split.X_test)
        for _ in range(2)
    ]
    assert np.array_equal(*xgb)


# ----- TC-23 Paradigm benchmark and experiment tracking ------------------------------------------
@pytest.fixture(scope="module")
def benchmark_run(tmp_path_factory):
    """ML vs DML vs QML benchmark with a small grid and a few epochs, written to a temporary folder."""
    if not RESEARCH_INSTALLED:
        pytest.skip("torch / pennylane not installed (requirements-research.txt)")
    benchmark = importlib.import_module("02_paradigm_benchmark")
    out = tmp_path_factory.mktemp("benchmark_run")
    config = _small_config()
    config["paradigms"] = {
        "include": ["ml", "dml", "qml"],
        "dml": {"hidden_layers": [16], "dropout": 0.2, "training": {"max_epochs": 3, "patience": 2}},
        "qml": {"n_qubits": 2, "n_layers": 1, "training": {"max_epochs": 1, "patience": 1, "batch_size": 256}},
    }
    with pytest.MonkeyPatch.context() as monkeypatch:
        (out / "plots").mkdir()
        monkeypatch.setattr(tc, "PLOTS_DIR", str(out / "plots"))
        monkeypatch.setattr(benchmark, "LEADERBOARD_FILE", str(out / "paradigm_metrics.json"))
        monkeypatch.setattr(benchmark, "COMPARISON_PLOT", str(out / "paradigm_comparison.png"))
        for paradigm in ("DML", "QML"):
            monkeypatch.setitem(pp.PARADIGM_MODEL_FILES, paradigm, str(out / f"{paradigm.lower()}_model.pkl"))
        previous_uri = _use_temporary_mlflow(out, benchmark.EXPERIMENT_NAME)
        try:
            leaderboard = benchmark.run_benchmark(config)
            runs = mlflow.search_runs(experiment_names=[benchmark.EXPERIMENT_NAME])
        finally:
            mlflow.set_tracking_uri(previous_uri)
    return SimpleNamespace(dir=out, leaderboard=leaderboard, runs=runs, module=benchmark)


@pytest.mark.tc("TC-23")
def test_tc23_benchmark_scores_and_logs_every_paradigm(benchmark_run):
    leaderboard = benchmark_run.leaderboard
    assert set(leaderboard["paradigm"]) == {"ML", "DML", "QML"}
    assert leaderboard["roc_auc"].is_monotonic_decreasing
    assert set(benchmark_run.runs["tags.paradigm"]) == {"ML", "DML", "QML"}
    assert (benchmark_run.runs["metrics.trainable_parameters"] > 0).all()
    saved = json.loads((benchmark_run.dir / "paradigm_metrics.json").read_text(encoding="utf-8"))
    assert len(saved) == len(leaderboard) == 4
    assert (benchmark_run.dir / "paradigm_comparison.png").exists()


@pytest.mark.tc("TC-23")
def test_tc23_benchmark_saves_loadable_serving_models(benchmark_run, prepared):
    split, _ = prepared
    for paradigm in ("DML", "QML"):
        payload = pp.load_paradigm_model(str(benchmark_run.dir / f"{paradigm.lower()}_model.pkl"), split.feature_names)
        assert payload["paradigm"] == paradigm
        assert payload["model"].predict_proba(split.X_test.head(5)).shape == (5, 2)


@pytest.mark.tc("TC-23")
def test_tc23_benchmark_cli_reports_bad_configuration(benchmark_run, monkeypatch, tmp_path):
    monkeypatch.setattr(tc, "PLOTS_DIR", str(tmp_path))
    bad_config = {**_small_config(), "paradigms": {"include": ["svm"]}}
    monkeypatch.setattr(tc, "load_config", lambda: bad_config)
    with warnings.catch_warnings(), pytest.raises(SystemExit) as stopped:
        benchmark_run.module.main()
    assert "Unknown paradigms" in str(stopped.value.code)


# ----- TC-24 Prediction latency ------------------------------------------------------------------
@pytest.mark.tc("TC-24")
def test_tc24_single_prediction_is_fast(request):
    model, features, encoders = pp.load_artifacts()
    customer = cp.get_preset_values(cp.DEFAULT_PRESET)
    pp.predict_churn(model, pp.preprocess_input(customer, encoders, features))  # warm-up
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        pp.predict_churn(model, pp.preprocess_input(customer, encoders, features))
        timings.append((time.perf_counter() - start) * 1000)
    median_ms = float(np.median(timings))
    _record(request, "median_prediction_ms", f"{median_ms:.1f}")
    assert median_ms < PREDICTION_LATENCY_BUDGET_MS


# ----- TC-25 Production model and encoder integrity ----------------------------------------------
@pytest.mark.tc("TC-25")
def test_tc25_production_model_matches_training_feature_order(prepared):
    split, _ = prepared
    model, features, encoders = pp.load_artifacts()
    assert features == split.feature_names
    assert model.n_features_in_ == len(features)
    assert set(pp.CATEGORICAL_COLUMNS) <= set(encoders)


@pytest.mark.tc("TC-25")
def test_tc25_production_payload_is_a_model_dictionary():
    with open(pp.DEFAULT_MODEL_FILE, "rb") as handle:
        payload = pickle.load(handle)
    assert set(payload) == {"model", "features_names"}
