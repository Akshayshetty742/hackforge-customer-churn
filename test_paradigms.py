"""Tests for the deep (DML) and quantum (QML) churn models and the tri-paradigm benchmark wiring.

Skipped when torch / pennylane (requirements-research.txt) or mlflow (requirements-mlops.txt)
are not installed. Models are trained for a couple of epochs on small slices, so this runs in seconds.
"""
# pylint: disable=redefined-outer-name  # pytest injects fixtures by parameter name

import importlib
import importlib.util
import pickle

import numpy as np
import pytest
from sklearn.base import clone

import customer_presets as cp
import prediction_pipeline as pp

# Checked without importing: torch must be loaded through dml_model (see the note there on macOS libomp)
if any(importlib.util.find_spec(name) is None for name in ("torch", "pennylane")):
    pytest.skip("torch / pennylane not installed (requirements-research.txt)", allow_module_level=True)

# pylint: disable=wrong-import-position  # only importable once the optional dependencies are known to exist
from dml_model import ChurnMLPClassifier, TrainingConfig  # noqa: E402
from qml_model import VariationalQuantumClassifier  # noqa: E402
import torch  # noqa: E402  # pylint: disable=wrong-import-order

FAST = TrainingConfig(max_epochs=2, batch_size=64, patience=2)


@pytest.fixture(scope="module")
def encoded_sample():
    """First 600 dataset rows encoded with the saved encoders: (X, y)."""
    _, feature_names, encoders = pp.load_artifacts()
    df = pp.load_dataset().head(600)
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    return df[feature_names], df["Churn"].to_numpy()


@pytest.fixture(scope="module")
def fitted_mlp(encoded_sample):
    """Small MLP trained for two epochs."""
    X, y = encoded_sample
    return ChurnMLPClassifier(hidden_layers=(8,), training=FAST).fit(X, y)


@pytest.fixture(scope="module")
def fitted_vqc(encoded_sample):
    """Two-qubit, one-layer VQC trained for two epochs."""
    X, y = encoded_sample
    return VariationalQuantumClassifier(n_qubits=2, n_layers=1, training=FAST).fit(X.head(300), y[:300])


@pytest.mark.parametrize("model_fixture", ["fitted_mlp", "fitted_vqc"])
def test_predict_proba_is_a_valid_distribution(model_fixture, encoded_sample, request):
    model = request.getfixturevalue(model_fixture)
    X, _ = encoded_sample
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(np.unique(model.predict(X))) <= {0, 1}


@pytest.mark.parametrize("model_fixture", ["fitted_mlp", "fitted_vqc"])
def test_pickle_round_trip_keeps_predictions(model_fixture, encoded_sample, request):
    model = request.getfixturevalue(model_fixture)
    X, _ = encoded_sample
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(restored.predict_proba(X), model.predict_proba(X))


@pytest.mark.parametrize("model_fixture", ["fitted_mlp", "fitted_vqc"])
def test_early_stopping_restores_best_epoch(model_fixture, request):
    model = request.getfixturevalue(model_fixture)
    assert 1 <= model.n_epochs_ <= FAST.max_epochs
    assert model.best_val_loss_ == pytest.approx(min(val for _, val in model.history_))


def test_clone_keeps_hyperparameters():
    model = ChurnMLPClassifier(hidden_layers=(16, 8), dropout=0.1, training=FAST, random_state=7)
    params = clone(model).get_params()
    assert params["hidden_layers"] == (16, 8)
    assert params["dropout"] == 0.1
    assert params["random_state"] == 7


def test_mlp_parameter_count_matches_architecture(fitted_mlp):
    n_in = fitted_mlp.n_features_
    # Linear(n_in, 8) + BatchNorm1d(8) (scale and shift) + Linear(8, 1)
    assert fitted_mlp.n_parameters == (n_in * 8 + 8) + 2 * 8 + (8 + 1)


def test_vqc_parameter_split():
    model = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
    network = model._build_network(4)  # pylint: disable=protected-access
    total = sum(p.numel() for p in network.parameters())
    assert model.n_quantum_parameters == 2 * 4 * 3  # one Rot (3 angles) per qubit per layer
    assert total - model.n_quantum_parameters == 4 + 1  # linear readout of 4 expectations


def test_vqc_features_are_rotation_angles(fitted_vqc, encoded_sample):
    X, _ = encoded_sample
    angles = fitted_vqc.preprocessor_.transform(X)  # includes rows the scaler never saw
    assert angles.shape[1] == fitted_vqc.n_qubits
    assert angles.min() >= 0.0
    assert angles.max() <= np.pi + 1e-9


def test_single_qubit_circuit_reduces_to_cosine():
    """With zero rotation weights the circuit is RY(x)|0>, whose Pauli-Z expectation is cos(x)."""
    quantum_layer = VariationalQuantumClassifier(n_qubits=1, n_layers=1)._build_network(1)[0]  # pylint: disable=protected-access
    with torch.no_grad():
        quantum_layer.weights.zero_()
        x = torch.linspace(0.0, np.pi, 7).reshape(-1, 1)
        expectation = quantum_layer(x).reshape(-1)
    assert torch.allclose(expectation, torch.cos(x.reshape(-1)), atol=1e-6)


def test_benchmark_rejects_unknown_paradigm():
    pytest.importorskip("mlflow")
    benchmark = importlib.import_module("02_paradigm_benchmark")
    config = {"prepare": {"random_state": 42}, "paradigms": {"include": ["ml", "quantum-annealing"]}}
    with pytest.raises(ValueError, match="quantum-annealing"):
        benchmark.build_plan(config)


def test_benchmark_plan_tags_each_paradigm():
    pytest.importorskip("mlflow")
    benchmark = importlib.import_module("02_paradigm_benchmark")
    plan = benchmark.build_plan({"prepare": {"random_state": 42}, "train": {}})
    assert [tag for tag, _ in plan] == ["ML", "ML", "DML", "QML"]


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")  # only shapes matter here
def test_classical_parameter_count_for_linear_model(encoded_sample):
    from sklearn.linear_model import LogisticRegression  # pylint: disable=import-outside-toplevel

    X, y = encoded_sample
    model = LogisticRegression(max_iter=500).fit(X, y)
    assert pp.count_model_parameters(model) == X.shape[1] + 1


def test_parameter_count_of_torch_models(fitted_mlp, fitted_vqc):
    assert pp.count_model_parameters(fitted_mlp) == fitted_mlp.n_parameters
    assert pp.count_model_parameters(fitted_vqc) == fitted_vqc.n_parameters


@pytest.fixture(scope="module")
def serving_models():
    """Production XGBoost plus the saved DML / QML models, keyed by paradigm."""
    model, feature_names, encoders = pp.load_artifacts()
    models = {"ML": model}
    for paradigm, path in pp.PARADIGM_MODEL_FILES.items():
        models[paradigm] = pp.load_paradigm_model(path, feature_names)["model"]
    return models, feature_names, encoders


def test_saved_paradigm_models_carry_their_test_metrics():
    _, feature_names, _ = pp.load_artifacts()
    for paradigm, path in pp.PARADIGM_MODEL_FILES.items():
        payload = pp.load_paradigm_model(path, feature_names)
        assert payload["paradigm"] == paradigm
        assert 0.5 < payload["test_metrics"]["roc_auc"] <= 1.0
        assert payload["trainable_parameters"] == pp.count_model_parameters(payload["model"])


def test_predict_all_paradigms_scores_one_customer_with_every_model(serving_models):
    models, feature_names, encoders = serving_models
    customer = pp.preprocess_input(cp.get_preset_values("High risk: new fiber customer"), encoders, feature_names)
    results = pp.predict_all_paradigms(customer, models, threshold=0.5)
    assert [r["paradigm"] for r in results] == ["ML", "DML", "QML"]
    for result in results:
        assert 0.0 <= result["churn_probability"] <= 1.0
        assert result["prediction"] == int(result["churn_probability"] >= 0.5)
        assert result["label"] == ("Churn" if result["prediction"] else "Stay")
        assert result["risk_tier"] == pp.get_risk_tier(result["churn_probability"])
    # The textbook high-risk profile is flagged by every paradigm
    assert {r["label"] for r in results} == {"Churn"}


def test_threshold_applies_to_every_paradigm(serving_models):
    models, feature_names, encoders = serving_models
    customer = pp.preprocess_input(cp.get_preset_values("High risk: new fiber customer"), encoders, feature_names)
    assert {r["label"] for r in pp.predict_all_paradigms(customer, models, threshold=0.999)} == {"Stay"}


def test_load_paradigm_model_errors(tmp_path):
    _, feature_names, _ = pp.load_artifacts()
    with pytest.raises(FileNotFoundError):
        pp.load_paradigm_model(str(tmp_path / "missing.pkl"), feature_names)
    with pytest.raises(ValueError, match="different input features"):
        pp.load_paradigm_model(pp.PARADIGM_MODEL_FILES["DML"], list(reversed(feature_names)))
    bogus = tmp_path / "bogus.pkl"
    bogus.write_bytes(pickle.dumps(["not", "a", "payload"]))
    with pytest.raises(ValueError, match="not a saved model payload"):
        pp.load_paradigm_model(str(bogus), feature_names)
