"""Deep machine learning (paradigm 2 of 3): a PyTorch multi-layer perceptron for tabular churn data.

TorchTabularClassifier holds everything the deep and quantum models share: input preprocessing,
class-weighted binary cross-entropy training with Adam and early stopping, and a scikit-learn
style fit / predict_proba API, so both plug into training_common.evaluate_and_log_run and
mlflow.sklearn like the classical models. Subclasses only build the preprocessing and network.
"""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# On macOS torch and XGBoost bundle separate libomp copies. Loading torch's first makes the
# XGBoost model segfault when it is unpickled, so XGBoost's runtime is loaded before torch.
# See also single_threaded_torch() below. On Linux both use the system libgomp and this is a no-op.
import xgboost  # noqa: F401  # pylint: disable=unused-import
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from torch import nn

import prediction_pipeline as pp


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation settings shared by the deep and quantum models."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 15
    validation_fraction: float = 0.15


class _SingleThreadedTorch:  # pylint: disable=too-few-public-methods  # a callable guard
    """Reference-counted torch.set_num_threads(1) that is safe across threads.

    On macOS, torch and XGBoost / scikit-learn each ship their own libomp. Once XGBoost has
    started its OpenMP thread pool, a multi-threaded torch kernel in the same process can wait
    forever on the other runtime's barrier. These networks are far too small to gain from
    intra-op threads, and one thread keeps torch out of OpenMP entirely (and deterministic).

    The setting is process wide, so with concurrent users (one Streamlit script thread per
    browser session) the previous value is only restored when the last user leaves.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._previous = 1

    @contextmanager
    def __call__(self) -> Iterator[None]:
        with self._lock:
            if self._active == 0:
                self._previous = torch.get_num_threads()
            self._active += 1
            torch.set_num_threads(1)
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
                if self._active == 0:
                    torch.set_num_threads(self._previous)


single_threaded_torch = _SingleThreadedTorch()


def tabular_preprocessor() -> ColumnTransformer:
    """One-hot encodes the label-encoded categorical codes and standardises the numeric columns.

    Label-encoded codes impose a meaningless order (e.g. payment method 0 < 3), which trees
    ignore but a weighted sum does not, so neural and quantum models get one-hot inputs.
    """
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), pp.CATEGORICAL_COLUMNS),
            ("numeric", StandardScaler(), pp.NUMERICAL_COLUMNS),
        ]
    )


class TorchTabularClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier trained with PyTorch on preprocessed tabular features.

    The positive (churn) class is up-weighted by the negative / positive ratio of the training
    labels, the same imbalance correction XGBoost gets through scale_pos_weight.
    """

    def __init__(self, training: Optional[TrainingConfig] = None, random_state: int = 42):
        self.training = training
        self.random_state = random_state

    # ----- hooks for subclasses -------------------------------------------------------------
    def _build_preprocessor(self) -> Any:
        return tabular_preprocessor()

    def _build_network(self, n_features: int) -> nn.Module:
        """Network mapping (batch, n_features) inputs to (batch, 1) or (batch,) churn logits."""
        raise NotImplementedError

    # ----- training -------------------------------------------------------------------------
    def _to_tensor(self, X: pd.DataFrame) -> torch.Tensor:
        return torch.as_tensor(np.asarray(self.preprocessor_.transform(X)), dtype=torch.float32)

    def _logits(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network_(inputs).reshape(-1)

    def _batches(self, n_rows: int, batch_size: int, generator: torch.Generator) -> List[torch.Tensor]:
        order = torch.randperm(n_rows, generator=generator)
        # BatchNorm cannot normalise a single row, so a trailing batch of one is skipped
        return [batch for batch in torch.split(order, batch_size) if len(batch) > 1]

    def fit(self, X: pd.DataFrame, y: Sequence[int]) -> "TorchTabularClassifier":
        """Trains on X / y, holding out a stratified validation slice for early stopping."""
        with single_threaded_torch():
            return self._fit(X, y)

    def _tensors(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._to_tensor(X), torch.as_tensor(y, dtype=torch.float32)

    def _train_epoch(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        generator: torch.Generator,
    ) -> float:
        """One shuffled pass over the training data; returns the mean training loss."""
        inputs, targets = data
        self.network_.train()
        total = 0.0
        for batch in self._batches(len(targets), batch_size, generator):
            optimizer.zero_grad()
            loss = loss_fn(self._logits(inputs[batch]), targets[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        return total / len(targets)

    def _validation_loss(self, data: Tuple[torch.Tensor, torch.Tensor], loss_fn: nn.Module) -> float:
        self.network_.eval()
        with torch.no_grad():
            return float(loss_fn(self._logits(data[0]), data[1]))

    def _fit(self, X: pd.DataFrame, y: Sequence[int]) -> "TorchTabularClassifier":
        config = self.training or TrainingConfig()
        torch.manual_seed(self.random_state)

        y = np.asarray(y)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=config.validation_fraction, stratify=y, random_state=self.random_state
        )
        self.preprocessor_ = self._build_preprocessor().fit(X_fit)  # fitted on the training part only
        train_data, val_data = self._tensors(X_fit, y_fit), self._tensors(X_val, y_val)
        self.n_features_ = int(train_data[0].shape[1])
        self.network_ = self._build_network(self.n_features_)

        targets = train_data[1]
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=(targets == 0).sum() / (targets == 1).sum().clamp(min=1))
        optimizer = torch.optim.Adam(
            self.network_.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        self.history_: List[Tuple[float, float]] = []
        self.network_.load_state_dict(self._train_with_early_stopping(train_data, val_data, loss_fn, optimizer, config))
        self.network_.eval()
        self.classes_ = np.array([0, 1])
        return self

    def _train_with_early_stopping(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
    ) -> Dict[str, torch.Tensor]:
        """Trains until validation loss stalls for `patience` epochs; returns the best epoch's weights."""
        generator = torch.Generator().manual_seed(self.random_state)
        best_state, stale_epochs = None, 0
        for _ in range(config.max_epochs):
            train_loss = self._train_epoch(train_data, loss_fn, optimizer, config.batch_size, generator)
            val_loss = self._validation_loss(val_data, loss_fn)
            improved = not self.history_ or val_loss < self.best_val_loss_
            self.history_.append((train_loss, val_loss))
            if improved:
                best_state, stale_epochs = copy.deepcopy(self.network_.state_dict()), 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break
        return best_state

    @property
    def n_epochs_(self) -> int:
        """Epochs run before early stopping."""
        return len(self.history_)

    @property
    def best_val_loss_(self) -> float:
        """Lowest validation loss; the fitted network holds the weights from that epoch."""
        return min(val_loss for _, val_loss in self.history_)

    # ----- inference ------------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Columns: P(stay), P(churn)."""
        check_is_fitted(self, "network_")
        with single_threaded_torch(), torch.no_grad():
            churn = torch.sigmoid(self._logits(self._to_tensor(X))).numpy().astype(np.float64)
        return np.column_stack([1.0 - churn, churn])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Churn (1) when P(churn) >= 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def n_parameters(self) -> int:
        """Trainable parameters of the fitted network."""
        check_is_fitted(self, "network_")
        return int(sum(p.numel() for p in self.network_.parameters() if p.requires_grad))

    # ----- pickling: store weights, rebuild the network on load -----------------------------
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        if "network_" in state:
            state["network_state_"] = state.pop("network_").state_dict()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        network_state = state.pop("network_state_", None)
        self.__dict__.update(state)
        if network_state is not None:
            self.network_ = self._build_network(self.n_features_)
            self.network_.load_state_dict(network_state)
            self.network_.eval()


class ChurnMLPClassifier(TorchTabularClassifier):
    """Fully connected network: [Linear -> BatchNorm -> ReLU -> Dropout] per hidden layer, then a logit."""

    def __init__(
        self,
        hidden_layers: Tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
        training: Optional[TrainingConfig] = None,
        random_state: int = 42,
    ):
        super().__init__(training=training, random_state=random_state)
        self.hidden_layers = hidden_layers
        self.dropout = dropout

    def _build_network(self, n_features: int) -> nn.Module:
        layers: List[nn.Module] = []
        width = n_features
        for units in self.hidden_layers:
            layers += [nn.Linear(width, units), nn.BatchNorm1d(units), nn.ReLU(), nn.Dropout(self.dropout)]
            width = units
        layers.append(nn.Linear(width, 1))
        return nn.Sequential(*layers)
