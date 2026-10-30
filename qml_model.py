"""Quantum machine learning (paradigm 3 of 3): a hybrid variational quantum classifier (VQC).

Pipeline for one customer x:

    45 one-hot / standardised features --PCA--> n_qubits components --scale--> angles in [0, pi]

    |0>^n -> [ RY(angles) embedding -> StronglyEntanglingLayer(theta_l) ] x n_layers -> <Z_i>, i = 1..n

    logit = w . <Z> + b,   P(churn) = sigmoid(logit)

* Angle embedding puts one feature on each qubit as an RY rotation. Re-uploading the data
  before every layer makes the circuit output a richer Fourier series in the inputs than
  embedding once (Schuld, Sweke & Meyer 2021).
* Each StronglyEntanglingLayer applies a general single-qubit rotation Rot = RZ RY RZ to every
  qubit (3 angles each) and then a ring of CNOTs, so the ansatz entangles all qubits.
* Each Pauli-Z expectation lies in [-1, 1]. The small classical head (n_qubits + 1 weights)
  rescales them into a logit so probabilities can be calibrated under the weighted loss.

The circuit runs on PennyLane's default.qubit state-vector simulator on CPU and is trained
end to end by backpropagation, with the same loop as the deep model (TorchTabularClassifier).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from dml_model import TorchTabularClassifier, TrainingConfig, tabular_preprocessor


class VariationalQuantumClassifier(TorchTabularClassifier):
    """Data re-uploading VQC with a linear readout of all qubit Pauli-Z expectations."""

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 3,
        training: Optional[TrainingConfig] = None,
        random_state: int = 42,
    ):
        super().__init__(training=training, random_state=random_state)
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def _build_preprocessor(self) -> Pipeline:
        return Pipeline(
            [
                ("features", tabular_preprocessor()),
                ("pca", PCA(n_components=self.n_qubits, random_state=self.random_state)),
                # clip keeps unseen customers inside the angle range the circuit was trained on
                ("angles", MinMaxScaler(feature_range=(0.0, np.pi), clip=True)),
            ]
        )

    def _build_network(self, n_features: int) -> nn.Module:
        n_qubits, n_layers = self.n_qubits, self.n_layers
        wires = range(n_qubits)
        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for layer in range(n_layers):
                qml.AngleEmbedding(inputs, wires=wires, rotation="Y")
                qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=wires)
            return [qml.expval(qml.PauliZ(wire)) for wire in wires]

        weight_shapes = {"weights": qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)}
        return nn.Sequential(qml.qnn.TorchLayer(circuit, weight_shapes), nn.Linear(n_qubits, 1))

    @property
    def n_quantum_parameters(self) -> int:
        """Rotation angles in the circuit: n_layers x n_qubits x 3."""
        return self.n_layers * self.n_qubits * 3

    @property
    def n_classical_parameters(self) -> int:
        """Weights of the linear readout head."""
        return self.n_parameters - self.n_quantum_parameters
