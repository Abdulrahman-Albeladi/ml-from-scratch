from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Array = np.ndarray


class Activation:
    """Elementwise activation interface for hidden layers."""

    def forward(self, x: Array) -> Array:
        raise NotImplementedError

    def derivative(self, x: Array) -> Array:
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x: Array) -> Array:
        return np.maximum(x, 0.0)

    def derivative(self, x: Array) -> Array:
        return (x > 0.0).astype(np.float64)


class Identity(Activation):
    def forward(self, x: Array) -> Array:
        return x

    def derivative(self, x: Array) -> Array:
        return np.ones_like(x, dtype=np.float64)


class Loss:
    """Loss interface operating on column-major batches.

    Inputs are expected to have shape (output_dim, n_samples).
    """

    def value(self, y_true: Array, y_pred: Array) -> float:
        raise NotImplementedError

    def gradient(self, y_true: Array, y_pred: Array) -> Array:
        raise NotImplementedError


class MeanSquaredError(Loss):
    def value(self, y_true: Array, y_pred: Array) -> float:
        n_samples = y_true.shape[1]
        diff = y_pred - y_true
        return float(0.5 * np.sum(diff * diff) / n_samples)

    def gradient(self, y_true: Array, y_pred: Array) -> Array:
        n_samples = y_true.shape[1]
        return (y_pred - y_true) / n_samples


@dataclass
class LayerParameters:
    weights: Array
    bias: Array


class MLP:
    """A small fully connected network implemented with NumPy.

    The network uses the convention that each column is one example:
    X.shape == (input_dim, n_samples)
    Y.shape == (output_dim, n_samples)

    Hidden layers share one activation function. The output layer is linear,
    which keeps the implementation general for regression and for external
    loss/decision rules.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Sequence[int] = (64,),
        activation: Activation | None = None,
        loss: Loss | None = None,
        random_state: int | None = None,
    ) -> None:
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if any(width <= 0 for width in hidden_layers):
            raise ValueError("hidden layer widths must be positive")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation if activation is not None else ReLU()
        self.loss = loss if loss is not None else MeanSquaredError()
        self._rng = np.random.default_rng(random_state)

        layer_dims = [input_dim, *self.hidden_layers, output_dim]
        self.layers: List[LayerParameters] = []
        for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
            weights = self._rng.standard_normal((fan_out, fan_in)) * np.sqrt(
                2.0 / fan_in
            )
            bias = np.zeros((fan_out, 1), dtype=np.float64)
            self.layers.append(
                LayerParameters(weights=weights.astype(np.float64), bias=bias)
            )

    def summary(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        return [(layer.weights.shape, layer.bias.shape) for layer in self.layers]

    def forward(self, x: Array) -> Array:
        self._validate_inputs(x)
        activations = x
        for layer in self.layers[:-1]:
            activations = self.activation.forward(
                layer.weights @ activations + layer.bias
            )
        output_layer = self.layers[-1]
        return output_layer.weights @ activations + output_layer.bias

    def predict(self, x: Array) -> Array:
        scores = self.forward(x)
        return np.argmax(scores, axis=0)

    def loss_and_gradients(
        self, x: Array, y: Array
    ) -> Tuple[float, List[Array], List[Array]]:
        self._validate_inputs(x)
        self._validate_targets(y, x.shape[1])

        activations: List[Array] = [x]
        pre_activations: List[Array] = []

        current = x
        for layer in self.layers[:-1]:
            z = layer.weights @ current + layer.bias
            pre_activations.append(z)
            current = self.activation.forward(z)
            activations.append(current)

        output_layer = self.layers[-1]
        y_pred = output_layer.weights @ current + output_layer.bias
        loss_value = self.loss.value(y, y_pred)

        grad_w: List[Array] = []
        grad_b: List[Array] = []

        delta = self.loss.gradient(y, y_pred)
        grad_w.append(delta @ activations[-1].T)
        grad_b.append(np.sum(delta, axis=1, keepdims=True))

        for layer_index in range(len(self.layers) - 2, -1, -1):
            next_weights = self.layers[layer_index + 1].weights
            z = pre_activations[layer_index]
            delta = (next_weights.T @ delta) * self.activation.derivative(z)
            grad_w.append(delta @ activations[layer_index].T)
            grad_b.append(np.sum(delta, axis=1, keepdims=True))

        grad_w.reverse()
        grad_b.reverse()
        return loss_value, grad_w, grad_b

    def step(
        self, grad_w: Sequence[Array], grad_b: Sequence[Array], learning_rate: float
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if len(grad_w) != len(self.layers) or len(grad_b) != len(self.layers):
            raise ValueError("gradient lists must match the number of layers")

        for layer, dw, db in zip(self.layers, grad_w, grad_b):
            if dw.shape != layer.weights.shape or db.shape != layer.bias.shape:
                raise ValueError("gradient shapes must match parameter shapes")
            layer.weights -= learning_rate * dw
            layer.bias -= learning_rate * db

    def _validate_inputs(self, x: Array) -> None:
        if x.ndim != 2:
            raise ValueError("x must be a 2D array with shape (input_dim, n_samples)")
        if x.shape[0] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[0]}")

    def _validate_targets(self, y: Array, n_samples: int) -> None:
        if y.ndim != 2:
            raise ValueError("y must be a 2D array with shape (output_dim, n_samples)")
        if y.shape[0] != self.output_dim:
            raise ValueError(f"expected output_dim={self.output_dim}, got {y.shape[0]}")
        if y.shape[1] != n_samples:
            raise ValueError("x and y must contain the same number of samples")
