from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class LossFunction(Protocol):
    """Interface for margin-based losses used by linear models."""

    def loss(self, y_true: FloatArray, scores: FloatArray) -> float: ...

    def gradient(
        self, x: FloatArray, y_true: FloatArray, scores: FloatArray
    ) -> FloatArray: ...


class SquaredLoss:
    """Squared error loss for linear regression-style objectives."""

    def loss(self, y_true: FloatArray, scores: FloatArray) -> float:
        residual = y_true - scores
        return float(0.5 * np.dot(residual, residual))

    def gradient(
        self, x: FloatArray, y_true: FloatArray, scores: FloatArray
    ) -> FloatArray:
        residual = y_true - scores
        return -(x.T @ residual)


class LogisticLoss:
    """Binary logistic loss for labels encoded as -1 and +1."""

    def loss(self, y_true: FloatArray, scores: FloatArray) -> float:
        margins = y_true * scores
        return float(np.sum(np.logaddexp(0.0, -margins)))

    def gradient(
        self, x: FloatArray, y_true: FloatArray, scores: FloatArray
    ) -> FloatArray:
        margins = y_true * scores
        weights = -y_true / (1.0 + np.exp(margins))
        return x.T @ weights


class HingeLoss:
    """Binary hinge loss for labels encoded as -1 and +1."""

    def loss(self, y_true: FloatArray, scores: FloatArray) -> float:
        margins = y_true * scores
        return float(np.sum(np.maximum(0.0, 1.0 - margins)))

    def gradient(
        self, x: FloatArray, y_true: FloatArray, scores: FloatArray
    ) -> FloatArray:
        active = (y_true * scores) < 1.0
        return -(x.T @ (y_true * active.astype(np.float64)))


@dataclass
class GradientDescentResult:
    weights: FloatArray
    objective_history: List[float]


class LinearModel:
    """L2-regularized linear model trained with batch gradient descent.

    This implementation is intentionally small and independent of any
    assignment-specific framework. For classification losses such as logistic
    and hinge, labels are expected to be encoded as -1 and +1.
    """

    def __init__(
        self,
        loss: LossFunction,
        regularization_strength: float = 0.0,
        learning_rate: float = 1e-2,
        max_iter: int = 1000,
        fit_intercept: bool = True,
    ) -> None:
        if regularization_strength < 0:
            raise ValueError("regularization_strength must be non-negative")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.loss = loss
        self.regularization_strength = regularization_strength
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

        self.weights_: Optional[FloatArray] = None
        self.objective_history_: List[float] = []

    def fit(self, x: ArrayLike, y: ArrayLike) -> "LinearModel":
        x_arr, y_arr = _validate_training_data(x, y)
        design = _add_intercept_column(x_arr) if self.fit_intercept else x_arr

        weights = np.zeros(design.shape[1], dtype=np.float64)
        history: List[float] = []

        for _ in range(self.max_iter):
            scores = design @ weights
            objective = self.loss.loss(y_arr, scores) + self._regularization_penalty(
                weights
            )
            gradient = self.loss.gradient(
                design, y_arr, scores
            ) + self._regularization_gradient(weights)

            history.append(float(objective))
            weights = weights - self.learning_rate * gradient

        self.weights_ = weights
        self.objective_history_ = history
        return self

    def decision_function(self, x: ArrayLike) -> FloatArray:
        if self.weights_ is None:
            raise ValueError("model is not fitted")

        x_arr = _as_2d_float_array(x)
        design = _add_intercept_column(x_arr) if self.fit_intercept else x_arr
        if design.shape[1] != self.weights_.shape[0]:
            raise ValueError("input feature dimension does not match fitted model")
        return design @ self.weights_

    def predict(self, x: ArrayLike) -> FloatArray:
        scores = self.decision_function(x)
        return np.where(scores >= 0.0, 1.0, -1.0)

    def fit_gradient_descent(self, x: ArrayLike, y: ArrayLike) -> GradientDescentResult:
        self.fit(x, y)
        assert self.weights_ is not None  # nosec B101
        return GradientDescentResult(
            weights=self.weights_.copy(),
            objective_history=list(self.objective_history_),
        )

    def _regularization_penalty(self, weights: FloatArray) -> float:
        if self.regularization_strength == 0.0:
            return 0.0
        regularized = self._regularized_weights(weights)
        return float(
            0.5 * self.regularization_strength * np.dot(regularized, regularized)
        )

    def _regularization_gradient(self, weights: FloatArray) -> FloatArray:
        gradient = np.zeros_like(weights)
        gradient_slice = self._regularized_weights(weights)
        if self.fit_intercept:
            gradient[1:] = self.regularization_strength * gradient_slice
        else:
            gradient[:] = self.regularization_strength * gradient_slice
        return gradient

    def _regularized_weights(self, weights: FloatArray) -> FloatArray:
        return weights[1:] if self.fit_intercept else weights


class LinearClassifier(LinearModel):
    """Alias for binary linear classification use cases."""


class LinearRegressor(LinearModel):
    """Linear model wrapper with regression-oriented prediction semantics."""

    def predict(self, x: ArrayLike) -> FloatArray:
        return self.decision_function(x)


def _validate_training_data(
    x: ArrayLike, y: ArrayLike
) -> Tuple[FloatArray, FloatArray]:
    x_arr = _as_2d_float_array(x)
    y_arr = np.asarray(y, dtype=np.float64)

    if y_arr.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    return x_arr, y_arr


def _as_2d_float_array(x: ArrayLike) -> FloatArray:
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
    if x_arr.ndim != 2:
        raise ValueError("x must be a one- or two-dimensional array")
    return x_arr


def _add_intercept_column(x: FloatArray) -> FloatArray:
    intercept = np.ones((x.shape[0], 1), dtype=np.float64)
    return np.hstack([intercept, x])


__all__: Sequence[str] = [
    "GradientDescentResult",
    "HingeLoss",
    "LinearClassifier",
    "LinearModel",
    "LinearRegressor",
    "LogisticLoss",
    "LossFunction",
    "SquaredLoss",
]
