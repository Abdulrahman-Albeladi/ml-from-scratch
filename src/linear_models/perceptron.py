from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


@dataclass
class Perceptron:
    """Binary perceptron classifier for labels in {-1, +1}.

    This implementation is independent of any course-specific framework and
    exposes a small, reusable API suitable for examples and tests.

    Parameters
    ----------
    averaged:
        If True, prediction uses the averaged perceptron parameters accumulated
        during training.
    fit_intercept:
        If True, learn a scalar bias term.
    max_epochs:
        Maximum number of passes over the training data.
    shuffle:
        If True, shuffle training examples between epochs.
    random_state:
        Optional seed for reproducible shuffling.
    """

    averaged: bool = False
    fit_intercept: bool = True
    max_epochs: int = 10
    shuffle: bool = True
    random_state: Optional[int] = None
    weights_: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    bias_: float = field(default=0.0, init=False)
    n_updates_: int = field(default=0, init=False)
    n_features_in_: Optional[int] = field(default=None, init=False)
    _avg_weights_sum: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _avg_bias_sum: float = field(default=0.0, init=False, repr=False)
    _avg_steps: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        """Clear learned parameters and training counters."""
        self.weights_ = None
        self.bias_ = 0.0
        self.n_updates_ = 0
        self.n_features_in_ = None
        self._avg_weights_sum = None
        self._avg_bias_sum = 0.0
        self._avg_steps = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "Perceptron":
        """Train the perceptron on a binary classification dataset.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features).
        y:
            Labels encoded as -1 and +1.
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y must not be empty")

        unique_labels = np.unique(y_arr)
        if not np.all(np.isin(unique_labels, (-1, 1))):
            raise ValueError("Perceptron requires labels encoded as -1 and +1")

        self.reset()
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features
        self.weights_ = np.zeros(n_features, dtype=float)
        self._avg_weights_sum = np.zeros(n_features, dtype=float)

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n_samples)

        for _ in range(self.max_epochs):
            if self.shuffle:
                rng.shuffle(indices)

            mistakes = 0
            for idx in indices:
                self.partial_fit(X_arr[idx], y_arr[idx])
                if (
                    y_arr[idx] * self.decision_function(X_arr[idx], use_average=False)
                    <= 0
                ):
                    # This branch is unreachable after the update itself, so we
                    # track mistakes before updating in partial_fit instead.
                    pass

            # Count convergence by replaying current parameters over the dataset.
            margins = y_arr * self.decision_function(X_arr, use_average=False)
            mistakes = int(np.sum(margins <= 0))
            if mistakes == 0:
                break

        return self

    def partial_fit(self, x: ArrayLike, y: float) -> "Perceptron":
        """Update the model with one labeled example."""
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim != 1:
            raise ValueError("x must be a 1D feature vector")
        if y not in (-1, 1):
            raise ValueError("y must be either -1 or +1")

        if self.weights_ is None:
            self.n_features_in_ = x_arr.shape[0]
            self.weights_ = np.zeros(self.n_features_in_, dtype=float)
            self._avg_weights_sum = np.zeros(self.n_features_in_, dtype=float)
        elif x_arr.shape[0] != self.n_features_in_:
            raise ValueError("x has a different number of features than the model")

        margin = y * self.decision_function(x_arr, use_average=False)
        if margin <= 0:
            self.weights_ += y * x_arr
            if self.fit_intercept:
                self.bias_ += float(y)
            self.n_updates_ += 1

        self._accumulate_average()
        return self

    def decision_function(
        self, X: ArrayLike, use_average: Optional[bool] = None
    ) -> np.ndarray | float:
        """Return signed margins for one sample or a batch of samples."""
        if self.weights_ is None:
            raise ValueError("Model is not fitted")

        use_avg = self.averaged if use_average is None else use_average
        weights, bias = self._active_parameters(use_avg)
        X_arr = np.asarray(X, dtype=float)

        if X_arr.ndim == 1:
            if X_arr.shape[0] != self.n_features_in_:
                raise ValueError(
                    "Input has a different number of features than the model"
                )
            return float(np.dot(X_arr, weights) + bias)
        if X_arr.ndim == 2:
            if X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    "Input has a different number of features than the model"
                )
            return X_arr @ weights + bias
        raise ValueError("X must be a 1D or 2D array")

    def predict(self, X: ArrayLike) -> np.ndarray | int:
        """Predict labels in {-1, +1}."""
        margins = self.decision_function(X)
        if np.isscalar(margins):
            return 1 if margins >= 0 else -1
        return np.where(margins >= 0, 1, -1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return mean classification accuracy."""
        y_arr = np.asarray(y)
        preds = np.asarray(self.predict(X))
        if preds.shape != y_arr.shape:
            raise ValueError("Predictions and labels must have the same shape")
        return float(np.mean(preds == y_arr))

    def get_parameters(self) -> Tuple[np.ndarray, float]:
        """Return the parameters currently used for prediction."""
        if self.weights_ is None:
            raise ValueError("Model is not fitted")
        weights, bias = self._active_parameters(self.averaged)
        return weights.copy(), float(bias)

    def _accumulate_average(self) -> None:
        if self.weights_ is None or self._avg_weights_sum is None:
            return
        self._avg_weights_sum += self.weights_
        self._avg_bias_sum += self.bias_
        self._avg_steps += 1

    def _active_parameters(self, use_average: bool) -> Tuple[np.ndarray, float]:
        if self.weights_ is None:
            raise ValueError("Model is not fitted")
        if use_average and self._avg_steps > 0 and self._avg_weights_sum is not None:
            return (
                self._avg_weights_sum / self._avg_steps,
                self._avg_bias_sum / self._avg_steps,
            )
        return self.weights_, self.bias_


__all__ = ["Perceptron"]
