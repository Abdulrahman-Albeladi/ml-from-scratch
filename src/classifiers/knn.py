"""k-nearest neighbors and radius-neighbors classifiers.

This module provides a small, independent implementation of neighbor-based
classification using Euclidean distance. It is organized for reuse in a
from-scratch machine learning repository rather than around course-specific
interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

NeighborMode = Literal["knn", "radius"]


@dataclass
class KNNClassifier:
    """Binary neighbor-based classifier using Euclidean distance.

    Labels are expected to be encoded as -1 and +1. Predictions are returned as
    signed vote totals, where positive values favor class +1 and negative values
    favor class -1. A zero vote indicates a tie or no neighbors.

    Parameters
    ----------
    n_neighbors:
        Number of nearest neighbors to use when ``mode='knn'``.
    radius:
        Distance threshold to use when ``mode='radius'``.
    mode:
        Either ``'knn'`` for k-nearest neighbors or ``'radius'`` for
        radius-neighbors voting.
    """

    n_neighbors: int = 5
    radius: float = 1.0
    mode: NeighborMode = "knn"

    def __post_init__(self) -> None:
        if self.mode not in {"knn", "radius"}:
            raise ValueError("mode must be either 'knn' or 'radius'")
        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Store training data for later neighbor queries."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if y.size == 0:
            raise ValueError("training data must contain at least one sample")

        self._X = X
        self._y = y
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return signed vote totals for each input sample."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array")
        if X.shape[1] != self._X.shape[1]:
            raise ValueError("X has a different number of features than training data")

        distances = self._pairwise_euclidean_distances(X, self._X)

        if self.mode == "knn":
            k = min(self.n_neighbors, self._X.shape[0])
            neighbor_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
            votes = self._y[neighbor_idx].sum(axis=1)
            return votes.astype(float)

        mask = distances < self.radius
        votes = (mask * self._y.reshape(1, -1)).sum(axis=1)
        return votes.astype(float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels in {-1, +1} from vote signs.

        Ties and empty-neighborhood cases map to 0.
        """
        scores = self.decision_function(X)
        return np.sign(scores).astype(int)

    def predict_one(self, x: np.ndarray) -> int:
        """Predict a single sample."""
        return int(self.predict(np.asarray(x))[0])

    def reset(self) -> None:
        """Clear stored training data."""
        self._X = None
        self._y = None

    @property
    def is_fitted(self) -> bool:
        return self._X is not None and self._y is not None

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("classifier has not been fitted")

    @staticmethod
    def _pairwise_euclidean_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between rows of A and rows of B."""
        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))
