"""Principal Component Analysis (PCA).

This module provides a small NumPy-based PCA implementation suitable for
educational use and lightweight experiments. It is intentionally independent
of any course-specific interfaces.
"""

from __future__ import annotations

import numpy as np


class PCA:
    """Principal Component Analysis using an eigen-decomposition of covariance.

    Parameters
    ----------
    n_components:
        Number of principal components to retain. Must be between 1 and
        ``min(n_samples, n_features)`` when fitting.

    Notes
    -----
    The implementation centers features before computing the sample covariance
    matrix. Components are ordered by decreasing explained variance.
    """

    def __init__(self, n_components: int):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit PCA to a 2D array of shape (n_samples, n_features)."""
        X = self._validate_input(X)
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)
        if self.n_components > max_components:
            raise ValueError(
                "n_components must be less than or equal to min(n_samples, n_features)"
            )
        if n_samples < 2:
            raise ValueError("PCA requires at least two samples")

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        covariance = (X_centered.T @ X_centered) / (n_samples - 1)

        # Covariance is symmetric, so eigh is more appropriate than eig and
        # avoids unnecessary complex outputs from numerical noise.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[order][: self.n_components]
        eigenvectors = eigenvectors[:, order][:, : self.n_components]

        self.explained_variance_ = eigenvalues
        self.components_ = eigenvectors
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data into the fitted principal-component subspace."""
        self._check_is_fitted()
        X = self._validate_input(X)
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and return the projected data."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_projected: np.ndarray) -> np.ndarray:
        """Map projected data back to the original feature space."""
        self._check_is_fitted()
        X_projected = np.asarray(X_projected, dtype=float)
        if X_projected.ndim != 2:
            raise ValueError("X_projected must be a 2D array")
        if X_projected.shape[1] != self.n_components:
            raise ValueError("X_projected has incompatible number of components")
        return X_projected @ self.components_.T + self.mean_

    def _check_is_fitted(self) -> None:
        if (
            self.mean_ is None
            or self.components_ is None
            or self.explained_variance_ is None
        ):
            raise ValueError("PCA instance is not fitted yet")

    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must have at least one sample and one feature")
        return X


def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a PCA projection.

    Parameters
    ----------
    X:
        Input data of shape ``(n_samples, n_features)``.
    n_components:
        Number of principal components to retain.

    Returns
    -------
    projected:
        Data projected into the principal-component subspace with shape
        ``(n_samples, n_components)``.
    components:
        Projection matrix of shape ``(n_features, n_components)``.
    explained_variance:
        Eigenvalues associated with the retained components.
    """
    model = PCA(n_components=n_components)
    projected = model.fit_transform(X)
    return projected, model.components_, model.explained_variance_
