"""Principal component analysis for dense numeric matrices."""

from numbers import Integral

import numpy as np


def pca(X, K):
    """Compute the leading principal components of a data matrix.

    Parameters
    ----------
    X : array_like, shape (N, D)
        Matrix containing N observations with D features.
    K : int
        Maximum number of principal components to return. The result is
        limited to ``min(N, D)`` components.

    Returns
    -------
    P : ndarray, shape (N, K)
        Data projected onto the selected component directions.
    Z : ndarray, shape (D, K)
        Projection matrix whose columns are principal component directions.
    evals : ndarray, shape (K,)
        Eigenvalues sorted from largest to smallest.

    Notes
    -----
    Component directions are estimated from centered data. As in the original
    implementation, returned projections are computed as ``X @ Z``.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional data matrix.")
    if not isinstance(K, Integral):
        raise TypeError("K must be an integer.")
    if K < 0:
        raise ValueError("K must be non-negative.")

    n_samples, n_features = X.shape
    if n_samples < 2:
        raise ValueError("PCA requires at least two observations.")

    K = min(K, n_samples, n_features)

    centered = X - X.mean(axis=0)
    covariance = centered.T @ centered / (n_samples - 1)

    # The covariance matrix is symmetric, so eigh returns real eigenpairs.
    evals, evecs = np.linalg.eigh(covariance)
    order = np.argsort(evals)[::-1][:K]

    evals = evals[order]
    Z = evecs[:, order]
    P = X @ Z

    return P, Z, evals
