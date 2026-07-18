"""Multiclass softmax regression optimized with L-BFGS-B.

Adapted from code by Jatin Shah.
"""

import numpy as np
import scipy.optimize


class SoftmaxRegression:
    """Linear softmax classifier for column-oriented feature matrices."""

    def __init__(self, numClasses, exSize, opts=None):
        """
        Parameters
        ----------
        numClasses : int
            Number of output classes.
        exSize : int
            Number of input features per example.
        opts : dict, optional
            Optimization options. ``maxIter`` controls the L-BFGS-B iteration limit.
        """
        self.numClasses = numClasses
        self.exSize = exSize
        self.opts = {"maxIter": 400} if opts is None else dict(opts)
        self.W = np.zeros((numClasses, exSize))

    def reset(self, numClasses, exSize, opts=None):
        """Reset the model dimensions, options, and weights."""
        self.__init__(numClasses, exSize, opts)

    def setOption(self, optName, optVal):
        """Set an optimization option."""
        self.opts[optName] = optVal

    def cost(self, X, Y, W=None):
        """Return the average cross-entropy loss and its weight gradient.

        ``X`` has shape ``(exSize, n_examples)`` and ``Y`` contains one class
        index per example.
        """
        X = np.asarray(X)
        labels = np.asarray(Y).reshape(-1)

        if X.ndim != 2 or X.shape[0] != self.exSize:
            raise ValueError(
                "X must have shape (exSize, n_examples); "
                f"expected {self.exSize} feature rows."
            )

        n_examples = X.shape[1]
        if labels.size != n_examples:
            raise ValueError("Y must contain one label for each column of X.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Y must contain integer class indices.")
        if np.any(labels < 0) or np.any(labels >= self.numClasses):
            raise ValueError("Y contains a class index outside the configured range.")
        if n_examples == 0:
            raise ValueError("X and Y must contain at least one example.")

        if W is None:
            W = self.W
        W = np.asarray(W).reshape(self.numClasses, self.exSize)

        scores = W.dot(X)
        scores -= np.max(scores, axis=0, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=0, keepdims=True)

        example_indices = np.arange(n_examples)
        cost = -np.mean(np.log(probabilities[labels, example_indices]))

        gradient = probabilities.copy()
        gradient[labels, example_indices] -= 1.0
        gradient = gradient.dot(X.T) / n_examples

        return cost, gradient.ravel()

    def train(self, X, Y):
        """Fit model weights by minimizing the cross-entropy objective."""
        initial_weights = np.zeros(self.numClasses * self.exSize)
        max_iter = self.opts.get("maxIter", 400)

        result = scipy.optimize.minimize(
            lambda weights: self.cost(X, Y, weights),
            initial_weights,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter},
        )
        self.W = result.x.reshape(self.numClasses, self.exSize)

    def predict(self, X):
        """Predict the most likely class for each column of ``X``."""
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] != self.exSize:
            raise ValueError(
                "X must have shape (exSize, n_examples); "
                f"expected {self.exSize} feature rows."
            )

        scores = self.W.dot(X)
        return np.argmax(scores, axis=0)
