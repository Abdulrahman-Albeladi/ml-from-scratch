"""Regularized linear classification and regression with pluggable losses."""

import numpy as np

from binary import BinaryClassifier
from gd import gd


class LossFunction:
    """Interface for losses optimized by :class:`LinearClassifier`."""

    def loss(self, Y, Yhat):
        """Return the aggregate loss for targets ``Y`` and predictions ``Yhat``."""
        raise NotImplementedError

    def lossGradient(self, X, Y, Yhat):
        """Return the gradient of the aggregate loss with respect to weights."""
        raise NotImplementedError


class SquaredLoss(LossFunction):
    """Half squared-error loss: ``0.5 * sum((Y - Yhat) ** 2)``."""

    def loss(self, Y, Yhat):
        residual = Y - Yhat
        return 0.5 * np.dot(residual, residual)

    def lossGradient(self, X, Y, Yhat):
        return X.T.dot(Yhat - Y)


class LogisticLoss(LossFunction):
    """Logistic loss: ``sum(log(1 + exp(-Y * Yhat)))``."""

    def loss(self, Y, Yhat):
        return np.sum(np.logaddexp(0, -Y * Yhat))

    def lossGradient(self, X, Y, Yhat):
        margins = Y * Yhat
        inverse_logistic = np.empty_like(margins, dtype=float)
        positive = margins >= 0
        inverse_logistic[positive] = np.exp(-margins[positive]) / (
            1 + np.exp(-margins[positive])
        )
        inverse_logistic[~positive] = 1 / (1 + np.exp(margins[~positive]))
        return -X.T.dot(Y * inverse_logistic)


class HingeLoss(LossFunction):
    """Hinge loss: ``sum(max(0, 1 - Y * Yhat))``."""

    def loss(self, Y, Yhat):
        return np.sum(np.maximum(0, 1 - Y * Yhat))

    def lossGradient(self, X, Y, Yhat):
        active = (1 - Y * Yhat) > 0
        return -X.T.dot(Y * active)


class LinearClassifier(BinaryClassifier):
    """Linear model with L2 regularization optimized by gradient descent.

    Required options are ``lossFunction``, ``lambda``, ``numIter``, and
    ``stepSize``. The loss function must implement the ``LossFunction`` API.
    """

    def __init__(self, opts):
        self.opts = opts
        self.reset()

    def reset(self):
        self.weights = 0

    def online(self):
        return False

    def __repr__(self):
        return "w=" + repr(self.weights)

    def predict(self, X):
        """Return the model margin; nonnegative margins represent class ``+1``."""
        if np.isscalar(self.weights):
            return 0
        return np.dot(X, self.weights)

    def getRepresentation(self):
        return self.weights

    def train(self, X, Y):
        """Fit the model by minimizing loss plus L2 regularization."""
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if Y.ndim != 1 or Y.shape[0] != X.shape[0]:
            raise ValueError("Y must be a one-dimensional vector matching X rows")

        loss_fn = self.opts["lossFunction"]
        lambd = self.opts["lambda"]
        num_iter = self.opts["numIter"]
        step_size = self.opts["stepSize"]

        if np.isscalar(self.weights):
            initial_weights = np.zeros(X.shape[1], dtype=float)
        else:
            initial_weights = np.asarray(self.weights, dtype=float)
            if initial_weights.shape != (X.shape[1],):
                raise ValueError("existing weights do not match the feature dimension")

        def func(weights):
            self.weights = weights
            predictions = X.dot(weights)
            return loss_fn.loss(Y, predictions) + (lambd / 2) * np.dot(weights, weights)

        def grad(weights):
            self.weights = weights
            predictions = X.dot(weights)
            return loss_fn.lossGradient(X, Y, predictions) + lambd * weights

        self.weights, self.trajectory = gd(
            func, grad, initial_weights, num_iter, step_size
        )
