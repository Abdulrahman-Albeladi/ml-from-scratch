"""Implementation of a k-nearest-neighbor classifier."""

import numpy as np

from binary import BinaryClassifier


class KNN(BinaryClassifier):
    """Binary classifier supporting k-nearest-neighbor and epsilon-ball voting."""

    def __init__(self, opts):
        self.opts = opts
        self.reset()

    def reset(self):
        """Clear stored training examples and labels."""
        self.trX = np.zeros((0, 0))
        self.trY = np.zeros(0)

    def online(self):
        """Nearest-neighbor training is not incremental in this implementation."""
        return False

    def __repr__(self):
        return "KNN(opts=%r)" % self.opts

    def predict(self, X):
        """Return the signed vote of neighbors near ``X``.

        In k-nearest-neighbor mode, the vote is the sum of the labels of the
        ``K`` closest training examples. In epsilon-ball mode, the vote is the
        sum of labels whose Euclidean distance is strictly less than ``eps``.
        """
        if self.trY.size == 0:
            return 0

        distances = np.sqrt(np.sum((self.trX - X) ** 2, axis=1))

        if self.opts["isKNN"]:
            neighbors = np.argsort(distances)[: self.opts["K"]]
        else:
            neighbors = np.where(distances < self.opts["eps"])[0]

        return np.sum(self.trY[neighbors])

    def getRepresentation(self):
        """Return the stored training examples and labels."""
        return self.trX, self.trY

    def train(self, X, Y):
        """Store training examples and their corresponding labels."""
        self.trX = X
        self.trY = Y
