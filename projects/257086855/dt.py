"""Binary decision tree classifier using minimum classification error splits."""

import numpy as np

from binary import BinaryClassifier
import util


class DT(BinaryClassifier):
    """Decision tree for binary classification with thresholded features."""

    def __init__(self, opts):
        self.opts = opts
        self.isLeaf = True
        self.label = 1

    def online(self):
        """Decision trees are trained in batch mode."""
        return False

    def __repr__(self):
        """Return a canonical string representation of the tree."""
        return self.displayTree(0)

    def displayTree(self, depth):
        """Recursively format the tree."""
        indent = " " * (depth * 2)
        if self.isLeaf:
            return indent + "Leaf " + repr(self.label) + "\n"

        return (
            indent
            + "Branch "
            + repr(self.feature)
            + "\n"
            + self.left.displayTree(depth + 1)
            + self.right.displayTree(depth + 1)
        )

    def predict(self, X):
        """Predict a label by traversing feature branches at threshold 0.5."""
        if self.isLeaf:
            return self.label
        if X[self.feature] < 0.5:
            return self.left.predict(X)
        return self.right.predict(X)

    def trainDT(self, X, Y, maxDepth, used):
        """Recursively build a tree without reusing features along a path."""
        N, D = X.shape

        if maxDepth <= 0 or len(util.uniq(Y)) <= 1:
            self.isLeaf = True
            self.label = util.mode(Y)
            return

        bestFeature = -1
        bestError = N

        for d in range(D):
            if d in used:
                continue

            leftY = Y[X[:, d] < 0.5]
            rightY = Y[X[:, d] >= 0.5]

            # A split with an empty branch cannot be assigned a mode.
            if len(leftY) == 0 or len(rightY) == 0:
                continue

            error = np.sum(leftY != util.mode(leftY)) + np.sum(
                rightY != util.mode(rightY)
            )

            if error <= bestError:
                bestFeature = d
                bestError = error

        if bestFeature < 0:
            self.isLeaf = True
            self.label = util.mode(Y)
            return

        self.isLeaf = False
        self.feature = bestFeature
        next_used = used + [bestFeature]
        left_mask = X[:, bestFeature] < 0.5
        right_mask = ~left_mask

        self.left = DT({"maxDepth": maxDepth - 1})
        self.right = DT({"maxDepth": maxDepth - 1})
        self.left.trainDT(X[left_mask], Y[left_mask], maxDepth - 1, next_used)
        self.right.trainDT(X[right_mask], Y[right_mask], maxDepth - 1, next_used)

    def train(self, X, Y):
        """Build a decision tree from an ``N x D`` feature matrix and labels."""
        self.trainDT(X, Y, self.opts["maxDepth"], [])

    def getRepresentation(self):
        """Return the tree representation."""
        return self
