"""Multiclass classification strategies built from binary classifiers.

Binary classifiers supplied to these wrappers must implement either ``fit(X, y)``
or ``train(X, y)`` and provide ``predict_proba(X)``. Training labels use ``+1``
for the positive class and ``-1`` for the negative class.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

ClassifierFactory = Callable[[], Any]


def _fit_binary_classifier(classifier: Any, X: np.ndarray, y: np.ndarray) -> None:
    """Train a classifier supporting either the scikit-learn or course API."""
    fit = getattr(classifier, "fit", None)
    if callable(fit):
        fit(X, y)
        return

    train = getattr(classifier, "train", None)
    if callable(train):
        train(X, y)
        return

    raise TypeError("Binary classifiers must define fit(X, y) or train(X, y).")


def _positive_probability(classifier: Any, X: np.ndarray) -> float:
    """Return the predicted probability associated with binary label ``+1``."""
    predict_proba = getattr(classifier, "predict_proba", None)
    if not callable(predict_proba):
        raise TypeError("Binary classifiers must define predict_proba(X).")

    probabilities = np.asarray(predict_proba(X))
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != 1
        or probabilities.shape[1] < 2
    ):
        raise ValueError(
            "predict_proba(X) must return one row with two class probabilities."
        )

    classes = getattr(classifier, "classes_", None)
    if classes is not None:
        positive_indices = np.flatnonzero(np.asarray(classes) == 1)
        if positive_indices.size == 1:
            return float(probabilities[0, positive_indices[0]])

    # Binary classifiers without classes_ follow the conventional [-1, +1] order.
    return float(probabilities[0, 1])


class OAA:
    """One-against-all multiclass classifier."""

    def __init__(self, K: int, mkClassifier: ClassifierFactory):
        self.K = K
        self.f = [mkClassifier() for _ in range(K)]

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        X = np.asarray(X)
        Y = np.asarray(Y)

        for k, classifier in enumerate(self.f):
            labels = 2 * (Y == k) - 1
            _fit_binary_classifier(classifier, X, labels)

    def predict(self, X: np.ndarray, useZeroOne: bool = False) -> int:
        X = np.asarray(X).reshape(1, -1)
        vote = np.zeros(self.K)

        for k, classifier in enumerate(self.f):
            probability = _positive_probability(classifier, X)
            vote[k] = float(probability > 0.5) if useZeroOne else probability

        return int(np.argmax(vote))

    def predictAll(self, X: np.ndarray, useZeroOne: bool = False) -> np.ndarray:
        X = np.asarray(X)
        predictions = np.zeros(X.shape[0], dtype=int)
        for index, row in enumerate(X):
            predictions[index] = self.predict(row, useZeroOne)
        return predictions


class AVA:
    """One-against-one multiclass classifier."""

    def __init__(self, K: int, mkClassifier: ClassifierFactory):
        self.K = K
        self.f = [[] for _ in range(K)]
        for i in range(K):
            for _j in range(i):
                self.f[i].append(mkClassifier())

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        X = np.asarray(X)
        Y = np.asarray(Y)

        for i in range(self.K):
            for j in range(i):
                mask = (Y == i) | (Y == j)
                pair_X = X[mask]
                # Class j is positive so its probability contributes to j's vote.
                pair_Y = 2 * (Y[mask] == j) - 1
                _fit_binary_classifier(self.f[i][j], pair_X, pair_Y)

    def predict(self, X: np.ndarray, useZeroOne: bool = False) -> int:
        X = np.asarray(X).reshape(1, -1)
        vote = np.zeros(self.K)

        for i in range(self.K):
            for j in range(i):
                probability = _positive_probability(self.f[i][j], X)
                score = float(probability > 0.5) if useZeroOne else probability
                vote[j] += score
                vote[i] -= score

        return int(np.argmax(vote))

    def predictAll(self, X: np.ndarray, useZeroOne: bool = False) -> np.ndarray:
        X = np.asarray(X)
        predictions = np.zeros(X.shape[0], dtype=int)
        for index, row in enumerate(X):
            predictions[index] = self.predict(row, useZeroOne)
        return predictions


class TreeNode:
    """A node in a binary class-partition tree."""

    def __init__(self) -> None:
        self.isLeaf = True
        self.label = 0
        self.info: Any = None
        self.left: TreeNode | None = None
        self.right: TreeNode | None = None

    def setLeafLabel(self, label: int) -> None:
        self.isLeaf = True
        self.label = label
        self.left = None
        self.right = None

    def setChildren(self, left: "TreeNode", right: "TreeNode") -> None:
        self.isLeaf = False
        self.left = left
        self.right = right

    def getLabel(self) -> int:
        if not self.isLeaf:
            raise ValueError("Cannot retrieve a label from an internal node.")
        return self.label

    def getLeft(self) -> "TreeNode":
        if self.isLeaf or self.left is None:
            raise ValueError("Cannot retrieve children from a leaf node.")
        return self.left

    def getRight(self) -> "TreeNode":
        if self.isLeaf or self.right is None:
            raise ValueError("Cannot retrieve children from a leaf node.")
        return self.right

    def setNodeInfo(self, info: Any) -> None:
        self.info = info

    def getNodeInfo(self) -> Any:
        return self.info

    def iterAllLabels(self) -> Iterable[int]:
        if self.isLeaf:
            yield self.label
            return
        yield from self.getLeft().iterAllLabels()
        yield from self.getRight().iterAllLabels()

    def iterNodes(self) -> Iterable["TreeNode"]:
        yield self
        if not self.isLeaf:
            yield from self.getLeft().iterNodes()
            yield from self.getRight().iterNodes()

    def __repr__(self) -> str:
        if self.isLeaf:
            return str(self.label)
        return f"[{self.getLeft()!r} {self.getRight()!r}]"


def makeBalancedTree(allK: Sequence[int]) -> TreeNode:
    """Build a balanced binary tree whose leaves are the supplied class labels."""
    if not allK:
        raise ValueError("Cannot construct a class tree with no labels.")

    tree = TreeNode()
    if len(allK) == 1:
        tree.setLeafLabel(allK[0])
        return tree

    split = len(allK) // 2
    tree.setChildren(makeBalancedTree(allK[:split]), makeBalancedTree(allK[split:]))
    return tree


class MCTree:
    """Multiclass classifier that routes examples through a class-partition tree."""

    def __init__(self, tree: TreeNode, mkClassifier: ClassifierFactory):
        self.tree = tree
        self.f = []
        for node in self.tree.iterNodes():
            if not node.isLeaf:
                node.setNodeInfo(mkClassifier())

    def train(self, X: np.ndarray, Y: np.ndarray) -> None:
        X = np.asarray(X)
        Y = np.asarray(Y)

        for node in self.tree.iterNodes():
            if node.isLeaf:
                continue

            left_labels = list(node.getLeft().iterAllLabels())
            right_labels = list(node.getRight().iterAllLabels())
            left_mask = np.isin(Y, left_labels)
            right_mask = np.isin(Y, right_labels)
            mask = left_mask | right_mask

            node_X = X[mask]
            node_Y = np.where(left_mask[mask], 1, -1)
            _fit_binary_classifier(node.getNodeInfo(), node_X, node_Y)

    def predict(self, X: np.ndarray) -> int:
        X = np.asarray(X).reshape(1, -1)
        node = self.tree

        while not node.isLeaf:
            probability = _positive_probability(node.getNodeInfo(), X)
            node = node.getLeft() if probability > 0.5 else node.getRight()

        return node.getLabel()

    def predictAll(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        predictions = np.zeros(X.shape[0], dtype=int)
        for index, row in enumerate(X):
            predictions[index] = self.predict(row)
        return predictions


def getMyTreeForWine() -> TreeNode:
    """Return the balanced 20-class tree used by the original wine example."""
    return makeBalancedTree(list(range(20)))
