from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Protocol, Sequence

import numpy as np


class BinaryProbabilisticClassifier(Protocol):
    """Protocol for binary classifiers used by multiclass reductions.

    Implementations are expected to train on labels encoded as {-1, +1}
    and expose class-1 probabilities through ``predict_proba``.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


ClassifierFactory = Callable[[], BinaryProbabilisticClassifier]


class _TrainAdapter:
    """Wraps estimators that use ``train`` instead of ``fit``."""

    def __init__(self, estimator: object) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        train = getattr(self.estimator, "train", None)
        if train is None:
            raise TypeError("Binary classifier must define fit(X, y) or train(X, y).")
        train(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predict_proba = getattr(self.estimator, "predict_proba", None)
        if predict_proba is None:
            raise TypeError("Binary classifier must define predict_proba(X).")
        return predict_proba(X)


def _coerce_binary_classifier(estimator: object) -> BinaryProbabilisticClassifier:
    if hasattr(estimator, "fit") and hasattr(estimator, "predict_proba"):
        return estimator  # type: ignore[return-value]
    return _TrainAdapter(estimator)


def _positive_class_probability(
    classifier: BinaryProbabilisticClassifier, x: np.ndarray
) -> float:
    probs = classifier.predict_proba(x.reshape(1, -1))
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[0] != 1 or probs.shape[1] < 2:
        raise ValueError(
            "predict_proba must return shape (1, 2) or compatible binary probabilities."
        )
    return float(probs[0, 1])


class OneVsRestClassifier:
    """Multiclass reduction using one binary classifier per class."""

    def __init__(self, n_classes: int, make_classifier: ClassifierFactory) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        self.n_classes = n_classes
        self.classifiers = [
            _coerce_binary_classifier(make_classifier()) for _ in range(n_classes)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        for class_index, classifier in enumerate(self.classifiers):
            binary_targets = np.where(y == class_index, 1, -1)
            classifier.fit(X, binary_targets)

    def predict_one(self, x: np.ndarray, hard_voting: bool = False) -> int:
        scores = np.zeros(self.n_classes, dtype=float)
        for class_index, classifier in enumerate(self.classifiers):
            p = _positive_class_probability(classifier, np.asarray(x))
            scores[class_index] = 1.0 if hard_voting and p > 0.5 else p
        return int(np.argmax(scores))

    def predict(self, X: np.ndarray, hard_voting: bool = False) -> np.ndarray:
        X = np.asarray(X)
        return np.array(
            [self.predict_one(row, hard_voting=hard_voting) for row in X], dtype=int
        )


class OneVsOneClassifier:
    """Multiclass reduction using one binary classifier per class pair."""

    def __init__(self, n_classes: int, make_classifier: ClassifierFactory) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2.")
        self.n_classes = n_classes
        self.classifiers: List[List[BinaryProbabilisticClassifier]] = []
        for i in range(n_classes):
            row: List[BinaryProbabilisticClassifier] = []
            for _ in range(i):
                row.append(_coerce_binary_classifier(make_classifier()))
            self.classifiers.append(row)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        for i in range(self.n_classes):
            for j in range(i):
                mask = (y == i) | (y == j)
                X_ij = X[mask]
                y_ij = np.where(y[mask] == j, 1, -1)
                self.classifiers[i][j].fit(X_ij, y_ij)

    def predict_one(self, x: np.ndarray, hard_voting: bool = False) -> int:
        scores = np.zeros(self.n_classes, dtype=float)
        x = np.asarray(x)
        for i in range(self.n_classes):
            for j in range(i):
                p = _positive_class_probability(self.classifiers[i][j], x)
                if hard_voting:
                    vote = 1.0 if p > 0.5 else 0.0
                    scores[j] += vote
                    scores[i] -= vote
                else:
                    scores[j] += p
                    scores[i] -= p
        return int(np.argmax(scores))

    def predict(self, X: np.ndarray, hard_voting: bool = False) -> np.ndarray:
        X = np.asarray(X)
        return np.array(
            [self.predict_one(row, hard_voting=hard_voting) for row in X], dtype=int
        )


@dataclass
class ClassTreeNode:
    """Binary tree over class labels for hierarchical multiclass reduction."""

    label: Optional[int] = None
    left: Optional["ClassTreeNode"] = None
    right: Optional["ClassTreeNode"] = None
    classifier: Optional[BinaryProbabilisticClassifier] = None

    @property
    def is_leaf(self) -> bool:
        return self.label is not None

    def labels(self) -> Iterable[int]:
        if self.is_leaf:
            yield int(self.label)
            return
        if self.left is None or self.right is None:
            raise ValueError(
                "Internal tree nodes must define both left and right children."
            )
        yield from self.left.labels()
        yield from self.right.labels()

    def nodes(self) -> Iterable["ClassTreeNode"]:
        yield self
        if self.left is not None:
            yield from self.left.nodes()
        if self.right is not None:
            yield from self.right.nodes()


def make_balanced_class_tree(class_labels: Sequence[int]) -> ClassTreeNode:
    labels = list(class_labels)
    if not labels:
        raise ValueError("class_labels must contain at least one label.")
    if len(labels) == 1:
        return ClassTreeNode(label=int(labels[0]))
    split = len(labels) // 2
    return ClassTreeNode(
        left=make_balanced_class_tree(labels[:split]),
        right=make_balanced_class_tree(labels[split:]),
    )


class HierarchicalMulticlassClassifier:
    """Hierarchical multiclass reduction over a user-provided class tree."""

    def __init__(self, tree: ClassTreeNode, make_classifier: ClassifierFactory) -> None:
        self.tree = tree
        for node in self.tree.nodes():
            if not node.is_leaf:
                node.classifier = _coerce_binary_classifier(make_classifier())

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y)
        for node in self.tree.nodes():
            if node.is_leaf:
                continue
            if node.left is None or node.right is None or node.classifier is None:
                raise ValueError(
                    "Internal tree nodes must have children and an assigned classifier."
                )

            left_labels = np.array(list(node.left.labels()), dtype=int)
            right_labels = np.array(list(node.right.labels()), dtype=int)
            left_mask = np.isin(y, left_labels)
            right_mask = np.isin(y, right_labels)
            mask = left_mask | right_mask
            X_node = X[mask]
            y_node = np.where(left_mask[mask], 1, -1)
            node.classifier.fit(X_node, y_node)

    def predict_one(self, x: np.ndarray) -> int:
        node = self.tree
        x = np.asarray(x)
        while not node.is_leaf:
            if node.left is None or node.right is None or node.classifier is None:
                raise ValueError(
                    "Internal tree nodes must have children and an assigned classifier."
                )
            p = _positive_class_probability(node.classifier, x)
            node = node.left if p > 0.5 else node.right
        if node.label is None:
            raise ValueError("Leaf nodes must define a class label.")
        return int(node.label)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.array([self.predict_one(row) for row in X], dtype=int)


__all__ = [
    "BinaryProbabilisticClassifier",
    "OneVsRestClassifier",
    "OneVsOneClassifier",
    "ClassTreeNode",
    "make_balanced_class_tree",
    "HierarchicalMulticlassClassifier",
]
