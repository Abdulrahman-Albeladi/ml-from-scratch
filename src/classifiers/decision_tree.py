from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _TreeNode:
    is_leaf: bool
    prediction: int
    feature_index: Optional[int] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None


class DecisionTreeClassifier:
    """
    Binary decision tree classifier using greedy splits on thresholded features.

    This implementation is intentionally small and explicit. Each internal node
    selects one unused feature and routes examples left when feature < 0.5 and
    right otherwise. Split quality is measured by the total classification error
    obtained by predicting the majority class in each child.

    Parameters
    ----------
    max_depth:
        Maximum number of split levels. A value of 1 produces a decision stump.
    """

    def __init__(self, max_depth: int = 1) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        self.max_depth = int(max_depth)
        self._root: Optional[_TreeNode] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        if X.shape[0] == 0:
            raise ValueError("X and y must be non-empty")

        labels = np.unique(y)
        if not np.all(np.isin(labels, [-1, 1])):
            raise ValueError(
                "This classifier expects binary labels encoded as -1 and 1"
            )

        self.n_features_in_ = X.shape[1]
        self._root = self._build_tree(
            X, y, depth_remaining=self.max_depth, used_features=set()
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._root is None:
            raise ValueError("Model has not been fitted")

        X = np.asarray(X)
        if X.ndim == 1:
            if self.n_features_in_ is None or X.shape[0] != self.n_features_in_:
                raise ValueError(
                    "1D input must have length equal to the fitted feature count"
                )
            return np.asarray([self._predict_one(X, self._root)], dtype=int)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array")
        if self.n_features_in_ is None or X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted data"
            )

        return np.asarray([self._predict_one(row, self._root) for row in X], dtype=int)

    def __repr__(self) -> str:
        if self._root is None:
            return "DecisionTreeClassifier(unfitted)"
        return self._format_tree(self._root, depth=0)

    def _predict_one(self, x: np.ndarray, node: _TreeNode) -> int:
        if node.is_leaf:
            return node.prediction
        assert node.feature_index is not None  # nosec B101
        assert node.left is not None and node.right is not None  # nosec B101
        if x[node.feature_index] < 0.5:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth_remaining: int,
        used_features: set[int],
    ) -> _TreeNode:
        majority = self._majority_label(y)

        if depth_remaining <= 0 or np.unique(y).size <= 1:
            return _TreeNode(is_leaf=True, prediction=majority)

        n_samples, n_features = X.shape
        best_feature = None
        best_error = None

        for feature_index in range(n_features):
            if feature_index in used_features:
                continue

            left_mask = X[:, feature_index] < 0.5
            right_mask = ~left_mask
            left_y = y[left_mask]
            right_y = y[right_mask]

            left_error = self._partition_error(left_y)
            right_error = self._partition_error(right_y)
            total_error = left_error + right_error

            if best_error is None or total_error <= best_error:
                best_error = total_error
                best_feature = feature_index

        if best_feature is None:
            return _TreeNode(is_leaf=True, prediction=majority)

        left_mask = X[:, best_feature] < 0.5
        right_mask = ~left_mask

        next_used = set(used_features)
        next_used.add(best_feature)

        left_child = (
            self._build_tree(
                X[left_mask],
                y[left_mask],
                depth_remaining=depth_remaining - 1,
                used_features=next_used,
            )
            if np.any(left_mask)
            else _TreeNode(is_leaf=True, prediction=majority)
        )

        right_child = (
            self._build_tree(
                X[right_mask],
                y[right_mask],
                depth_remaining=depth_remaining - 1,
                used_features=next_used,
            )
            if np.any(right_mask)
            else _TreeNode(is_leaf=True, prediction=majority)
        )

        return _TreeNode(
            is_leaf=False,
            prediction=majority,
            feature_index=best_feature,
            left=left_child,
            right=right_child,
        )

    @staticmethod
    def _majority_label(y: np.ndarray) -> int:
        if y.size == 0:
            return 1
        positives = np.sum(y == 1)
        negatives = np.sum(y == -1)
        return 1 if positives >= negatives else -1

    def _partition_error(self, y: np.ndarray) -> int:
        if y.size == 0:
            return 0
        majority = self._majority_label(y)
        return int(np.sum(y != majority))

    def _format_tree(self, node: _TreeNode, depth: int) -> str:
        indent = " " * (depth * 2)
        if node.is_leaf:
            return f"{indent}Leaf {node.prediction}\n"
        assert node.feature_index is not None  # nosec B101
        assert node.left is not None and node.right is not None  # nosec B101
        return (
            f"{indent}Branch {node.feature_index}\n"
            f"{self._format_tree(node.left, depth + 1)}"
            f"{self._format_tree(node.right, depth + 1)}"
        )
