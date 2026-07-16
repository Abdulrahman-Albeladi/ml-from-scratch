"""Simple baseline classifiers for binary labels.

These models are intentionally minimal. They are useful as sanity checks
when comparing more expressive classifiers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


class BaselineClassifier:
    """Small common interface for baseline classifiers."""

    def fit(
        self, features: Sequence[Sequence[float]], labels: Sequence[Any]
    ) -> "BaselineClassifier":
        return self

    def predict_one(self, features: Sequence[float]) -> Any:
        raise NotImplementedError

    def predict(self, features: Sequence[Sequence[float]]) -> list[Any]:
        return [self.predict_one(row) for row in features]


@dataclass
class ConstantClassifier(BaselineClassifier):
    """Predict the same label for every example."""

    label: Any = 1

    def predict_one(self, features: Sequence[float]) -> Any:
        return self.label


@dataclass
class MajorityClassClassifier(BaselineClassifier):
    """Predict the most frequent label observed during fitting.

    If ``fit`` has not been called, the classifier uses ``default_label``.
    Ties are resolved by first occurrence in the input label sequence.
    """

    default_label: Any = 1
    majority_label: Any = field(default=1)

    def __post_init__(self) -> None:
        self.majority_label = self.default_label

    def fit(
        self, features: Sequence[Sequence[float]], labels: Sequence[Any]
    ) -> "MajorityClassClassifier":
        del features
        if not labels:
            self.majority_label = self.default_label
            return self

        counts = Counter(labels)
        best_label = labels[0]
        best_count = counts[best_label]
        for label in labels:
            count = counts[label]
            if count > best_count:
                best_label = label
                best_count = count

        self.majority_label = best_label
        return self

    def predict_one(self, features: Sequence[float]) -> Any:
        del features
        return self.majority_label


@dataclass
class FirstFeatureSignClassifier(BaselineClassifier):
    """Split on the sign of the first feature and predict per side.

    The model learns one label for examples with first feature > 0 and one
    label for examples with first feature <= 0.
    """

    default_positive_label: Any = 1
    default_nonpositive_label: Any = 1
    positive_label: Any = field(default=1)
    nonpositive_label: Any = field(default=1)

    def __post_init__(self) -> None:
        self.positive_label = self.default_positive_label
        self.nonpositive_label = self.default_nonpositive_label

    def fit(
        self, features: Sequence[Sequence[float]], labels: Sequence[Any]
    ) -> "FirstFeatureSignClassifier":
        if len(features) != len(labels):
            raise ValueError("features and labels must have the same length")

        positive_labels: list[Any] = []
        nonpositive_labels: list[Any] = []

        for row, label in zip(features, labels):
            if len(row) == 0:
                raise ValueError("each feature vector must contain at least one value")
            if row[0] > 0:
                positive_labels.append(label)
            else:
                nonpositive_labels.append(label)

        if positive_labels:
            self.positive_label = _majority_label(positive_labels)
        else:
            self.positive_label = self.default_positive_label

        if nonpositive_labels:
            self.nonpositive_label = _majority_label(nonpositive_labels)
        else:
            self.nonpositive_label = self.default_nonpositive_label

        return self

    def predict_one(self, features: Sequence[float]) -> Any:
        if len(features) == 0:
            raise ValueError("feature vector must contain at least one value")
        return self.positive_label if features[0] > 0 else self.nonpositive_label


def _majority_label(labels: Iterable[Any]) -> Any:
    ordered_labels = list(labels)
    counts = Counter(ordered_labels)
    best_label = ordered_labels[0]
    best_count = counts[best_label]
    for label in ordered_labels:
        count = counts[label]
        if count > best_count:
            best_label = label
            best_count = count
    return best_label
