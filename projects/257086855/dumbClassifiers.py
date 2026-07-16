"""Baseline binary classifiers.

The classifiers in this module provide simple reference models for binary
classification experiments:

* predict ``+1`` for every example;
* predict the most frequent training label; and
* predict according to whether the first feature is positive.
"""

from binary import BinaryClassifier


class AlwaysPredictOne(BinaryClassifier):
    """Classifier that always predicts ``+1``."""

    def __init__(self, opts):
        pass

    def online(self):
        return False

    def __repr__(self):
        return "AlwaysPredictOne"

    def predict(self, X):
        return 1

    def train(self, X, Y):
        pass


class AlwaysPredictMostFrequent(BinaryClassifier):
    """Classifier that predicts the most common observed training label."""

    def __init__(self, opts):
        self.mostFrequentClass = 1

    def online(self):
        return False

    def __repr__(self):
        return "AlwaysPredictMostFrequent(%d)" % self.mostFrequentClass

    def predict(self, X):
        return self.mostFrequentClass

    def train(self, X, Y):
        """Store the most frequent label in ``Y``.

        An empty label sequence leaves the default prediction unchanged.
        """
        frequencies = {}
        for label in Y:
            frequencies[label] = frequencies.get(label, 0) + 1

        if frequencies:
            self.mostFrequentClass = max(frequencies, key=frequencies.get)


class FirstFeatureClassifier(BinaryClassifier):
    """Classifier that selects a label from the sign of the first feature."""

    def __init__(self, opts):
        self.classForPos = 1
        self.classForNeg = 1

    def online(self):
        return False

    def __repr__(self):
        return "FirstFeatureClassifier(%d,%d)" % (
            self.classForPos,
            self.classForNeg,
        )

    def predict(self, X):
        if X[0] > 0:
            return self.classForPos
        return self.classForNeg

    def train(self, X, Y):
        """Learn the most frequent label for each first-feature sign."""
        frequencies = {False: {}, True: {}}

        for features, label in zip(X, Y):
            is_positive = features[0] > 0
            counts = frequencies[is_positive]
            counts[label] = counts.get(label, 0) + 1

        if frequencies[False]:
            self.classForNeg = max(frequencies[False], key=frequencies[False].get)
        if frequencies[True]:
            self.classForPos = max(frequencies[True], key=frequencies[True].get)
