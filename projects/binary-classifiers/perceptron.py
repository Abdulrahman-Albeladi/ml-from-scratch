"""Online binary perceptron classifier."""

from numpy import dot

from binary import BinaryClassifier


class Perceptron(BinaryClassifier):
    """Binary classifier trained with standard online perceptron updates."""

    def __init__(self, opts):
        BinaryClassifier.__init__(self, opts)
        self.opts = opts
        self.reset()

    def reset(self):
        """Reset the model parameters and update counter."""
        self.weights = 0
        self.bias = 0
        self.numUpd = 0

    def online(self):
        """Indicate that this classifier updates one example at a time."""
        return True

    def __repr__(self):
        return "w=" + repr(self.weights) + ", b=" + repr(self.bias)

    def predict(self, X):
        """Return the signed margin for feature vector ``X``."""
        if self.numUpd == 0:
            return 0
        return dot(self.weights, X) + self.bias

    def nextExample(self, X, Y):
        """Update the model when example ``X`` is misclassified or on-margin."""
        if Y * self.predict(X) <= 0:
            self.numUpd += 1
            self.weights = self.weights + X * Y
            self.bias = self.bias + Y

    def nextIteration(self):
        """Mark the end of a training pass."""
        return

    def getRepresentation(self):
        """Return ``(number_of_updates, weights, bias)``."""
        return (self.numUpd, self.weights, self.bias)
