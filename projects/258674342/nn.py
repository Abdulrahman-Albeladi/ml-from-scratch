import numpy as np


class NN:
    """Fully connected neural network for column-major batches.

    Inputs have shape ``(input_d, batch_size)`` and targets have shape
    ``(output_d, batch_size)``. The final layer is linear; hidden layers use
    the supplied activation function.
    """

    def __init__(
        self,
        activation_function,
        loss_function,
        hidden_layers=None,
        input_d=784,
        output_d=10,
    ):
        self.weights = []
        self.biases = []
        self.activation_function = activation_function
        self.loss_function = loss_function

        if hidden_layers is None:
            hidden_layers = [1024]
        layer_dimensions = list(hidden_layers) + [output_d]

        input_dimension = input_d
        for output_dimension in layer_dimensions:
            self.weights.append(
                np.random.randn(output_dimension, input_dimension)
                * np.sqrt(2.0 / input_dimension)
            )
            self.biases.append(np.zeros((output_dimension, 1)))
            input_dimension = output_dimension

    def print_model(self):
        """Print the activation, loss, and parameter shapes for each layer."""
        print("activation:{}".format(self.activation_function.__class__.__name__))
        print("loss function:{}".format(self.loss_function.__class__.__name__))
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases), 1):
            print("Layer {}\tw:{}\tb:{}".format(index, weights.shape, biases.shape))

    def predict(self, X):
        """Return the highest-scoring output class for each example in ``X``."""
        activations = X
        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            activations = self.activation_function.activate(
                weights @ activations + biases
            )

        scores = self.weights[-1] @ activations + self.biases[-1]
        return np.argmax(scores, axis=0)

    def compute_gradients(self, X, Y):
        """Compute loss and parameter gradients for a batch without updating it."""
        activations = [X]
        pre_activations = []

        current = X
        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            pre_activation = weights @ current + biases
            pre_activations.append(pre_activation)
            current = self.activation_function.activate(pre_activation)
            activations.append(current)

        predictions = self.weights[-1] @ current + self.biases[-1]
        training_loss = self.loss_function.loss(Y, predictions)

        gradients_w = []
        gradients_b = []

        gradient = self.loss_function.lossGradient(Y, predictions)
        gradients_b.append(np.sum(gradient, axis=1, keepdims=True))
        gradients_w.append(gradient @ activations[-1].T)

        for index in range(len(self.weights) - 2, -1, -1):
            gradient = (
                self.weights[index + 1].T @ gradient
            ) * self.activation_function.backprop_grad(pre_activations[index])
            gradients_b.append(np.sum(gradient, axis=1, keepdims=True))
            gradients_w.append(gradient @ activations[index].T)

        return training_loss, gradients_w[::-1], gradients_b[::-1]

    def update(self, grad_Ws, grad_bs, learning_rate):
        """Apply a gradient-descent update to all network parameters."""
        for index, (grad_w, grad_b) in enumerate(zip(grad_Ws, grad_bs)):
            self.weights[index] -= learning_rate * grad_w
            self.biases[index] -= learning_rate * grad_b


class activationFunction:
    """Base interface for hidden-layer activation functions."""

    def activate(self, X):
        """Return activations with the same shape as ``X``."""
        raise NotImplementedError("Abstract class.")

    def backprop_grad(self, X):
        """Return activation derivatives with the same shape as ``X``."""
        raise NotImplementedError("Abstract class.")


class Relu(activationFunction):
    """Rectified linear unit activation."""

    def activate(self, X):
        return X * (X > 0)

    def backprop_grad(self, X):
        return (X > 0).astype(np.float64)


class Linear(activationFunction):
    """Identity activation."""

    def activate(self, X):
        return X

    def backprop_grad(self, X):
        return np.ones(X.shape, dtype=np.float64)


class LossFunction:
    """Base interface for losses evaluated over column-major batches."""

    def loss(self, Y, Yhat):
        """Return the scalar loss for targets ``Y`` and predictions ``Yhat``."""
        raise NotImplementedError("Abstract class.")

    def lossGradient(self, Y, Yhat):
        """Return the gradient of the loss with respect to ``Yhat``."""
        raise NotImplementedError("Abstract class.")


class SquaredLoss(LossFunction):
    """Mean half-squared-error loss over a batch."""

    def loss(self, Y, Yhat):
        batch_size = Y.shape[1]
        return np.linalg.norm(Yhat - Y) ** 2 / (2 * batch_size)

    def lossGradient(self, Y, Yhat):
        batch_size = Y.shape[1]
        return (Yhat - Y) / batch_size


class CELoss(LossFunction):
    """Placeholder for a cross-entropy loss implementation."""

    def loss(self, Y, Yhat):
        raise NotImplementedError("CELoss is not implemented.")

    def lossGradient(self, Y, Yhat):
        raise NotImplementedError("CELoss is not implemented.")
