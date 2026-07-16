"""Gradient descent with a square-root learning-rate schedule."""

import numpy as np


def gd(func, grad, x0, numIter, stepSize):
    """Minimize ``func`` using gradient descent.

    Args:
        func: Callable that evaluates the objective at a parameter vector.
        grad: Callable that evaluates the objective gradient at a parameter
            vector.
        x0: Initial parameter vector.
        numIter: Number of gradient-descent updates to perform.
        stepSize: Initial learning-rate scale. The learning rate at iteration
            ``t`` is ``stepSize / sqrt(t + 1)``.

    Returns:
        A tuple ``(x, trajectory)``, where ``x`` is the final parameter vector
        and ``trajectory`` contains the objective value before optimization and
        after each update.
    """
    x = x0
    trajectory = np.zeros(numIter + 1)
    trajectory[0] = func(x)

    for iteration in range(numIter):
        learning_rate = stepSize / np.sqrt(iteration + 1)
        x = x - learning_rate * grad(x)
        trajectory[iteration + 1] = func(x)

    return x, trajectory
