"""Basic gradient descent optimizer.

This module provides a small, independent implementation of gradient descent
with a diminishing learning-rate schedule. It is intended for educational and
small-scale experimental use in from-scratch machine learning code.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ArrayLike = np.ndarray
ObjectiveFn = Callable[[ArrayLike], float]
GradientFn = Callable[[ArrayLike], ArrayLike]
StepScheduleFn = Callable[[int, float], float]


def inverse_sqrt_decay(iteration: int, initial_step_size: float) -> float:
    """Return ``initial_step_size / sqrt(iteration + 1)``.

    Parameters
    ----------
    iteration:
        Zero-based iteration index.
    initial_step_size:
        Positive base learning rate.
    """
    if initial_step_size <= 0:
        raise ValueError("initial_step_size must be positive")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    return initial_step_size / np.sqrt(iteration + 1.0)


def gradient_descent(
    objective: ObjectiveFn,
    gradient: GradientFn,
    x0: ArrayLike,
    num_iterations: int,
    initial_step_size: float,
    step_schedule: StepScheduleFn | None = None,
) -> tuple[ArrayLike, np.ndarray]:
    """Minimize an objective with gradient descent.

    Parameters
    ----------
    objective:
        Function mapping a parameter vector to a scalar objective value.
    gradient:
        Function returning the gradient of ``objective`` at a parameter vector.
    x0:
        Initial parameter vector.
    num_iterations:
        Number of update steps to run.
    initial_step_size:
        Base learning rate used by the step schedule.
    step_schedule:
        Optional callable ``f(iteration, initial_step_size) -> step_size``.
        Defaults to inverse square-root decay.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Final parameter vector and objective-value history. The history has
        length ``num_iterations + 1`` and includes the initial objective value.
    """
    if num_iterations < 0:
        raise ValueError("num_iterations must be non-negative")
    if initial_step_size <= 0:
        raise ValueError("initial_step_size must be positive")

    schedule = step_schedule or inverse_sqrt_decay
    x = np.array(x0, dtype=float, copy=True)

    trajectory = np.empty(num_iterations + 1, dtype=float)
    trajectory[0] = float(objective(x))

    for iteration in range(num_iterations):
        grad_value = np.asarray(gradient(x), dtype=float)
        if grad_value.shape != x.shape:
            raise ValueError(
                "gradient(x) must return the same shape as x; "
                f"got gradient shape {grad_value.shape} and x shape {x.shape}"
            )

        step_size = float(schedule(iteration, initial_step_size))
        x = x - step_size * grad_value
        trajectory[iteration + 1] = float(objective(x))

    return x, trajectory


__all__ = ["gradient_descent", "inverse_sqrt_decay"]
