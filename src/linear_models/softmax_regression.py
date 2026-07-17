from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ArrayLike = np.ndarray


@dataclass
class SoftmaxRegression:
    """Multiclass linear classifier trained with softmax cross-entropy.

    This implementation is organized as an independent, clean-room reconstruction
    of the underlying algorithmic idea rather than a course-specific submission.

    Conventions
    -----------
    * Features are expected in shape ``(n_samples, n_features)``.
    * Labels are expected in shape ``(n_samples,)`` with integer class ids in
      ``[0, n_classes)``.
    * A bias term is included internally.
    """

    n_classes: int
    max_iter: int = 400
    fit_intercept: bool = True
    l2_penalty: float = 0.0
    random_state: int | None = None
    weights_: ArrayLike | None = field(default=None, init=False, repr=False)

    def _validate_inputs(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> tuple[ArrayLike, ArrayLike | None]:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        if y is None:
            return X, None

        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of integer class labels.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError("y must contain integer class labels.")
        if np.any(y < 0) or np.any(y >= self.n_classes):
            raise ValueError(
                "y contains labels outside the valid range [0, n_classes)."
            )

        return X, y

    def _add_intercept(self, X: ArrayLike) -> ArrayLike:
        if not self.fit_intercept:
            return X
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack((ones, X))

    def _parameter_shape(self, n_features: int) -> tuple[int, int]:
        return self.n_classes, n_features

    def _softmax(self, scores: ArrayLike) -> ArrayLike:
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _loss_and_gradient(
        self, flat_weights: ArrayLike, X: ArrayLike, y: ArrayLike
    ) -> tuple[float, ArrayLike]:
        n_samples, n_features = X.shape
        W = flat_weights.reshape(self._parameter_shape(n_features))

        scores = X @ W.T
        probabilities = self._softmax(scores)

        true_class_prob = probabilities[np.arange(n_samples), y]
        loss = -np.mean(np.log(true_class_prob + 1e-12))

        if self.l2_penalty > 0.0:
            penalty_weights = W[:, 1:] if self.fit_intercept else W
            loss += 0.5 * self.l2_penalty * np.sum(penalty_weights**2)

        error = probabilities.copy()
        error[np.arange(n_samples), y] -= 1.0
        gradient = (error.T @ X) / n_samples

        if self.l2_penalty > 0.0:
            reg = self.l2_penalty * W
            if self.fit_intercept:
                reg[:, 0] = 0.0
            gradient += reg

        return float(loss), gradient.ravel()

    def fit(self, X: ArrayLike, y: ArrayLike) -> "SoftmaxRegression":
        """Fit the model to X, y.

        SciPy is used for the optimizer if available. The import is performed
        lazily so that merely importing this module does not require SciPy to be
        installed (tests that only import modules will succeed).
        """
        X, y = self._validate_inputs(X, y)
        X_design = self._add_intercept(X)
        _, n_features = X_design.shape

        rng = np.random.default_rng(self.random_state)
        initial = rng.normal(loc=0.0, scale=1e-3, size=self.n_classes * n_features)

        try:
            # Import lazily to avoid an import-time dependency on SciPy.
            from scipy.optimize import minimize
        except Exception as exc:  # pragma: no cover - defensive import handling
            raise ImportError(
                "scipy is required to fit SoftmaxRegression. Install scipy to use fit()."
            ) from exc

        result = minimize(
            fun=self._loss_and_gradient,
            x0=initial,
            args=(X_design, y),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter},
        )

        self.weights_ = result.x.reshape(self._parameter_shape(n_features))
        return self

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        if self.weights_ is None:
            raise ValueError("Model is not fitted. Call fit before predict_proba.")

        X, _ = self._validate_inputs(X)
        X_design = self._add_intercept(X)
        scores = X_design @ self.weights_.T
        return self._softmax(scores)

    def predict(self, X: ArrayLike) -> ArrayLike:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
