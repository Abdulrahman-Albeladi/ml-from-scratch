# ML From Scratch


Machine-learning algorithms implemented from first principles in Python/NumPy with a clean, reusable package layout.

Technologies: Python · NumPy · SciPy · pytest

## Overview

This repository is a compact, from-scratch implementation set organized as a conventional Python package under `src/`. It focuses on translating core ML math into readable, testable NumPy code with small, composable APIs.

What’s included (selected):
- Linear models: Perceptron; L2-regularized linear model with Hinge/Logistic/Squared losses; Softmax regression
- Classifiers: k-nearest neighbors (KNN) and radius-neighbors; a compact decision tree; multiclass reductions (one-vs-rest, one-vs-one, hierarchical)
- Decomposition: PCA (Principal Component Analysis)
- Neural nets: A tiny fully connected MLP with ReLU and MSE loss
- Optimization: Gradient descent with an inverse-square-root schedule

Explore the source:
- `src/classifiers/` — baselines, KNN, decision tree, multiclass
- `src/linear_models/` — perceptron, linear classifiers, softmax regression
- `src/decomposition/` — PCA
- `src/neural_nets/` — small MLP
- `src/optimization/` — gradient descent

## Installation

- Python 3.10+
- From the repository root:

  ```bash
  python -m pip install -e ".[dev]"
  ```

## Quick start examples

Perceptron (binary labels in {-1, +1}):

```python
import numpy as np
from linear_models.perceptron import Perceptron

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = np.array([-1, -1, -1, 1])  # linearly separable example

model = Perceptron(averaged=True, max_epochs=20, random_state=0)
model.fit(X, y)
preds = model.predict([[1.0, 1.0], [0.0, 1.0]])
```

PCA (reduce to 2D):

```python
import numpy as np
from decomposition.pca import PCA

X = np.array([
    [2.5,  2.4,  0.7],
    [0.5,  0.7, -1.2],
    [2.2,  2.9,  1.1],
    [1.9,  2.2,  0.0],
])

pca = PCA(n_components=2)
X_proj = pca.fit_transform(X)
```

Decision tree (binary labels in {-1, +1}):

```python
import numpy as np
from classifiers.decision_tree import DecisionTreeClassifier

X = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
])
y = np.array([1, -1, -1, 1])

clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)
preds = clf.predict(X)
```

These minimal snippets illustrate usage only; they are not benchmarks.

## Testing

Run the import surface tests:

```bash
pytest
```

## Design choices

- From-scratch NumPy implementations with explicit shapes and small APIs
- Clear separation of concerns (optimization vs. models vs. utilities)
- Typed where helpful; dataclasses for simple model state
- Minimal, focused tests; extend as needed for behavior/metrics

## License and attribution

Use and redistribution are governed by the repository’s [LICENSE](LICENSE).

Historical, assignment-shaped materials were moved to a private archive to keep this public repo employer-focused. Any third-party adapted code remains private until licensing/attribution is fully verified.
