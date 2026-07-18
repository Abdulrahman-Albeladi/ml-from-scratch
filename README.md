# ML From Scratch

A collection of educational machine-learning implementations recovered from course and research materials and organized as a Python repository. The repository contains maintained source modules under `src/` and preserved recovered project artifacts under `projects/`.

The implementations cover baseline classifiers, decision trees, nearest-neighbor classification, linear classifiers, gradient-based optimization, multiclass methods, principal component analysis, softmax regression, and a multilayer perceptron.

## Repository layout

- `src/` — consolidated implementation modules.
  - `classifiers/` — baselines, decision tree, k-nearest neighbors, and multiclass utilities.
  - `linear_models/` — perceptron, linear classifiers, and softmax regression.
  - `optimization/` — gradient-descent routines.
  - `decomposition/` — principal component analysis.
  - `neural_nets/` — multilayer perceptron code.
- `tests/test_imports.py` — import-level test coverage.
- `projects/` — recovered project directories retained for provenance.
- `docs/design-decisions.md` — repository design notes.
- `pyproject.toml` — Python project and tool configuration.
- `.github/workflows/ci.yml` — continuous-integration workflow configuration.

## Included implementations

| Area | Consolidated module | Recovered source material |
| --- | --- | --- |
| Baseline classifiers | `src/classifiers/baselines.py` | `projects/binary-classifiers/dumbClassifiers.py` |
| Decision tree | `src/classifiers/decision_tree.py` | `projects/binary-classifiers/dt.py` |
| k-nearest neighbors | `src/classifiers/knn.py` | `projects/binary-classifiers/knn.py` |
| Perceptron | `src/linear_models/perceptron.py` | `projects/binary-classifiers/perceptron.py` |
| Gradient descent | `src/optimization/gradient_descent.py` | `projects/linear-and-multiclass-models/gd.py` |
| Linear classifiers | `src/linear_models/linear_classifiers.py` | `projects/linear-and-multiclass-models/linear.py` |
| Multiclass classification | `src/classifiers/multiclass.py` | `projects/linear-and-multiclass-models/multiclass.py` |
| PCA / dimensionality reduction | `src/decomposition/pca.py` | `projects/pca-and-neural-networks/dr.py` |
| Multilayer perceptron | `src/neural_nets/mlp.py` | `projects/pca-and-neural-networks/nn.py` |
| Softmax regression | `src/linear_models/softmax_regression.py` | `projects/pca-and-neural-networks/softmax.py` |

## Setup

Use the project configuration in `pyproject.toml` to create an environment and install dependencies. The exact dependency set and supported Python versions should be taken from that file.

A typical local workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If the project defines optional development dependencies or tool environments, install them according to `pyproject.toml` before running linting or tests.

## Data and privacy

No dataset files are listed in this repository snapshot. The recovered source files appear to be implementation-focused rather than packaged experiments.

Use only data that may be lawfully processed and shared. Do not commit private course data, research data, credentials, participant information, institutional paths, or generated outputs containing sensitive records. For local experiments, provide data through local configuration or a separate ignored directory rather than embedding private paths in source code.

## Limitations

- These are from-scratch educational implementations, not replacements for mature production libraries.
- Public APIs, numerical behavior, input validation, performance characteristics, and supported edge cases should be verified against the source before use.
- The available test file is import-oriented; algorithmic correctness, numerical stability, shape handling, reproducibility, and performance are not established by the file list alone.
- Recovered PDFs and metadata remain in the provenance directories and may contain historical context that is not part of the maintained API.

## Provenance

The repository consolidates publish-eligible material recovered from three historical project directories:

- `projects/binary-classifiers/`
- `projects/linear-and-multiclass-models/`
- `projects/pca-and-neural-networks/`

These directories retain original filenames, metadata files, partner files where present, and writeups where present. The modules under `src/` provide the repository-facing organization of the recovered implementation topics. Historical artifacts are retained for attribution and traceability; they should not be interpreted as current documentation, test evidence, or a guarantee that every recovered artifact is suitable for redistribution outside its original context.

## Contributing and security

See `CONTRIBUTING.md` for contribution guidance and `SECURITY.md` for security reporting information. Review `LICENSE_REVIEW.md` before redistributing recovered materials or adding third-party content.

<!-- internal-projects:start -->
## Projects

| Project | Location |
|---|---|
| Binary Classifiers | `projects/binary-classifiers` |
| Linear and Multiclass Models | `projects/linear-and-multiclass-models` |
| PCA and Neural Networks | `projects/pca-and-neural-networks` |
<!-- internal-projects:end -->
