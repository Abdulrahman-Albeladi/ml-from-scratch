# ml-from-scratch

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
| Baseline classifiers | `src/classifiers/baselines.py` | `projects/257086855/dumbClassifiers.py` |
| Decision tree | `src/classifiers/decision_tree.py` | `projects/257086855/dt.py` |
| k-nearest neighbors | `src/classifiers/knn.py` | `projects/257086855/knn.py` |
| Perceptron | `src/linear_models/perceptron.py` | `projects/257086855/perceptron.py` |
| Gradient descent | `src/optimization/gradient_descent.py` | `projects/258095197/gd.py` |
| Linear classifiers | `src/linear_models/linear_classifiers.py` | `projects/258095197/linear.py` |
| Multiclass classification | `src/classifiers/multiclass.py` | `projects/258095197/multiclass.py` |
| PCA / dimensionality reduction | `src/decomposition/pca.py` | `projects/258674342/dr.py` |
| Multilayer perceptron | `src/neural_nets/mlp.py` | `projects/258674342/nn.py` |
| Softmax regression | `src/linear_models/softmax_regression.py` | `projects/258674342/softmax.py` |

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

## Validation status

The repository includes `tests/test_imports.py`, which indicates import-level validation is intended. No supplied test output, CI result, coverage report, benchmark result, or training evaluation result is available here. This README therefore does not claim that tests, linting, or workflows currently pass.

Recommended validation commands are listed in the project index and should be run in a configured local environment.

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

- `projects/257086855/`
- `projects/258095197/`
- `projects/258674342/`

These directories retain original filenames, metadata files, partner files where present, and writeups where present. The modules under `src/` provide the repository-facing organization of the recovered implementation topics. Historical artifacts are retained for attribution and traceability; they should not be interpreted as current documentation, test evidence, or a guarantee that every recovered artifact is suitable for redistribution outside its original context.

## Contributing and security

See `CONTRIBUTING.md` for contribution guidance and `SECURITY.md` for security reporting information. Review `LICENSE_REVIEW.md` before redistributing recovered materials or adding third-party content.

## Current repository layout

- `.github/` — 1 files
- `docs/` — 1 files
- `projects/` — 13 files
- `src/` — 10 files
- `tests/` — 1 files

## Public-release status

**READY FOR FINAL MANUAL PUBLIC-RELEASE CHECK**

Automated security and documentation checks pass. Complete the ownership checklist and verify build or test claims before changing visibility.

This repository uses an all-rights-reserved portfolio license. Review `LICENSE`,
`LICENSE_REVIEW.md`, `THIRD_PARTY_NOTICES.md`, and `OWNERSHIP_REVIEW.md`.
