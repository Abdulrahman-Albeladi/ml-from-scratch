# Project index

## Current consolidated modules

| Module | Scope | Validation evidence present | Data requirements | Known limitations | Provenance |
| --- | --- | --- | --- | --- | --- |
| `src/classifiers/baselines.py` | Baseline classifier implementations. | Repository contains import-oriented test coverage only; no result is provided. | No data files are included. Supply compatible local inputs. | Behavior, supported labels, and edge-case handling require source review. | Corresponds by topic and filename to `projects/257086855/dumbClassifiers.py`. |
| `src/classifiers/decision_tree.py` | Decision-tree classification implementation. | Repository contains import-oriented test coverage only; no result is provided. | No data files are included. Supply compatible local feature and target data. | Split criteria, stopping behavior, and handling of categorical or missing values are not established by the file list. | Corresponds by topic and filename to `projects/257086855/dt.py`. |
| `src/classifiers/knn.py` | k-nearest-neighbor classification implementation. | Repository contains import-oriented test coverage only; no result is provided. | No data files are included. Supply compatible local training and query data. | Distance metric, tie handling, scaling assumptions, and computational complexity require source review. | Corresponds by topic and filename to `projects/257086855/knn.py`. |
| `src/classifiers/multiclass.py` | Multiclass classification utilities or strategies. | Repository contains import-oriented test coverage only; no result is provided. | No data files are included. Supply compatible local labeled data. | The multiclass reduction or prediction strategy must be confirmed from source. | Corresponds by topic and filename to `projects/258095197/multiclass.py`. |
| `src/linear_models/perceptron.py` | Perceptron implementation. | Repository contains import-oriented test coverage only; no result is provided. | No data files are included. Supply compatible local labeled feature data. | Convergence behavior, label encoding, intercept treatment, and iteration controls require source review. | Corresponds by topic and filename to `projects/257086855/perceptron.py`. |
| `src/linear_models/linear_classifiers.py` | Linear-classifier implementation(s). | Repository contains import-oriented test coverage only; no result is provided. | No data files are included. Supply compatible local labeled feature data. | Loss function, regularization, optimization settings, and output conventions require source review. | Corresponds by topic and filename to `projects/258095197/linear.py`. |
| `src/linear_models/softmax_regression.py` | Softmax-regression implementation. | Repository contains import-oriented test coverage only; no result is provided. | No data files are included. Supply compatible multiclass labeled data. | Numerical stabilization, class encoding, regularization, and convergence behavior require source review. | Corresponds by topic and filename to `projects/258674342/softmax.py`. |
| `src/optimization/gradient_descent.py` | Gradient-descent routines. | Repository contains import-oriented test coverage only; no result is provided. | No data files are included; callers provide objectives and/or numerical inputs as supported by the code. | Step-size policy, stopping criteria, and supported objective interfaces require source review. | Corresponds by topic and filename to `projects/258095197/gd.py`. |
| `src/decomposition/pca.py` | Principal component analysis or dimensionality-reduction implementation. | Repository contains import-oriented test coverage only; no result is provided. | No data files are included. Supply compatible numeric feature matrices. | Centering, component selection, decomposition method, and rank-deficient behavior require source review. | Corresponds by topic and filename to `projects/258674342/dr.py`. |
| `src/neural_nets/mlp.py` | Multilayer perceptron implementation. | Repository contains import-oriented test coverage only; no result is provided. | No data files are included. Supply compatible local training data. | Architecture, activations, initialization, optimization, batching, and reproducibility controls require source review. | Corresponds by topic and filename to `projects/258674342/nn.py`. |

## Recovered project directories

### `projects/257086855/` — classifier implementations

**Contents**

- `dt.py` — decision-tree implementation source.
- `dumbClassifiers.py` — baseline classifier source.
- `knn.py` — k-nearest-neighbor source.
- `perceptron.py` — perceptron source.
- `metadata.yml` — recovered project metadata.
- `partners.txt` — historical collaboration record.
- `writeup.pdf` — recovered project writeup.

**Private-data requirements**

No dataset is listed in this directory. If the writeup or source references course-provided, private, or otherwise restricted data, that data is not included and should not be reconstructed or committed. Use synthetic data or separately configured public data for demonstrations.

**Validation status**

No project-specific test directory or test result is listed. The consolidated repository has `tests/test_imports.py`, but no execution result is supplied.

**Limitations**

The recovered source and writeup may reflect an original course context. Treat the maintained `src/` modules as the repository entry points, and consult the recovered files only for provenance and implementation history.

**Provenance**

This directory is retained as recovered historical material. `partners.txt` and `writeup.pdf` are preserved rather than rewritten, so collaboration and submission-era context remain distinct from current repository documentation.

### `projects/258095197/` — optimization and linear classification

**Contents**

- `gd.py` — gradient-descent source.
- `linear.py` — linear-classifier source.
- `multiclass.py` — multiclass-classification source.
- `metadata.yml` — recovered project metadata.
- `P2_WU (1).pdf` — recovered project writeup.

**Private-data requirements**

No dataset is listed in this directory. Any data assumed by the original assignment or writeup must be supplied outside version control unless it is confirmed public and redistributable.

**Validation status**

No project-specific tests or recorded validation results are listed. The repository-level import test is the only identified test artifact.

**Limitations**

The filename of the writeup includes a historical duplicate suffix. It is retained as recovered provenance, not normalized evidence of an experiment or result. Algorithmic behavior and public interfaces should be validated from the current source.

**Provenance**

This directory contains the recovered originals associated by topic and filename with the consolidated optimization, linear-model, and multiclass modules.

### `projects/258674342/` — dimensionality reduction and neural models

**Contents**

- `dr.py` — dimensionality-reduction source.
- `nn.py` — neural-network source.
- `softmax.py` — softmax-regression source.
- `metadata.yml` — recovered project metadata.
- `partners.txt` — historical collaboration record.
- `writeup.pdf` — recovered project writeup.

**Private-data requirements**

No dataset is listed in this directory. Neural-network and dimensionality-reduction experiments often depend on external input data; use a local, ignored configuration for non-public data and document the source and license of any public replacement data.

**Validation status**

No project-specific tests or recorded run results are listed. The repository-level `tests/test_imports.py` provides only identified import-level coverage, with no supplied outcome.

**Limitations**

Training behavior, convergence, numerical stability, and reproducibility cannot be inferred from filenames or recovered writeups alone. Avoid presenting model quality claims without reproducible data, configuration, and executed evaluation.

**Provenance**

This directory preserves recovered source, metadata, collaboration information, and a writeup. The corresponding `src/` modules provide the current repository organization.

## Setup and validation

Install the repository according to `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Then run the available checks where their tools are installed:

```bash
python -m pytest
ruff check .
```

These commands are recommendations, not recorded passing results. The configured CI workflow in `.github/workflows/ci.yml` should also be reviewed for the repository's intended automated checks.

## Repository hygiene notes

- Do not use `.ruff_cache/` as a source of project history or validation evidence; it is generated cache content.
- Keep recovered PDFs, metadata, and partner records separate from maintained implementation documentation.
- Do not add private data, institutional paths, secrets, or assignment-only infrastructure to the repository.
- Before publishing or redistributing recovered artifacts, review `LICENSE_REVIEW.md` and confirm that the material is eligible for public release.
