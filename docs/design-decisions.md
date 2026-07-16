# Design decisions

## Reconstruction boundary

This repository is documented as an independent reconstruction of machine-learning implementation themes. That means the public version should emphasize:

- fresh module boundaries
- neutral naming not tied to assignment wording
- examples built from synthetic or clearly publishable data
- documentation that explains implementation choices without reproducing course framing

## Why a single repository

The transformed manifest combines overlapping topics that are often taught separately:

- linear models
- optimization
- multiclass reduction
- nonparametric and tree-based classifiers
- dimensionality reduction
- a small neural network

Keeping them together is reasonable when the goal is to show implementation breadth and consistent numerical style.

## Package structure

The current structure separates algorithms by role:

- `src/optimization/` for reusable optimization logic
- `src/linear_models/` for linear prediction methods
- `src/classifiers/` for general classification algorithms and wrappers
- `src/decomposition/` for representation learning or dimensionality reduction
- `src/neural_nets/` for neural-network code

This is a better portfolio structure than publishing multiple assignment-shaped repositories.

## Documentation constraints

Documentation should avoid claims that are not supported by evidence. In particular:

- do not claim benchmark accuracy unless reproducible results are added
- do not claim tests pass unless test execution evidence exists
- do not describe datasets unless they are actually included or clearly referenced

## Validation-driven cleanup priorities

Observed from validation evidence:

1. `ruff` found unused imports.
2. `black --check` indicates formatting drift.
3. `bandit` flagged `assert` usage in runtime paths.
4. `pytest` was skipped, so behavior is not yet validated by executed tests.

Recommended order:

1. remove cache artifacts from version control
2. normalize formatting and lint issues
3. replace runtime `assert` checks with explicit exceptions where needed
4. add unit tests around fit/predict behavior, shape handling, and edge cases

## Publication risk notes

This repository is a public candidate, but publication is appropriate only if:

- no raw coursework files remain
- no course-specific prompts or grading artifacts remain
- ownership and attribution are clear
- examples do not expose restricted data

If any source file still tracks too closely to a course submission, route it to manual review and rewrite before release.
