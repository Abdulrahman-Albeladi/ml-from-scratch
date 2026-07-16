# Contributing

## Scope for contributions

Useful contributions for this repository include:

- focused unit tests for algorithm behavior
- formatting and lint cleanup
- API consistency improvements across estimators
- small synthetic examples that do not depend on restricted data
- documentation clarifying assumptions, shapes, and failure modes

## Do not contribute

Do not add:

- raw coursework submissions
- assignment prompts or grading rubrics
- instructor or autograder tests
- starter code with unclear publication rights
- private or restricted datasets

## Development workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
ruff check .
black --check .
bandit -q -r src
pytest -q
```

## Code expectations

- keep implementations readable and numerically explicit
- prefer small, deterministic tests
- document array shape expectations in docstrings or tests
- use explicit exceptions for runtime validation where production-style behavior matters
- avoid adding benchmark claims without reproducible evidence

## Pull requests

A pull request should describe:

- what module changed
- what behavior was added or corrected
- what validation was run locally
- whether any public-release risk was identified
