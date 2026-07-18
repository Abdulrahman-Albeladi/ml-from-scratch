# ML From Scratch

Machine-learning algorithms implemented from first principles using Python and NumPy.

**Technologies:** Python · NumPy · pytest

## Highlights

- Core classifiers and optimization logic implemented without high-level model APIs.
- Linear, multiclass, dimensionality-reduction, and neural-network exercises.
- Packaged source layout with tests for selected reusable components.

## Projects

| Project | Location |
|---|---|
| Binary Classifiers | [`projects/binary-classifiers`](projects/binary-classifiers) |
| Linear and Multiclass Models | [`projects/linear-and-multiclass-models`](projects/linear-and-multiclass-models) |
| PCA and Neural Networks | [`projects/pca-and-neural-networks`](projects/pca-and-neural-networks) |
| Reusable package | [`src`](src) |
| Tests | [`tests`](tests) |

## Getting started

1. Create a Python 3.10+ virtual environment.
2. Run `python -m pip install -e ".[dev]"` from the repository root.
3. Run `pytest` for the packaged components.

## Portfolio note

The repository is educational: it emphasizes the mathematics and implementation mechanics behind common ML methods.

## License and attribution

Third-party and collaborator attribution is documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Use and redistribution are governed by the repository's [`LICENSE`](LICENSE).
