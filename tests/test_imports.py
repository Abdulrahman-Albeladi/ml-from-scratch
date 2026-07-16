from importlib import import_module

MODULES = [
    "classifiers.baselines",
    "classifiers.decision_tree",
    "classifiers.knn",
    "classifiers.multiclass",
    "decomposition.pca",
    "linear_models.linear_classifiers",
    "linear_models.perceptron",
    "linear_models.softmax_regression",
    "neural_nets.mlp",
    "optimization.gradient_descent",
]


def test_modules_import():
    for module_name in MODULES:
        import_module(module_name)
