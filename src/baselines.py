"""scikit-learn baselines with hyperparameters MATCHED to our own models.

sklearn is used here only as a reference to benchmark against, never as one of
our core models. "Matched" means each baseline minimises the same objective
(or grows the same kind of tree) as our NumPy model, so any remaining gap is
due to the optimiser or tie-breaking, not a different problem.

Matching rules
--------------
Trees      same criterion, max_depth, min_samples_leaf, min_samples_split and
           min_impurity_decrease. sklearn's "squared_error" == our "mse".
Linear SVM our objective  J = (lambda/2)||w||^2 + (1/n) sum hinge_i
           SVC objective  (1/2)||w||^2 + C sum hinge_i
           multiply ours by 1/lambda  ->  C = 1 / (lambda * n_train).
           SGDClassifier(loss="hinge") minimises alpha*(1/2)||w||^2 + mean
           hinge, so alpha = lambda directly. Neither regularises the bias.
RBF SVM    our RFF map approximates exp(-gamma ||x-y||^2); sklearn's
           RBFSampler uses the same kernel and the same N(0, 2*gamma*I)
           frequencies, so RBFSampler + SGD hinge is the matched pipeline.
           (An exact kernel SVC is O(n^2) and impractical on ~70k rows.)
All SVM baselines sit inside a Pipeline with StandardScaler, so the scaler
statistics are learned from the training rows only.
"""
from __future__ import annotations

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42

TREE_KEYS = ("max_depth", "min_samples_split", "min_samples_leaf", "min_impurity_decrease")


def _tree_kwargs(config: dict) -> dict:
    """Keep only the hyperparameters both implementations understand."""
    return {key: config[key] for key in TREE_KEYS if key in config}


def svc_C_from_lambda(lambda_: float, n_train: int) -> float:
    """C that makes SVC's objective a rescaling of the Pegasos objective."""
    if lambda_ <= 0 or n_train <= 0:
        raise ValueError("lambda_ and n_train must be positive")
    return 1.0 / (lambda_ * n_train)


def build_baselines(
    tree_regression: dict,
    tree_classification: dict,
    svm: dict,
    n_train: int,
    random_state: int = SEED,
) -> dict:
    """Unfitted sklearn models whose settings match ours.

    tree_regression      e.g. {"max_depth": 8, "min_samples_leaf": 100}
    tree_classification  same keys plus "criterion": "gini" | "entropy"
    svm                  {"lambda_": 1e-3, "n_iters": 20000} and, if our final
                         SVM used random Fourier features, also
                         {"gamma": ..., "n_components": ...}
    """
    lambda_ = svm["lambda_"]
    models = {
        "DecisionTreeRegressor": DecisionTreeRegressor(
            criterion="squared_error",
            random_state=random_state,
            **_tree_kwargs(tree_regression),
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            criterion=tree_classification.get("criterion", "gini"),
            random_state=random_state,
            **_tree_kwargs(tree_classification),
        ),
        "StandardScaler+SVC(linear)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="linear",
                        C=svc_C_from_lambda(lambda_, n_train),
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "StandardScaler+SGD(hinge)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "sgd",
                    SGDClassifier(
                        loss="hinge",
                        penalty="l2",
                        alpha=lambda_,
                        max_iter=2000,
                        tol=1e-3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    if svm.get("gamma") is not None:
        models["StandardScaler+RBFSampler+SGD(hinge)"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "rff",
                    RBFSampler(
                        gamma=svm["gamma"],
                        n_components=svm.get("n_components", 256),
                        random_state=random_state,
                    ),
                ),
                (
                    "sgd",
                    SGDClassifier(
                        loss="hinge",
                        penalty="l2",
                        alpha=lambda_,
                        max_iter=2000,
                        tol=1e-3,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


REGRESSION_BASELINES = ("DecisionTreeRegressor",)


def fit_baselines(
    models: dict, X_train, y_price_train, y_tier_train
) -> dict:
    """Fit every baseline on the TRAINING split only; returns fit seconds."""
    from .evaluate import timeit

    fit_seconds = {}
    for name, model in models.items():
        target = y_price_train if name in REGRESSION_BASELINES else y_tier_train
        _, fit_seconds[name] = timeit(model.fit, X_train, target)
    return fit_seconds
