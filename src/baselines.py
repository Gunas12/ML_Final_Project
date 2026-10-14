from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def build_baselines(random_state: int = 42):
    return {
        "DecisionTreeRegressor": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=100, random_state=random_state
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=100,
            random_state=random_state,
        ),
        "StandardScaler+SVC": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svc",
                    SVC(
                        kernel="linear",
                        random_state=random_state,
                        tol=1e-2,
                        max_iter=5000,
                    ),
                ),
            ]
        ),
        "StandardScaler+SGD hinge": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "sgd",
                    SGDClassifier(
                        loss="hinge",
                        max_iter=2000,
                        tol=1e-3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def fit_baselines(X_train, y_price_train, y_tier_train, random_state: int = 42):
    models = build_baselines(random_state=random_state)
    models["DecisionTreeRegressor"].fit(X_train, y_price_train)
    for name in (
        "DecisionTreeClassifier",
        "StandardScaler+SVC",
        "StandardScaler+SGD hinge",
    ):
        models[name].fit(X_train, y_tier_train)
    return models
