"""Tests for src/baselines.py: the sklearn baselines must be MATCHED to ours."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import baselines
from src.decision_tree import DecisionTree

TREE_REG = {"max_depth": 6, "min_samples_leaf": 20, "min_samples_split": 40}
TREE_CLF = {"criterion": "entropy", "max_depth": 5, "min_samples_leaf": 10}
SVM_CFG = {"lambda_": 1e-3, "n_iters": 1000}


def _models(**svm_extra):
    return baselines.build_baselines(
        TREE_REG, TREE_CLF, {**SVM_CFG, **svm_extra}, n_train=1000
    )


def test_svc_C_is_one_over_lambda_n():
    assert baselines.svc_C_from_lambda(1e-3, 1000) == pytest.approx(1.0)
    assert baselines.svc_C_from_lambda(1e-2, 500) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        baselines.svc_C_from_lambda(0.0, 10)


def test_tree_baselines_carry_matched_hyperparameters():
    models = _models()
    reg = models["DecisionTreeRegressor"]
    clf = models["DecisionTreeClassifier"]
    assert isinstance(reg, DecisionTreeRegressor) and isinstance(clf, DecisionTreeClassifier)
    assert (reg.max_depth, reg.min_samples_leaf, reg.min_samples_split) == (6, 20, 40)
    assert reg.criterion == "squared_error"
    assert (clf.criterion, clf.max_depth, clf.min_samples_leaf) == ("entropy", 5, 10)


def test_linear_svm_baselines_are_scaled_and_matched():
    models = _models()
    svc = models["StandardScaler+SVC(linear)"]
    sgd = models["StandardScaler+SGD(hinge)"]
    assert list(svc.named_steps)[0] == "scaler" and list(sgd.named_steps)[0] == "scaler"
    assert isinstance(svc.named_steps["svc"], SVC)
    assert svc.named_steps["svc"].C == pytest.approx(1.0)
    assert isinstance(sgd.named_steps["sgd"], SGDClassifier)
    assert sgd.named_steps["sgd"].loss == "hinge"
    assert sgd.named_steps["sgd"].alpha == pytest.approx(1e-3)


def test_rbf_baseline_only_added_when_gamma_given():
    assert "StandardScaler+RBFSampler+SGD(hinge)" not in _models()
    rbf = _models(gamma=0.1, n_components=64)["StandardScaler+RBFSampler+SGD(hinge)"]
    assert rbf.named_steps["rff"].gamma == pytest.approx(0.1)
    assert rbf.named_steps["rff"].n_components == 64


def test_fit_baselines_fits_everything_and_reports_time():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 4))
    price = np.exp(10 + X[:, 0])
    tier = (price > np.median(price)).astype(int)
    models = baselines.build_baselines(TREE_REG, TREE_CLF, SVM_CFG, n_train=300)
    seconds = baselines.fit_baselines(models, X, price, tier)
    assert set(seconds) == set(models) and all(v >= 0 for v in seconds.values())
    assert models["DecisionTreeRegressor"].predict(X).shape == (300,)
    assert models["StandardScaler+SVC(linear)"].predict(X).shape == (300,)


def test_matched_sklearn_tree_agrees_with_our_tree():
    """Same data + same settings -> (nearly) identical predictions."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 5))
    y = np.exp(10 + X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=400))
    ours = DecisionTree(task="regression", criterion="mse", max_depth=4, min_samples_leaf=10).fit(X, y)
    theirs = baselines.build_baselines(
        {"max_depth": 4, "min_samples_leaf": 10}, TREE_CLF, SVM_CFG, n_train=400
    )["DecisionTreeRegressor"].fit(X, y)
    agreement = np.mean(np.isclose(ours.predict(X), theirs.predict(X)))
    assert agreement > 0.99
