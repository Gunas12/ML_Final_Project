"""Unit tests for the from-scratch bonus models (synthetic data only)."""
import numpy as np
import pytest

from src.bonus_boosting import AdaBoost
from src.bonus_ensembles import RandomForest
from src.bonus_linear import RidgeScratch, SVRScratch
from src.decision_tree import DecisionTree


def _regression_data(n=400, p=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    w = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
    return X, X @ w + 1.5 + 0.05 * rng.normal(size=n), w


def _classification_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    return X, (X[:, 0] + X[:, 1] ** 2 > 1.0).astype(int)


def test_ridge_alpha_zero_recovers_least_squares():
    X, y, w = _regression_data()
    m = RidgeScratch(0.0).fit(X, y)
    assert np.allclose(m.w, w, atol=0.05) and abs(m.b - 1.5) < 0.05


def test_ridge_shrinks_with_alpha():
    X, y, _ = _regression_data()
    assert np.linalg.norm(RidgeScratch(1000).fit(X, y).w) < np.linalg.norm(RidgeScratch(0.1).fit(X, y).w)


def test_ridge_matches_sklearn():
    from sklearn.linear_model import Ridge
    X, y, _ = _regression_data()
    ours, ref = RidgeScratch(3.0).fit(X, y), Ridge(alpha=3.0).fit(X, y)
    assert np.allclose(ours.w, ref.coef_, atol=1e-8) and abs(ours.b - ref.intercept_) < 1e-8


def test_svr_objective_decreases_and_fits():
    X, y, _ = _regression_data()
    y = (y - y.mean()) / y.std()
    m = SVRScratch(1e-2, 0.05, 6000).fit(X, y)
    assert m.loss_history_[-1] < m.loss_history_[0]
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.95


def test_svr_rejects_bad_params():
    with pytest.raises(ValueError):
        SVRScratch(lambda_=0).fit(np.zeros((3, 2)), np.zeros(3))


def test_forest_beats_single_tree_and_is_reproducible():
    X, y = _classification_data()
    tr, te = slice(0, 400), slice(400, 600)
    forest = RandomForest("classification", 25, random_state=1).fit(X[tr], y[tr])
    single = DecisionTree("classification", random_state=1).fit(X[tr], y[tr])
    acc_f = np.mean(forest.predict(X[te]) == y[te])
    acc_t = np.mean(single.predict(X[te]) == y[te])
    assert acc_f >= acc_t - 0.03   # forest is at least as good up to noise
    again = RandomForest("classification", 25, random_state=1).fit(X[tr], y[tr])
    assert np.array_equal(forest.predict(X[te]), again.predict(X[te]))
    assert 0.5 < forest.oob_score_ <= 1.0
    assert np.allclose(forest.predict_proba(X[te]).sum(axis=1), 1.0)


def test_forest_regression_and_staged_outputs():
    X, y, _ = _regression_data()
    f = RandomForest("regression", 10, min_samples_leaf=2).fit(X[:300], y[:300])
    stages = f.staged_outputs(X[300:])
    assert len(stages) == 10 and np.allclose(stages[-1], f.predict(X[300:]))
    assert f.oob_score_ > 0.5


def test_tree_max_features_changes_tree_but_default_does_not():
    X, y = _classification_data()
    base = DecisionTree("classification", random_state=3).fit(X, y)
    same = DecisionTree("classification", random_state=3, max_features=None).fit(X, y)
    sub = DecisionTree("classification", random_state=3, max_features=1).fit(X, y)
    assert base.n_nodes_ == same.n_nodes_
    assert sub.n_nodes_ != base.n_nodes_
    with pytest.raises(ValueError):
        DecisionTree(max_features=0).fit(X, y)


def test_adaboost_improves_over_one_stump():
    X, y = _classification_data()
    tr, te = slice(0, 400), slice(400, 600)
    one = AdaBoost(1, 1).fit(X[tr], y[tr])
    many = AdaBoost(40, 2).fit(X[tr], y[tr])
    assert np.mean(many.predict(X[te]) == y[te]) >= np.mean(one.predict(X[te]) == y[te])
    assert len(many.alphas_) > 1 and set(np.unique(many.predict(X[te]))) <= {0, 1}
    assert np.array_equal(many.predict(X[te]), AdaBoost(40, 2).fit(X[tr], y[tr]).predict(X[te]))
