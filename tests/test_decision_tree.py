import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from src.decision_tree import DecisionTree


# ---------------------------------------------------------------- impurity
def test_gini_entropy_known_values():
    y = np.array([0, 0, 1, 1])
    assert np.isclose(DecisionTree(criterion="gini")._impurity(y), 0.5)
    assert np.isclose(DecisionTree(criterion="entropy")._impurity(y), 1.0)


def test_pure_node_has_zero_impurity():
    for crit in ("gini", "entropy"):
        assert DecisionTree(criterion=crit)._impurity(np.ones(7, dtype=int)) == 0.0


def test_mse_impurity_is_variance():
    t = DecisionTree(task="regression", criterion="mse")
    assert np.isclose(t._impurity(np.array([1.0, 2.0, 3.0])), 2.0 / 3.0)
    assert t._impurity(np.array([5.0, 5.0, 5.0])) == 0.0


def test_invalid_task_criterion_combo_raises():
    with pytest.raises(ValueError):
        DecisionTree(task="regression", criterion="gini").fit(np.zeros((3, 1)), np.zeros(3))
    with pytest.raises(ValueError):
        DecisionTree(task="classification", criterion="mse").fit(np.zeros((3, 1)), np.zeros(3))


# --------------------------------------------------------------- best split
def test_best_split_hand_example():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    f, thr, dec = DecisionTree()._best_split(X, y)
    assert f == 0 and np.isclose(thr, 2.5) and np.isclose(dec, 0.5)


def test_best_split_picks_informative_feature():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(60, 1))
    signal = np.r_[np.zeros(30), np.ones(30)].reshape(-1, 1) + 0.01 * rng.normal(size=(60, 1))
    X = np.hstack([noise, signal])
    y = np.r_[np.zeros(30), np.ones(30)].astype(int)
    f, thr, dec = DecisionTree()._best_split(X, y)
    assert f == 1 and 0.0 < thr < 1.0 and np.isclose(dec, 0.5)


def test_best_split_none_cases():
    t = DecisionTree()
    assert t._best_split(np.arange(5.0).reshape(-1, 1), np.ones(5, dtype=int)) is None      # pure
    assert t._best_split(np.ones((6, 2)), np.array([0, 1, 0, 1, 0, 1])) is None              # constant features
    assert DecisionTree(min_samples_leaf=4)._best_split(
        np.arange(6.0).reshape(-1, 1), np.array([0, 0, 0, 1, 1, 1])) is None                  # leaf size blocks all
    assert t._best_split(np.array([[1.0]]), np.array([0])) is None                            # single sample


def test_never_splits_between_equal_values():
    X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    y = np.array([0, 1, 0, 1, 0, 1])
    f, thr, _ = DecisionTree()._best_split(X, y)
    assert 1.0 < thr < 2.0


def test_min_samples_leaf_respected():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    y = rng.integers(0, 2, 80)
    for msl in (1, 5, 20, 40):
        s = DecisionTree(min_samples_leaf=msl)._best_split(X, y)
        if s is None:
            continue
        f, thr, _ = s
        n_left = int(np.sum(X[:, f] <= thr))
        assert n_left >= msl and 80 - n_left >= msl


def test_weighted_decrease_uses_total_n():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    t = DecisionTree()
    t._n_total = 8                                # node holds half of the training data
    assert np.isclose(t._best_split(X, y)[2], 0.25)


# ------------------------------------------- brute force reference (O(n^2))
def _brute_force_gain(tree, X, y):
    n = len(y)
    parent = tree._impurity(y)
    best = -np.inf
    for j in range(X.shape[1]):
        vals = np.unique(X[:, j])
        for a, b in zip(vals[:-1], vals[1:]):
            thr = (a + b) / 2
            m = X[:, j] <= thr
            if m.sum() < tree.min_samples_leaf or (~m).sum() < tree.min_samples_leaf:
                continue
            child = (m.sum() * tree._impurity(y[m]) + (~m).sum() * tree._impurity(y[~m])) / n
            best = max(best, parent - child)
    return best


@pytest.mark.parametrize("task,crit", [("classification", "gini"), ("classification", "entropy"),
                                       ("regression", "mse")])
@pytest.mark.parametrize("msl", [1, 3])
def test_matches_bruteforce(task, crit, msl):
    rng = np.random.default_rng(5)
    for _ in range(5):
        X = np.round(rng.normal(size=(40, 3)), 1)            # rounding creates tied values on purpose
        y = rng.integers(0, 3, 40) if task == "classification" else rng.normal(size=40) * 10 + 100
        t = DecisionTree(task=task, criterion=crit, min_samples_leaf=msl)
        _, _, dec = t._best_split(X, y)
        assert np.isclose(dec, _brute_force_gain(t, X, y))


# ------------------------------------------------ same root split as sklearn
@pytest.mark.parametrize("crit", ["gini", "entropy"])
def test_root_split_matches_sklearn_classification(crit):
    for seed in range(10):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(50, 4))
        y = (X[:, 2] + 0.5 * rng.normal(size=50) > 0).astype(int)
        sk = DecisionTreeClassifier(max_depth=1, criterion=crit, random_state=0).fit(X, y)
        f, thr, _ = DecisionTree(criterion=crit)._best_split(X, y)
        assert f == sk.tree_.feature[0]
        assert np.isclose(thr, sk.tree_.threshold[0])


def test_root_split_matches_sklearn_regression():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(60, 4))
        y = 3 * X[:, 1] + rng.normal(size=60)
        sk = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y)
        f, thr, _ = DecisionTree(task="regression", criterion="mse")._best_split(X, y)
        assert f == sk.tree_.feature[0]
        assert np.isclose(thr, sk.tree_.threshold[0])