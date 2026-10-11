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


# =====================================================================
# Step 2: full tree (grow / predict / predict_proba / importances)
# =====================================================================
def _clf_data(seed, n=60, d=4, classes=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    score = X[:, 0] + 0.7 * X[:, 1] + 0.5 * rng.normal(size=n)
    y = np.digitize(score, np.quantile(score, np.linspace(0, 1, classes + 1)[1:-1]))
    return X, y


def _reg_data(seed, n=60, d=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 5 * X[:, 0] - 3 * (X[:, 1] > 0) + rng.normal(size=n)
    return X, y


PARAM_GRID = [
    {},
    {"max_depth": 3},
    {"min_samples_leaf": 5},
    {"min_samples_split": 10},
    {"min_impurity_decrease": 0.01},
    {"max_depth": 4, "min_samples_leaf": 3, "min_samples_split": 8, "min_impurity_decrease": 0.005},
]


def _split_gain(tree, Xn, yn, feature, thr, n_total):
    """Weighted impurity decrease of an arbitrary split (reference computation)."""
    m = Xn[:, feature] <= thr
    child = (m.sum() * tree._impurity(yn[m]) + (~m).sum() * tree._impurity(yn[~m])) / len(yn)
    return (tree._impurity(yn) - child) * len(yn) / n_total


def _first_divergence(mine, sk, sid, Xn, yn, ref, n_total):
    """
    Walk my tree and sklearn's tree together. Return None if they are identical,
    else (my_gain, sklearn_gain) at the first node where they choose different splits.
    """
    sk_leaf = sk.children_left[sid] == -1
    if mine.is_leaf() or sk_leaf:
        assert mine.is_leaf() == sk_leaf, "one tree stops where the other keeps splitting"
        return None
    f, thr = mine.feature, mine.threshold
    sf, st = int(sk.feature[sid]), float(sk.threshold[sid])
    if f != sf or abs(thr - st) > 1e-6:
        return (_split_gain(ref, Xn, yn, f, thr, n_total), _split_gain(ref, Xn, yn, sf, st, n_total))
    m = Xn[:, f] <= thr
    return (_first_divergence(mine.left, sk, sk.children_left[sid], Xn[m], yn[m], ref, n_total)
            or _first_divergence(mine.right, sk, sk.children_right[sid], Xn[~m], yn[~m], ref, n_total))


@pytest.mark.parametrize("params", PARAM_GRID)
@pytest.mark.parametrize("crit", ["gini", "entropy"])
def test_split_choices_are_optimal_like_sklearn_classification(params, crit):
    """
    Trees may differ ONLY where several splits have exactly the same gain
    (scikit-learn breaks such ties randomly, we pick the lowest feature index).
    So at the first divergence the two gains must be equal.
    """
    for seed in range(6):
        k = 3 if seed % 2 else 2
        X, y = _clf_data(seed, classes=k)
        mine = DecisionTree(criterion=crit, **params).fit(X, y)
        sk = DecisionTreeClassifier(criterion=crit, random_state=0, **params).fit(X, y)
        ref = DecisionTree(criterion=crit, **params)
        ref._n_total = len(y)
        div = _first_divergence(mine.root, sk.tree_, 0, X, np.searchsorted(mine.classes_, y), ref, len(y))
        if div is None:
            assert mine.depth_ == sk.get_depth() and mine.n_leaves_ == sk.get_n_leaves()
        else:
            assert np.isclose(div[0], div[1], atol=1e-9), f"my gain {div[0]} vs sklearn gain {div[1]}"


@pytest.mark.parametrize("params", PARAM_GRID)
def test_split_choices_are_optimal_like_sklearn_regression(params):
    for seed in range(6):
        X, y = _reg_data(seed)
        mine = DecisionTree(task="regression", criterion="mse", **params).fit(X, y)
        sk = DecisionTreeRegressor(random_state=0, **params).fit(X, y)
        ref = DecisionTree(task="regression", criterion="mse", **params)
        ref._n_total = len(y)
        div = _first_divergence(mine.root, sk.tree_, 0, X, y, ref, len(y))
        if div is None:
            assert mine.depth_ == sk.get_depth() and mine.n_leaves_ == sk.get_n_leaves()
        else:
            assert np.isclose(div[0], div[1], atol=1e-9 * max(1.0, abs(div[0])))


# Configurations where no ties occur on this data: the trees must be IDENTICAL.
@pytest.mark.parametrize("params", [{"max_depth": 2}, {"min_samples_split": 120}])
@pytest.mark.parametrize("crit", ["gini", "entropy"])
def test_identical_to_sklearn_classification(params, crit):
    for seed in range(8):
        k = 3 if seed % 2 else 2
        X, y = _clf_data(seed, n=300, classes=k)
        Xte, _ = _clf_data(seed + 100, n=500, classes=k)
        mine = DecisionTree(criterion=crit, **params).fit(X, y)
        sk = DecisionTreeClassifier(criterion=crit, random_state=0, **params).fit(X, y)
        assert mine.depth_ == sk.get_depth() and mine.n_leaves_ == sk.get_n_leaves()
        assert (mine.predict(Xte) == sk.predict(Xte)).all()
        assert np.allclose(mine.predict_proba(Xte), sk.predict_proba(Xte))
        assert np.allclose(mine.feature_importances_, sk.feature_importances_)


@pytest.mark.parametrize("params", [
    {"max_depth": 3}, {"min_samples_leaf": 40}, {"min_samples_split": 120}, {"min_impurity_decrease": 1.0},
    {"max_depth": 4, "min_samples_leaf": 15, "min_samples_split": 40, "min_impurity_decrease": 0.005},
])
def test_identical_to_sklearn_regression(params):
    for seed in range(8):
        X, y = _reg_data(seed, n=300)
        Xte, _ = _reg_data(seed + 100, n=500)
        mine = DecisionTree(task="regression", criterion="mse", **params).fit(X, y)
        sk = DecisionTreeRegressor(random_state=0, **params).fit(X, y)
        assert mine.depth_ == sk.get_depth() and mine.n_leaves_ == sk.get_n_leaves()
        assert np.allclose(mine.predict(Xte), sk.predict(Xte))
        assert np.allclose(mine.feature_importances_, sk.feature_importances_)


# ---------------------------------------------- hyperparameters change the tree
def test_max_depth_limits_and_grows_tree():
    X, y = _clf_data(0, n=300)
    leaves = []
    for d in (1, 2, 3, 5, 8):
        t = DecisionTree(max_depth=d).fit(X, y)
        assert t.depth_ <= d
        leaves.append(t.n_leaves_)
    assert leaves == sorted(leaves) and leaves[0] < leaves[-1]


def test_min_samples_leaf_enforced_and_shrinks_tree():
    X, y = _clf_data(1, n=300)
    prev = None
    for msl in (1, 5, 20, 60):
        t = DecisionTree(min_samples_leaf=msl).fit(X, y)
        assert all(nd.n_samples >= msl for nd, _ in t._nodes() if nd.is_leaf())
        if prev is not None:
            assert t.n_leaves_ <= prev
        prev = t.n_leaves_
    assert DecisionTree(min_samples_leaf=1).fit(X, y).n_leaves_ > DecisionTree(min_samples_leaf=60).fit(X, y).n_leaves_


def test_min_samples_split_enforced():
    X, y = _clf_data(2, n=300)
    t = DecisionTree(min_samples_split=40).fit(X, y)
    assert all(nd.n_samples >= 40 for nd, _ in t._nodes() if not nd.is_leaf())
    assert t.n_leaves_ < DecisionTree().fit(X, y).n_leaves_


def test_min_impurity_decrease_prunes_tree():
    X, y = _clf_data(3, n=300)
    sizes = [DecisionTree(min_impurity_decrease=m).fit(X, y).n_nodes_ for m in (0.0, 0.005, 0.02, 0.5)]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] > sizes[-1]
    assert sizes[-1] == 1                         # too demanding -> root stays a leaf


# ------------------------------------------------------------- predictions
def test_predict_proba_rows_sum_to_one_and_agree_with_predict():
    X, y = _clf_data(4, n=200, classes=3)
    t = DecisionTree(max_depth=4).fit(X, y)
    P = t.predict_proba(X)
    assert P.shape == (200, 3) and np.allclose(P.sum(axis=1), 1.0)
    assert (t.classes_[P.argmax(axis=1)] == t.predict(X)).all()


def test_predict_one_matches_batch_predict():
    X, y = _clf_data(5, n=80)
    t = DecisionTree(max_depth=5).fit(X, y)
    singles = [t.classes_[int(np.argmax(t._predict_one(x, t.root)))] for x in X]
    assert (np.array(singles) == t.predict(X)).all()


def test_constant_target_gives_single_leaf():
    X = np.random.default_rng(0).normal(size=(20, 3))
    t = DecisionTree().fit(X, np.ones(20, dtype=int))
    assert t.n_nodes_ == 1 and t.depth_ == 0 and (t.predict(X) == 1).all()
    assert np.allclose(t.feature_importances_, 0.0)
    r = DecisionTree(task="regression", criterion="mse").fit(X, np.full(20, 7.5))
    assert r.n_nodes_ == 1 and np.allclose(r.predict(X), 7.5)


def test_unlimited_depth_memorizes_training_data():
    X, y = _clf_data(6, n=150, classes=3)
    assert (DecisionTree().fit(X, y).predict(X) == y).all()
    Xr, yr = _reg_data(6, n=150)
    assert np.allclose(DecisionTree(task="regression", criterion="mse").fit(Xr, yr).predict(Xr), yr)


def test_string_labels_supported():
    X, y = _clf_data(7, n=100)
    labels = np.where(y == 1, "premium", "standard")
    t = DecisionTree(max_depth=3).fit(X, labels)
    assert set(t.predict(X)) <= {"premium", "standard"}
    assert list(t.classes_) == ["premium", "standard"]


def test_xor_needs_zero_gain_first_split():
    X = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float), (10, 1))
    y = np.tile([0, 1, 1, 0], 10)
    # every first split has gain 0, but depth 2 solves XOR (same behaviour as sklearn)
    assert (DecisionTree(max_depth=2).fit(X, y).predict(X) == y).all()
    assert DecisionTree(max_depth=1).fit(X, y).depth_ == 1       # a zero-gain split is still made


def test_feature_importances_sum_to_one_and_find_signal():
    X, y = _clf_data(8, n=300)
    imp = DecisionTree(max_depth=4).fit(X, y).feature_importances_
    assert np.isclose(imp.sum(), 1.0) and np.argmax(imp) in (0, 1)


# -------------------------------------------------------------- robustness
def test_errors_before_fit_and_on_bad_input():
    t = DecisionTree()
    with pytest.raises(RuntimeError):
        t.predict(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        _ = t.feature_importances_
    X, y = _clf_data(9, n=40)
    t.fit(X, y)
    with pytest.raises(ValueError):
        t.predict(np.zeros((3, 99)))
    with pytest.raises(ValueError):
        DecisionTree().fit(np.array([[np.nan, 1.0], [2.0, 3.0]]), np.array([0, 1]))
    with pytest.raises(ValueError):
        DecisionTree(task="regression", criterion="mse").fit(X, y).predict_proba(X)


def test_fit_returns_self_and_repr_and_export_text():
    X, y = _clf_data(10, n=60)
    t = DecisionTree(max_depth=2)
    assert "not fitted" in repr(t)
    assert t.fit(X, y) is t
    assert "depth=" in repr(t)
    text = t.export_text(feature_names=["a", "b", "c", "d"])
    assert "<=" in text and "class:" in text