"""Member 3 bonus: scratch implementation, numerical safety and comparisons."""
import ast
from pathlib import Path

import numpy as np
import pytest

from src import bonus_boosting as boosting


def toy_data():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(160, 3))
    y = (X[:, 0] + 0.6 * X[:, 1] > 0).astype(int)
    return X, y


def test_boosting_learns_deterministically_and_stages_match_final_score():
    X, y = toy_data()
    original = X.copy()
    first = boosting.AdaBoost(20, 1, 42).fit(X, y)
    second = boosting.AdaBoost(20, 1, 42).fit(X, y)
    assert np.mean(first.predict(X) == y) > 0.90
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    np.testing.assert_array_equal(first.alphas_, second.alphas_)
    stages = list(first.staged_decision_function(X))
    np.testing.assert_allclose(stages[-1], first.decision_function(X))
    assert len(stages) == len(first.trees_) <= 20
    assert np.isfinite(first.alphas_).all()
    np.testing.assert_array_equal(X, original)


def test_perfect_weak_learner_stops_with_finite_alpha_and_string_labels():
    X = np.r_[np.zeros(50), np.ones(50)][:, None]
    y = np.where(X[:, 0] > 0, "premium", "standard")
    model = boosting.AdaBoost(100).fit(X, y)
    assert len(model.trees_) == 1
    assert np.isfinite(model.alphas_).all()
    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize("params", [dict(n_estimators=0), dict(n_estimators=1.5),
                                   dict(max_depth=0), dict(max_depth=True)])
def test_invalid_parameters_fail_clearly(params):
    with pytest.raises(ValueError):
        boosting.AdaBoost(**params).fit(*toy_data())


@pytest.mark.parametrize("X,y", [([1, 2], [0, 1]), ([[0], [np.nan]], [0, 1]),
                                 ([[0], [np.inf]], [0, 1]), ([[0], [1]], [[0], [1]]),
                                 ([[0], [1]], [0]), ([[0], [1]], [0, 0]),
                                 ([[0], [1]], [0, np.nan]), ([[0], [1]], [None, 1]),
                                 ([[0], [1]], np.array([0, np.inf], dtype=object))])
def test_invalid_training_data_fail_clearly(X, y):
    with pytest.raises(ValueError):
        boosting.AdaBoost().fit(X, y)


def test_unfitted_and_wrong_prediction_shapes_fail_clearly():
    with pytest.raises(ValueError, match="fitted"):
        boosting.AdaBoost().predict([[0]])
    model = boosting.AdaBoost(5).fit(*toy_data())
    for X in ([[0]], [0, 1, 2], [[0, np.inf, 1]]):
        with pytest.raises(ValueError):
            model.predict(X)


def test_no_useful_weak_learner_does_not_silently_predict_one_class():
    model = boosting.AdaBoost(3)
    with pytest.raises(ValueError, match="better than chance"):
        model.fit(np.ones((20, 1)), np.tile([0, 1], 10))
    with pytest.raises(ValueError, match="fitted"):
        model.predict([[1]])


def test_unlucky_resample_preserves_previously_learned_weights(monkeypatch):
    probabilities = []

    class FakeRNG:
        def choice(self, n, size, replace, p):
            probabilities.append(p.copy())
            return np.zeros(4, dtype=int) if len(probabilities) == 2 else np.arange(4)

        def integers(self, high):
            return 1

    class FakeTree:
        count = 0

        def __init__(self, *args, **kwargs):
            FakeTree.count += 1
            self.number = FakeTree.count

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array([-1, -1, 1, -1] if self.number == 1 else [-1, -1, 1, 1])

    monkeypatch.setattr(boosting.np.random, "default_rng", lambda seed: FakeRNG())
    monkeypatch.setattr(boosting, "DecisionTree", FakeTree)
    model = boosting.AdaBoost(2).fit(np.arange(4)[:, None], [0, 0, 1, 1])
    assert len(model.trees_) == 2
    np.testing.assert_allclose(probabilities[1], [1/6, 1/6, 1/6, 1/2])
    np.testing.assert_array_equal(probabilities[1], probabilities[2])


def test_sklearn_is_only_used_by_explicit_discrete_comparison(monkeypatch):
    import sklearn.ensemble
    captured = {}

    class Reference:
        def __init__(self, estimator, n_estimators, random_state, learning_rate, algorithm=None):
            captured.update(estimator=estimator, algorithm=algorithm)

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array([0, 1])

        def decision_function(self, X):
            return np.array([-1, 1])

    monkeypatch.setattr(sklearn.ensemble, "AdaBoostClassifier", Reference)
    scores = boosting._sklearn_reference([[0], [1]], [0, 1], [[0], [1]], [0, 1], 5, 1, 42)
    assert captured["algorithm"] == "SAMME"
    assert captured["estimator"].criterion == "gini"
    assert scores["F1"] == 1
    tree = ast.parse(Path(boosting.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_sklearn_reference":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                assert not (child.module or "").startswith("sklearn")
            if isinstance(child, ast.Import):
                assert not any(alias.name.startswith("sklearn") for alias in child.names)


def test_new_sklearn_api_does_not_receive_removed_algorithm_keyword(monkeypatch):
    import sklearn.ensemble

    class NewReference:
        def __init__(self, estimator, n_estimators, random_state, learning_rate):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array([0, 1])

        def decision_function(self, X):
            return np.array([-1, 1])

    monkeypatch.setattr(sklearn.ensemble, "AdaBoostClassifier", NewReference)
    assert boosting._sklearn_reference([[0], [1]], [0, 1], [[0], [1]], [0, 1], 2, 1, 42)["F1"] == 1


def test_fast_bonus_outputs_are_separate_and_data_path_is_forwarded(tmp_path, monkeypatch):
    from src import run_all
    X, y = toy_data()
    calls = []

    def prepare(path, fast):
        calls.append((path, fast))
        return {"X_tr": X[:100], "tier_tr": y[:100],
                "X_val": X[100:130], "tier_val": y[100:130],
                "X_te": X[130:], "tier_te": y[130:]}

    monkeypatch.setattr(run_all, "prepare_data", prepare)
    monkeypatch.chdir(tmp_path)
    result = boosting.run(fast=True, data_path="chosen.csv")
    assert calls == [("chosen.csv", True)]
    assert result["reference_algorithm"] == "SAMME"
    assert result["best_depth"] in (1, 3)
    assert Path(result["figure"]).is_file()
    assert str(result["figure"]).replace("\\", "/").startswith("results/fast/")
    assert not Path("report").exists()
    # Changing only test labels must not change validation/model selection.
    y[130:] = 1 - y[130:]
    changed = boosting.run(fast=True, data_path="chosen.csv")
    assert changed["best_depth"] == result["best_depth"]
    assert changed["val_f1_by_depth"] == result["val_f1_by_depth"]
