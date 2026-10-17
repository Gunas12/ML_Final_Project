"""Tests for src/evaluate.py.

Every metric in evaluate.py is our own NumPy code. scikit-learn is imported
here ONLY as a reference oracle: each test checks that our number equals
sklearn's on small, seeded examples.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm

from src import evaluate as ev

RNG = np.random.default_rng(42)


# --------------------------------------------------------------------------
# regression metrics
# --------------------------------------------------------------------------
@pytest.fixture
def regression_pair():
    y_true = RNG.normal(200_000, 80_000, size=500)
    y_pred = y_true + RNG.normal(0, 30_000, size=500)
    return y_true, y_pred


def test_rmse_matches_sklearn(regression_pair):
    y_true, y_pred = regression_pair
    expected = np.sqrt(skm.mean_squared_error(y_true, y_pred))
    assert ev.rmse(y_true, y_pred) == pytest.approx(expected)


def test_mae_matches_sklearn(regression_pair):
    y_true, y_pred = regression_pair
    assert ev.mae(y_true, y_pred) == pytest.approx(
        skm.mean_absolute_error(y_true, y_pred)
    )


def test_r2_matches_sklearn(regression_pair):
    y_true, y_pred = regression_pair
    assert ev.r2(y_true, y_pred) == pytest.approx(skm.r2_score(y_true, y_pred))


def test_perfect_regression_scores():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert ev.rmse(y, y) == 0.0
    assert ev.mae(y, y) == 0.0
    assert ev.r2(y, y) == 1.0


def test_r2_of_mean_predictor_is_zero():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    assert ev.r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_r2_constant_target_is_defined():
    y = np.full(5, 7.0)
    assert ev.r2(y, y) == 1.0
    assert ev.r2(y, y + 1.0) == 0.0


@pytest.mark.parametrize("metric", [ev.rmse, ev.mae, ev.r2])
def test_regression_metrics_reject_bad_shapes(metric):
    with pytest.raises(ValueError):
        metric([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        metric([], [])


# --------------------------------------------------------------------------
# classification metrics
# --------------------------------------------------------------------------
@pytest.fixture
def classification_triple():
    y_true = RNG.integers(0, 2, size=400)
    scores = y_true * 0.8 + RNG.normal(0, 0.7, size=400)
    y_pred = (scores > 0.4).astype(int)
    return y_true, y_pred, scores


def test_confusion_matrix_matches_sklearn(classification_triple):
    y_true, y_pred, _ = classification_triple
    np.testing.assert_array_equal(
        ev.confusion_matrix(y_true, y_pred),
        skm.confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )


def test_precision_recall_f1_match_sklearn(classification_triple):
    y_true, y_pred, _ = classification_triple
    precision, recall, f1 = ev.precision_recall_f1(y_true, y_pred)
    assert precision == pytest.approx(skm.precision_score(y_true, y_pred))
    assert recall == pytest.approx(skm.recall_score(y_true, y_pred))
    assert f1 == pytest.approx(skm.f1_score(y_true, y_pred))


def test_accuracy_matches_sklearn(classification_triple):
    y_true, y_pred, _ = classification_triple
    assert ev.accuracy(y_true, y_pred) == pytest.approx(
        skm.accuracy_score(y_true, y_pred)
    )


def test_roc_auc_matches_sklearn(classification_triple):
    y_true, _, scores = classification_triple
    assert ev.roc_auc(y_true, scores) == pytest.approx(
        skm.roc_auc_score(y_true, scores)
    )


def test_roc_auc_handles_tied_scores():
    # Heavy ties (like a shallow tree's leaf probabilities) need mid-ranks.
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    scores = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.5, 0.5, 0.1, 0.9, 0.5])
    assert ev.roc_auc(y_true, scores) == pytest.approx(
        skm.roc_auc_score(y_true, scores)
    )


def test_roc_auc_extremes():
    y = np.array([0, 0, 1, 1])
    assert ev.roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ev.roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert ev.roc_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_roc_auc_requires_both_classes():
    with pytest.raises(ValueError):
        ev.roc_auc([1, 1, 1], [0.2, 0.3, 0.4])


def test_roc_auc_rejects_nan_scores():
    with pytest.raises(ValueError):
        ev.roc_auc([0, 1, 0, 1], [0.1, np.nan, 0.3, 0.9])


def test_precision_is_zero_when_nothing_predicted_positive():
    precision, recall, f1 = ev.precision_recall_f1([0, 1, 1, 0], [0, 0, 0, 0])
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# compare / format_table / timeit
# --------------------------------------------------------------------------
def test_format_table_contains_models_and_metrics():
    table = ev.format_table(
        {"ours": {"RMSE": 1.5, "R2": 0.5}, "sklearn": {"RMSE": 1.5, "R2": 0.5}}
    )
    for token in ("Model", "RMSE", "R2", "ours", "sklearn"):
        assert token in table


def test_compare_prints_table(capsys):
    ev.compare({"ours": {"F1": 0.9}})
    assert "ours" in capsys.readouterr().out


def test_compare_rejects_empty():
    with pytest.raises(ValueError):
        ev.compare({})


def test_timeit_returns_result_and_nonnegative_seconds():
    result, seconds = ev.timeit(sum, [1, 2, 3])
    assert result == 6
    assert seconds >= 0.0


# --------------------------------------------------------------------------
# PR-AUC (average precision) and ROC curve
# --------------------------------------------------------------------------
def test_average_precision_matches_sklearn(classification_triple):
    y_true, _, scores = classification_triple
    assert ev.average_precision(y_true, scores) == pytest.approx(
        skm.average_precision_score(y_true, scores)
    )


def test_average_precision_handles_ties():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    scores = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.5, 0.5, 0.1, 0.9, 0.5])
    assert ev.average_precision(y_true, scores) == pytest.approx(
        skm.average_precision_score(y_true, scores)
    )


def test_average_precision_perfect_ranking_is_one():
    assert ev.average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_roc_curve_matches_sklearn(classification_triple):
    y_true, _, scores = classification_triple
    fpr, tpr, _ = ev.roc_curve(y_true, scores)
    sk_fpr, sk_tpr, _ = skm.roc_curve(y_true, scores, drop_intermediate=False)
    np.testing.assert_allclose(fpr, sk_fpr)
    np.testing.assert_allclose(tpr, sk_tpr)


def test_area_under_roc_curve_equals_roc_auc(classification_triple):
    y_true, _, scores = classification_triple
    fpr, tpr, _ = ev.roc_curve(y_true, scores)
    area = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    assert area == pytest.approx(ev.roc_auc(y_true, scores))


# --------------------------------------------------------------------------
# K-fold index generators
# --------------------------------------------------------------------------
@pytest.mark.parametrize("n,k", [(10, 5), (103, 5), (7, 3), (20, 20)])
def test_kfold_covers_every_sample_exactly_once(n, k):
    folds = ev.kfold_indices(n, k, seed=42)
    assert len(folds) == k
    all_val = np.concatenate([val for _, val in folds])
    np.testing.assert_array_equal(np.sort(all_val), np.arange(n))


def test_kfold_train_and_val_are_disjoint_and_complete():
    for train, val in ev.kfold_indices(57, 4, seed=1):
        assert np.intersect1d(train, val).size == 0
        assert train.size + val.size == 57


def test_kfold_fold_sizes_differ_by_at_most_one():
    sizes = [val.size for _, val in ev.kfold_indices(103, 5)]
    assert max(sizes) - min(sizes) <= 1


def test_kfold_is_deterministic_for_a_seed_and_changes_with_seed():
    a = ev.kfold_indices(50, 5, seed=42)
    b = ev.kfold_indices(50, 5, seed=42)
    c = ev.kfold_indices(50, 5, seed=7)
    for (_, va), (_, vb) in zip(a, b):
        np.testing.assert_array_equal(va, vb)
    assert any(not np.array_equal(va, vc) for (_, va), (_, vc) in zip(a, c))


def test_kfold_rejects_bad_k():
    with pytest.raises(ValueError):
        ev.kfold_indices(10, 1)
    with pytest.raises(ValueError):
        ev.kfold_indices(3, 5)


def test_stratified_kfold_covers_every_sample_once_and_keeps_ratio():
    y = np.array([0] * 80 + [1] * 20)
    folds = ev.stratified_kfold_indices(y, 5, seed=42)
    all_val = np.concatenate([val for _, val in folds])
    np.testing.assert_array_equal(np.sort(all_val), np.arange(100))
    for _, val in folds:
        assert y[val].mean() == pytest.approx(0.2)


# --------------------------------------------------------------------------
# scoring bundles and cross_validate
# --------------------------------------------------------------------------
class _MeanRegressor:
    """Trivial model that records how much data it was trained on."""

    seen_train_sizes: list = []

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        _MeanRegressor.seen_train_sizes.append(len(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class _ThresholdClassifier:
    """Classifies by the sign of the first feature; exposes decision_function."""

    def fit(self, X, y):
        return self

    def decision_function(self, X):
        return X[:, 0]

    def predict(self, X):
        return (X[:, 0] > 0).astype(int)


def test_score_regression_has_log_metrics():
    y = np.array([100_000.0, 200_000.0, 400_000.0])
    scores = ev.score_regression(y, y * 1.1)
    assert set(scores) == {"RMSE", "MAE", "R2", "RMSE_log", "R2_log"}
    assert scores["RMSE_log"] == pytest.approx(np.log(1.1))


def test_score_regression_survives_nonpositive_predictions():
    scores = ev.score_regression(np.array([10.0, 20.0]), np.array([-5.0, 0.0]))
    assert np.isfinite(scores["RMSE_log"])


def test_score_classification_matches_sklearn(classification_triple):
    y_true, y_pred, scores = classification_triple
    out = ev.score_classification(y_true, y_pred, scores)
    assert out["ROC-AUC"] == pytest.approx(skm.roc_auc_score(y_true, scores))
    assert out["PR-AUC"] == pytest.approx(
        skm.average_precision_score(y_true, scores)
    )
    assert out["F1"] == pytest.approx(skm.f1_score(y_true, y_pred))


def test_mean_std_uses_sample_std():
    mean, std = ev.mean_std([{"a": 1.0}, {"a": 3.0}])
    assert mean["a"] == pytest.approx(2.0)
    assert std["a"] == pytest.approx(np.std([1.0, 3.0], ddof=1))


def test_cross_validate_regression_structure_and_training_size():
    X = RNG.normal(size=(100, 3))
    y = X[:, 0] * 10 + 50
    _MeanRegressor.seen_train_sizes = []
    result = ev.cross_validate(_MeanRegressor, X, y, k=5, task="regression")
    assert result["k"] == 5 and len(result["folds"]) == 5
    assert _MeanRegressor.seen_train_sizes == [80] * 5   # 4/5 of the rows
    for name in ("RMSE", "MAE", "R2"):
        assert name in result["mean"] and name in result["std"]
    assert result["fit_seconds"] >= 0 and result["predict_seconds"] >= 0


def test_cross_validate_is_deterministic():
    X = RNG.normal(size=(60, 2))
    y = np.abs(X[:, 0]) + 5
    a = ev.cross_validate(_MeanRegressor, X, y, k=4, seed=42)
    b = ev.cross_validate(_MeanRegressor, X, y, k=4, seed=42)
    assert a["mean"] == b["mean"] and a["std"] == b["std"]


def test_cross_validate_classification_uses_scores_and_stratifies():
    X = RNG.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    result = ev.cross_validate(
        _ThresholdClassifier, X, y, k=5, task="classification"
    )
    assert result["mean"]["Accuracy"] == pytest.approx(1.0)
    assert result["mean"]["ROC-AUC"] == pytest.approx(1.0)
    assert result["std"]["F1"] == pytest.approx(0.0)


def test_cross_validate_preprocess_sees_training_rows_only():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.arange(20, dtype=float) + 1
    calls = []

    def preprocess(train, val):
        calls.append((len(train), len(val)))
        mean = train.mean(axis=0)            # statistic from TRAIN only
        return train - mean, val - mean

    ev.cross_validate(_MeanRegressor, X, y, k=4, preprocess=preprocess)
    assert calls == [(15, 5)] * 4


def test_cross_validate_accepts_dataframes():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(RNG.normal(size=(50, 2)), columns=["a", "b"])
    y = np.abs(X["a"].to_numpy()) + 5
    assert "RMSE" in ev.cross_validate(_MeanRegressor, X, y, k=5)["mean"]


def test_cross_validate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        ev.cross_validate(_MeanRegressor, np.zeros((5, 1)), np.zeros(4))
    with pytest.raises(ValueError):
        ev.cross_validate(_MeanRegressor, np.zeros((5, 1)), np.zeros(5), task="x")
