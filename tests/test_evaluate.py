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
