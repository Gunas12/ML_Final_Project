"""Unit tests for the from-scratch Random Forest model."""

import numpy as np

from src.bonus_ensembles import RandomForest


def _regression_data(n=400, p=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    w = np.array([2.0, -1.0, 0.5, 0.0, 3.0])
    y = X @ w + 1.5 + 0.05 * rng.normal(size=n)
    return X, y


def _classification_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + X[:, 1] ** 2 > 1.0).astype(int)
    return X, y


def test_forest_classification_is_reproducible():
    X, y = _classification_data()
    X_train, X_test = X[:400], X[400:]
    y_train = y[:400]

    forest = RandomForest(
        "classification", 25, random_state=1
    ).fit(X_train, y_train)

    again = RandomForest(
        "classification", 25, random_state=1
    ).fit(X_train, y_train)

    assert np.array_equal(
        forest.predict(X_test), again.predict(X_test)
    )


def test_forest_classification_probabilities_sum_to_one():
    X, y = _classification_data()
    forest = RandomForest(
        "classification", 25, random_state=1
    ).fit(X[:400], y[:400])

    probabilities = forest.predict_proba(X[400:])

    assert probabilities.shape == (200, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_forest_classification_oob_score():
    X, y = _classification_data()
    forest = RandomForest(
        "classification", 25, random_state=1
    ).fit(X[:400], y[:400])

    assert 0.5 < forest.oob_score_ <= 1.0


def test_forest_regression_and_staged_outputs():
    X, y = _regression_data()
    forest = RandomForest(
        "regression", 10, min_samples_leaf=2
    ).fit(X[:300], y[:300])

    stages = forest.staged_outputs(X[300:])

    assert len(stages) == 10
    assert np.allclose(stages[-1], forest.predict(X[300:]))
    assert forest.oob_score_ > 0.5


def test_forest_regression_predictions_have_correct_shape():
    X, y = _regression_data()
    forest = RandomForest(
        "regression", 10, random_state=42
    ).fit(X[:300], y[:300])

    predictions = forest.predict(X[300:])

    assert predictions.shape == (100,)
    assert np.all(np.isfinite(predictions))
