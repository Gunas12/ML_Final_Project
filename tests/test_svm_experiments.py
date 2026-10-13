"""Tests for validation-only SVM model selection and experiment reporting."""

from pathlib import Path

import numpy as np
import pytest

from src.svm_experiments import (
    gamma_sweep,
    lambda_sweep,
    run,
    validate_splits,
)


def binary_f1(y_true, y_pred):
    """Small test-local F1 helper for the positive label 1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_positive = np.count_nonzero((y_true == 1) & (y_pred == 1))
    false_positive = np.count_nonzero((y_true == 0) & (y_pred == 1))
    false_negative = np.count_nonzero((y_true == 1) & (y_pred == 0))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def make_xor_splits():
    rng = np.random.default_rng(314)
    centers = np.array(
        [[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]]
    )
    labels = np.array([0, 0, 1, 1])
    X = np.vstack(
        [rng.normal(center, 0.16, size=(60, 2)) for center in centers]
    )
    y = np.repeat(labels, 60)
    order = rng.permutation(y.size)
    X, y = X[order], y[order]
    return X[:144], y[:144], X[144:200], y[144:200], X[200:], y[200:]


def test_validate_splits_accepts_valid_arrays_without_modifying_them():
    splits = make_xor_splits()
    originals = tuple(array.copy() for array in splits)

    validated = validate_splits(splits)

    assert len(validated) == 6
    assert all(isinstance(array, np.ndarray) for array in validated)
    for original, current in zip(originals, splits):
        np.testing.assert_array_equal(current, original)


def test_validate_splits_rejects_wrong_shape_size_and_non_finite_data():
    valid = list(make_xor_splits())
    invalid_cases = []

    wrong_X_dimension = valid.copy()
    wrong_X_dimension[0] = wrong_X_dimension[0][:, 0]
    invalid_cases.append(wrong_X_dimension)

    wrong_y_dimension = valid.copy()
    wrong_y_dimension[1] = wrong_y_dimension[1][:, None]
    invalid_cases.append(wrong_y_dimension)

    wrong_sample_count = valid.copy()
    wrong_sample_count[3] = wrong_sample_count[3][:-1]
    invalid_cases.append(wrong_sample_count)

    wrong_feature_count = valid.copy()
    wrong_feature_count[4] = np.ones((wrong_feature_count[4].shape[0], 3))
    invalid_cases.append(wrong_feature_count)

    empty_split = valid.copy()
    empty_split[4] = np.empty((0, 2))
    empty_split[5] = np.array([], dtype=int)
    invalid_cases.append(empty_split)

    non_finite_X = valid.copy()
    non_finite_X[0] = non_finite_X[0].copy()
    non_finite_X[0][0, 0] = np.nan
    invalid_cases.append(non_finite_X)

    non_finite_y = valid.copy()
    non_finite_y[1] = non_finite_y[1].astype(float)
    non_finite_y[1][0] = np.inf
    invalid_cases.append(non_finite_y)

    invalid_cases.append(valid[:-1])

    for invalid in invalid_cases:
        with pytest.raises(ValueError):
            validate_splits(invalid)


def test_lambda_sweep_runs_every_value_and_selects_by_validation_score():
    X_train, y_train, X_validation, y_validation, _, _ = make_xor_splits()
    lambda_values = [1e-3, 1e-2, 1e-1]

    results, best = lambda_sweep(
        X_train,
        y_train,
        X_validation,
        y_validation,
        lambda_values,
        n_iters=800,
        random_state=9,
        score_fn=binary_f1,
    )

    assert [result["lambda"] for result in results] == lambda_values
    expected = max(
        results, key=lambda result: (result["validation_score"], result["lambda"])
    )
    assert best is expected
    assert best["lambda"] == expected["lambda"]
    assert all(result["model"] is not None for result in results)


def test_gamma_sweep_reuses_each_training_mapping_for_validation():
    X_train, y_train, X_validation, y_validation, _, _ = make_xor_splits()

    results, best = gamma_sweep(
        X_train,
        y_train,
        X_validation,
        y_validation,
        gamma_values=[0.1, 1.0],
        n_components=48,
        lambda_=0.1,
        n_iters=1_200,
        random_state=13,
        score_fn=binary_f1,
    )

    assert len(results) == 2
    for result in results:
        rff = result["rff"]
        W_before = rff.W_.copy()
        offset_before = rff.offset_.copy()
        validation_predictions = result["model"].predict(
            rff.transform(X_validation)
        )
        assert result["validation_score"] == pytest.approx(
            binary_f1(y_validation, validation_predictions)
        )
        np.testing.assert_array_equal(rff.W_, W_before)
        np.testing.assert_array_equal(rff.offset_, offset_before)

    expected = max(
        results, key=lambda result: (result["validation_score"], -result["gamma"])
    )
    assert best is expected


def test_run_uses_test_labels_only_for_the_final_score(tmp_path):
    splits = make_xor_splits()
    score_input_sizes = []

    def tracking_score(y_true, y_pred):
        score_input_sizes.append(len(y_true))
        return binary_f1(y_true, y_pred)

    run(
        splits,
        out_dir=tmp_path,
        score_fn=tracking_score,
        fast=True,
        random_state=21,
    )

    test_size = len(splits[5])
    assert score_input_sizes.count(test_size) == 1
    assert score_input_sizes[-1] == test_size


def test_fast_run_is_deterministic_creates_figures_and_preserves_inputs(tmp_path):
    splits = make_xor_splits()
    originals = tuple(array.copy() for array in splits)

    first = run(
        splits,
        out_dir=tmp_path / "first",
        score_fn=binary_f1,
        fast=True,
        random_state=42,
    )
    second = run(
        splits,
        out_dir=tmp_path / "second",
        score_fn=binary_f1,
        fast=True,
        random_state=42,
    )

    deterministic_keys = (
        "selected_model_type",
        "best_lambda",
        "best_gamma",
        "n_components",
        "validation_score",
        "test_score",
        "final_loss",
        "weight_norm",
        "margin_width",
        "support_count",
    )
    for key in deterministic_keys:
        assert first[key] == second[key]

    required_keys = {
        "selected_model_type",
        "best_lambda",
        "best_gamma",
        "n_components",
        "validation_score",
        "test_score",
        "final_loss",
        "weight_norm",
        "margin_width",
        "support_count",
        "fit_time_seconds",
        "figure_paths",
        "lambda_results",
        "gamma_results",
    }
    assert required_keys <= first.keys()
    assert first["n_components"] == 64
    assert len(first["lambda_results"]) == 3
    assert len(first["gamma_results"]) == 2

    expected_names = {
        "fig_svm_convergence.png",
        "fig_svm_lambda_sweep.png",
        "fig_svm_gamma_sweep.png",
    }
    figure_paths = [Path(path) for path in first["figure_paths"].values()]
    assert {path.name for path in figure_paths} == expected_names
    assert all(path.is_file() and path.stat().st_size > 0 for path in figure_paths)

    for original, current in zip(originals, splits):
        np.testing.assert_array_equal(current, original)
