"""Reproducible linear and RFF Pegasos SVM experiment helpers.

Inputs are expected to be standardized before they reach this module.  Model
selection uses training and validation data only; the test split is evaluated
once, after the winning configuration has been selected.
"""

from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd

from .svm import PegasosSVM, RandomFourierFeatures

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _has_non_finite_values(values) -> bool:
    """Return whether numeric/object labels contain missing or infinite values."""
    values = np.asarray(values, dtype=object)
    for value in values.ravel():
        if value is None or pd.isna(value):
            return True
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            return True
    return False


def validate_splits(splits):
    """Validate and return train/validation/test arrays in canonical order."""
    try:
        if len(splits) != 6:
            raise ValueError("splits must contain exactly six arrays")
    except TypeError as exc:
        raise ValueError("splits must contain exactly six arrays") from exc

    validated = []
    feature_count = None
    split_names = ("train", "validation", "test")

    for index, split_name in enumerate(split_names):
        try:
            X = np.asarray(splits[2 * index], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"X_{split_name} must contain numeric values") from exc
        y = np.asarray(splits[2 * index + 1])

        if X.ndim != 2:
            raise ValueError(f"X_{split_name} must be two-dimensional")
        if y.ndim != 1:
            raise ValueError(f"y_{split_name} must be one-dimensional")
        if X.shape[0] == 0 or X.shape[1] == 0 or y.size == 0:
            raise ValueError(f"{split_name} split must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X_{split_name} and y_{split_name} must have the same sample count"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError(f"X_{split_name} must not contain NaN or infinity")
        if _has_non_finite_values(y):
            raise ValueError(f"y_{split_name} must not contain NaN or infinity")

        if feature_count is None:
            feature_count = X.shape[1]
        elif X.shape[1] != feature_count:
            raise ValueError("all feature matrices must have the same feature count")

        validated.extend((X, y))

    return tuple(validated)


def lambda_sweep(
    X_train,
    y_train,
    X_validation,
    y_validation,
    lambda_values,
    n_iters,
    random_state,
    score_fn,
):
    """Evaluate linear Pegasos models and return all results plus the winner."""
    if not callable(score_fn):
        raise ValueError("score_fn must be callable")
    if len(lambda_values) == 0:
        raise ValueError("lambda_values must not be empty")

    results = []
    for lambda_ in lambda_values:
        model = PegasosSVM(
            lambda_=lambda_, n_iters=n_iters, random_state=random_state
        )
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_time = perf_counter() - start

        train_predictions = model.predict(X_train)
        validation_predictions = model.predict(X_validation)
        results.append(
            {
                "lambda": float(lambda_),
                "validation_score": float(
                    score_fn(y_validation, validation_predictions)
                ),
                "train_score": float(score_fn(y_train, train_predictions)),
                "final_loss": float(model.loss_history_[-1]),
                "weight_norm": model.weight_norm(),
                "margin_width": model.margin_width(),
                "support_count": int(np.count_nonzero(model.support_mask(X_train, y_train))),
                "fit_time_seconds": float(fit_time),
                "model": model,
            }
        )

    best_result = max(
        results, key=lambda result: (result["validation_score"], result["lambda"])
    )
    return results, best_result


def gamma_sweep(
    X_train,
    y_train,
    X_validation,
    y_validation,
    gamma_values,
    n_components,
    lambda_,
    n_iters,
    random_state,
    score_fn,
):
    """Evaluate RFF Pegasos models and return all results plus the winner."""
    if not callable(score_fn):
        raise ValueError("score_fn must be callable")
    if len(gamma_values) == 0:
        raise ValueError("gamma_values must not be empty")

    results = []
    for gamma in gamma_values:
        rff = RandomFourierFeatures(
            n_components=n_components,
            gamma=gamma,
            random_state=random_state,
        )
        model = PegasosSVM(
            lambda_=lambda_, n_iters=n_iters, random_state=random_state
        )

        start = perf_counter()
        X_train_rff = rff.fit_transform(X_train)
        model.fit(X_train_rff, y_train)
        fit_time = perf_counter() - start
        X_validation_rff = rff.transform(X_validation)

        train_predictions = model.predict(X_train_rff)
        validation_predictions = model.predict(X_validation_rff)
        results.append(
            {
                "gamma": float(gamma),
                "validation_score": float(
                    score_fn(y_validation, validation_predictions)
                ),
                "train_score": float(score_fn(y_train, train_predictions)),
                "final_loss": float(model.loss_history_[-1]),
                "weight_norm": model.weight_norm(),
                "margin_width": model.margin_width(),
                "support_count": int(
                    np.count_nonzero(model.support_mask(X_train_rff, y_train))
                ),
                "fit_time_seconds": float(fit_time),
                "rff": rff,
                "model": model,
            }
        )

    best_result = max(
        results,
        key=lambda result: (result["validation_score"], -result["gamma"]),
    )
    return results, best_result


def _prepare_output_path(output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_convergence(model, output_path):
    """Save the selected model's objective-loss trajectory as a PNG."""
    path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(model.loss_steps_, model.loss_history_, color="tab:blue")
    axis.set_xlabel("Pegasos iteration")
    axis.set_ylabel("Objective loss")
    axis.set_title("Pegasos convergence")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=150)
    plt.close(figure)
    return str(path)


def plot_lambda_results(results, output_path):
    """Save validation-score and margin-width curves for the lambda sweep."""
    path = _prepare_output_path(output_path)
    ordered = sorted(results, key=lambda result: result["lambda"])
    lambdas = [result["lambda"] for result in ordered]
    scores = [result["validation_score"] for result in ordered]
    margins = [result["margin_width"] for result in ordered]

    figure, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(lambdas, scores, marker="o", label="Validation score")
    axes[0].set_ylabel("Validation score")
    axes[0].set_title("Linear Pegasos lambda sweep")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(lambdas, margins, marker="o", color="tab:orange", label="Margin width")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Lambda")
    axes[1].set_ylabel("Margin width")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=150)
    plt.close(figure)
    return str(path)


def plot_gamma_results(results, output_path):
    """Save validation score across the RFF gamma sweep."""
    path = _prepare_output_path(output_path)
    ordered = sorted(results, key=lambda result: result["gamma"])
    gammas = [result["gamma"] for result in ordered]
    scores = [result["validation_score"] for result in ordered]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(gammas, scores, marker="o", label="Validation score")
    axis.set_xscale("log")
    axis.set_xlabel("Gamma")
    axis.set_ylabel("Validation score")
    axis.set_title("RFF gamma sweep")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=150)
    plt.close(figure)
    return str(path)


def run(
    splits,
    out_dir="report/figures",
    score_fn=None,
    fast=False,
    random_state=42,
) -> dict:
    """Run validation-only model selection, then evaluate the test split once."""
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = validate_splits(splits)

    if score_fn is None:
        from .evaluate import precision_recall_f1

        def score_fn(y_true, y_pred):
            return precision_recall_f1(y_true, y_pred)[2]

    if not callable(score_fn):
        raise ValueError("score_fn must be callable")

    if fast:
        lambda_values = [1e-3, 1e-2, 1e-1]
        gamma_values = [0.1, 1.0]
        n_components = 64
        n_iters = 2_000
    else:
        lambda_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        gamma_values = [0.01, 0.1, 1.0, 10.0]
        n_components = 256
        n_iters = 20_000

    lambda_results, best_linear = lambda_sweep(
        X_train,
        y_train,
        X_validation,
        y_validation,
        lambda_values,
        n_iters,
        random_state,
        score_fn,
    )
    gamma_results, best_rff = gamma_sweep(
        X_train,
        y_train,
        X_validation,
        y_validation,
        gamma_values,
        n_components,
        best_linear["lambda"],
        n_iters,
        random_state,
        score_fn,
    )

    if best_rff["validation_score"] > best_linear["validation_score"]:
        selected_model_type = "rff"
        selected = best_rff
        X_test_selected = selected["rff"].transform(X_test)
        best_gamma = selected["gamma"]
    else:
        selected_model_type = "linear"
        selected = best_linear
        X_test_selected = X_test
        best_gamma = None

    test_predictions = selected["model"].predict(X_test_selected)
    test_score = float(score_fn(y_test, test_predictions))

    output_directory = Path(out_dir)
    figure_paths = {
        "convergence": plot_convergence(
            selected["model"], output_directory / "fig_svm_convergence.png"
        ),
        "lambda_sweep": plot_lambda_results(
            lambda_results, output_directory / "fig_svm_lambda_sweep.png"
        ),
        "gamma_sweep": plot_gamma_results(
            gamma_results, output_directory / "fig_svm_gamma_sweep.png"
        ),
    }

    return {
        "selected_model_type": selected_model_type,
        "best_lambda": best_linear["lambda"],
        "best_gamma": best_gamma,
        "n_components": n_components,
        "validation_score": selected["validation_score"],
        "test_score": test_score,
        "final_loss": selected["final_loss"],
        "weight_norm": selected["weight_norm"],
        "margin_width": selected["margin_width"],
        "support_count": selected["support_count"],
        "fit_time_seconds": selected["fit_time_seconds"],
        "figure_paths": figure_paths,
        "lambda_results": lambda_results,
        "gamma_results": gamma_results,
    }
