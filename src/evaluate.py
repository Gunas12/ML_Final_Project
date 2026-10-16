from __future__ import annotations

from time import perf_counter

import numpy as np


def _paired_vectors(y_true, y_pred):
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    return actual, predicted


def rmse(y_true, y_pred) -> float:
    actual, predicted = _paired_vectors(y_true, y_pred)
    difference = actual.astype(float) - predicted.astype(float)
    return float(np.sqrt(np.mean(difference**2)))


def mae(y_true, y_pred) -> float:
    actual, predicted = _paired_vectors(y_true, y_pred)
    return float(np.mean(np.abs(actual.astype(float) - predicted.astype(float))))


def r2(y_true, y_pred) -> float:
    actual, predicted = _paired_vectors(y_true, y_pred)
    actual = actual.astype(float)
    predicted = predicted.astype(float)
    residual_sum = float(np.sum((actual - predicted) ** 2))
    total_sum = float(np.sum((actual - actual.mean()) ** 2))
    if total_sum == 0.0:
        return 1.0 if residual_sum == 0.0 else 0.0
    return float(1.0 - residual_sum / total_sum)


def _binary_vectors(y_true, y_pred):
    actual, predicted = _paired_vectors(y_true, y_pred)
    labels = np.unique(np.concatenate((actual, predicted)))
    if labels.size > 2:
        raise ValueError("classification metrics require binary labels")
    if labels.size == 1:
        positive = 1 if labels[0] in (0, 1) else labels[0]
    else:
        positive = 1 if 1 in labels else labels[-1]
    return actual == positive, predicted == positive


def confusion_matrix(y_true, y_pred):
    actual_positive, predicted_positive = _binary_vectors(y_true, y_pred)
    true_negative = np.count_nonzero(~actual_positive & ~predicted_positive)
    false_positive = np.count_nonzero(~actual_positive & predicted_positive)
    false_negative = np.count_nonzero(actual_positive & ~predicted_positive)
    true_positive = np.count_nonzero(actual_positive & predicted_positive)
    return np.array(
        [[true_negative, false_positive], [false_negative, true_positive]],
        dtype=int,
    )


def precision_recall_f1(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    true_positive = int(matrix[1, 1])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def accuracy(y_true, y_pred) -> float:
    actual, predicted = _paired_vectors(y_true, y_pred)
    return float(np.mean(actual == predicted))


def roc_auc(y_true, scores) -> float:
    actual = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if actual.ndim != 1 or values.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if actual.size == 0 or actual.size != values.size:
        raise ValueError("y_true and scores must have the same non-zero length")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must not contain NaN or infinite values")
    labels = np.unique(actual)
    if labels.size != 2:
        raise ValueError("ROC-AUC requires both classes to be present")
    positive = 1 if 1 in labels else labels[-1]
    is_positive = actual == positive
    positive_count = int(is_positive.sum())
    negative_count = int(actual.size - positive_count)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[is_positive].sum())
    return float(
        (rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _binary_scores(y_true, scores):
    """Validate (y_true, scores) and return (is_positive, scores) arrays."""
    actual = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    if actual.ndim != 1 or values.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if actual.size == 0 or actual.size != values.size:
        raise ValueError("y_true and scores must have the same non-zero length")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must not contain NaN or infinite values")
    labels = np.unique(actual)
    if labels.size != 2:
        raise ValueError("both classes must be present")
    positive = 1 if 1 in labels else labels[-1]
    return actual == positive, values


def _cumulative_counts(is_positive, values):
    """True/false positive counts at every DISTINCT score threshold.

    Samples are sorted by descending score; a threshold may only sit between
    two different score values, so tied scores are never split.
    """
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    sorted_positive = is_positive[order]
    distinct = np.where(np.diff(sorted_values))[0]
    last = np.r_[distinct, sorted_values.size - 1]
    true_positives = np.cumsum(sorted_positive)[last].astype(float)
    false_positives = (1 + last) - true_positives
    return true_positives, false_positives, sorted_values[last]


def roc_curve(y_true, scores):
    """Return (fpr, tpr, thresholds) of the ROC curve, starting at (0, 0)."""
    is_positive, values = _binary_scores(y_true, scores)
    tp, fp, thresholds = _cumulative_counts(is_positive, values)
    tpr = np.r_[0.0, tp / tp[-1]]
    fpr = np.r_[0.0, fp / fp[-1]]
    return fpr, tpr, np.r_[np.inf, thresholds]


def average_precision(y_true, scores) -> float:
    """PR-AUC as average precision: AP = sum_n (R_n - R_{n-1}) * P_n.

    This is the step-wise (non-interpolated) area under the precision-recall
    curve, the same definition scikit-learn uses, and is the more informative
    summary when the positive class is rare.
    """
    is_positive, values = _binary_scores(y_true, scores)
    tp, fp, _ = _cumulative_counts(is_positive, values)
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def compare(name_to_metrics: dict) -> None:
    if not name_to_metrics:
        raise ValueError("name_to_metrics must not be empty")
    metric_names = []
    for metrics in name_to_metrics.values():
        for name in metrics:
            if name not in metric_names:
                metric_names.append(name)
    row_name_width = max(5, max(len(str(name)) for name in name_to_metrics))
    metric_widths = {
        name: max(len(name), 12) for name in metric_names
    }
    heading = f"{'Model':<{row_name_width}}"
    for metric in metric_names:
        heading += f"  {metric:>{metric_widths[metric]}}"
    print(heading)
    print("-" * len(heading))
    for model_name, metrics in name_to_metrics.items():
        row = f"{str(model_name):<{row_name_width}}"
        for metric in metric_names:
            value = metrics.get(metric)
            if value is None or not np.isfinite(value):
                formatted = "n/a"
            elif isinstance(value, (int, np.integer)):
                formatted = str(int(value))
            else:
                formatted = f"{float(value):.5g}"
            row += f"  {formatted:>{metric_widths[metric]}}"
        print(row)


def format_table(name_to_metrics: dict) -> str:
    if not name_to_metrics:
        raise ValueError("name_to_metrics must not be empty")
    metric_names = []
    for metrics in name_to_metrics.values():
        for name in metrics:
            if name not in metric_names:
                metric_names.append(name)
    row_name_width = max(5, max(len(str(name)) for name in name_to_metrics))
    metric_widths = {name: max(len(name), 12) for name in metric_names}
    lines = [
        f"{'Model':<{row_name_width}}"
        + "".join(f"  {metric:>{metric_widths[metric]}}" for metric in metric_names)
    ]
    lines.append("-" * len(lines[0]))
    for model_name, metrics in name_to_metrics.items():
        cells = []
        for metric in metric_names:
            value = metrics.get(metric)
            if value is None or not np.isfinite(value):
                formatted = "n/a"
            elif isinstance(value, (int, np.integer)):
                formatted = str(int(value))
            else:
                formatted = f"{float(value):.5g}"
            cells.append(f"  {formatted:>{metric_widths[metric]}}")
        lines.append(f"{str(model_name):<{row_name_width}}" + "".join(cells))
    return "\n".join(lines)


def kfold_indices(n: int, k: int = 5, seed: int = 42):
    """Shuffled K-fold split of range(n).

    Returns a list of k (train_idx, val_idx) pairs. Every index appears in
    exactly one validation fold; fold sizes differ by at most one.
    """
    if not 2 <= k <= n:
        raise ValueError("k must satisfy 2 <= k <= n")
    permutation = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(permutation, k)
    return [
        (np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
        for i in range(k)
    ]


def stratified_kfold_indices(y, k: int = 5, seed: int = 42):
    """K-fold split that keeps each class's share the same in every fold."""
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size < k or k < 2:
        raise ValueError("y must be 1-D with at least k samples and k >= 2")
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == cls))
        for i, part in enumerate(np.array_split(members, k)):
            folds[i].append(part)
    folds = [np.sort(np.concatenate(parts)) for parts in folds]
    return [
        (np.concatenate([folds[j] for j in range(k) if j != i]), folds[i])
        for i in range(k)
    ]


def score_regression(y_true, y_pred, log_metrics: bool = True) -> dict:
    """RMSE, MAE and R2 on the price scale (+ RMSE/R2 on log(price)).

    Prices are heavy-tailed, so the log-scale numbers show how the model does
    across the whole market, not just on a few luxury listings. Predictions
    are floored at 1 before taking the log so a bad prediction cannot crash it.
    """
    scores = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    if log_metrics:
        log_true = np.log(np.asarray(y_true, dtype=float))
        log_pred = np.log(np.maximum(np.asarray(y_pred, dtype=float), 1.0))
        scores["RMSE_log"] = rmse(log_true, log_pred)
        scores["R2_log"] = r2(log_true, log_pred)
    return scores


def score_classification(y_true, y_pred, scores) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC and PR-AUC in one dict."""
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    return {
        "Accuracy": accuracy(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc(y_true, scores),
        "PR-AUC": average_precision(y_true, scores),
    }


def positive_scores(model, X) -> np.ndarray:
    """Ranking score for the positive class, whatever the model offers.

    predict_proba (trees) -> P(class 1); decision_function (SVMs) -> margin;
    otherwise the hard prediction. ROC/PR-AUC only need the ordering.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def mean_std(fold_metrics: list) -> tuple:
    """Per-metric (mean, std) across folds; std is the sample std (ddof=1)."""
    names = list(fold_metrics[0])
    table = np.array([[fold[name] for name in names] for fold in fold_metrics])
    ddof = 1 if len(fold_metrics) > 1 else 0
    return (
        {name: float(table[:, j].mean()) for j, name in enumerate(names)},
        {name: float(table[:, j].std(ddof=ddof)) for j, name in enumerate(names)},
    )


def cross_validate(
    model_factory,
    X,
    y,
    k: int = 5,
    seed: int = 42,
    task: str = "regression",
    preprocess=None,
    log_metrics: bool = True,
) -> dict:
    """K-fold cross-validation with mean +/- std of every metric.

    model_factory : zero-argument callable returning a FRESH model with
                    fit(X, y) and predict(X) (our models and sklearn ones).
    preprocess    : optional callable (X_train, X_val) -> (X_train', X_val').
                    It is called once per fold with that fold's training rows
                    only, so scalers never see validation data (no leakage).
    task          : "regression" (plain folds) or "classification"
                    (stratified folds on y).

    Returns {"folds", "mean", "std", "fit_seconds", "predict_seconds", "k"}.
    Pass only train/validation rows here, never the held-out test set.
    """
    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or 'classification'")
    features = np.asarray(X, dtype=float)
    targets = np.asarray(y)
    if features.shape[0] != targets.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if task == "classification":
        splits = stratified_kfold_indices(targets, k, seed)
    else:
        splits = kfold_indices(len(targets), k, seed)

    folds, fit_times, predict_times = [], [], []
    for train_idx, val_idx in splits:
        X_train, X_val = features[train_idx], features[val_idx]
        if preprocess is not None:
            X_train, X_val = preprocess(X_train, X_val)
        model = model_factory()
        _, fit_seconds = timeit(model.fit, X_train, targets[train_idx])
        predictions, predict_seconds = timeit(model.predict, X_val)
        if task == "regression":
            fold = score_regression(targets[val_idx], predictions, log_metrics)
        else:
            fold = score_classification(
                targets[val_idx], predictions, positive_scores(model, X_val)
            )
        folds.append(fold)
        fit_times.append(fit_seconds)
        predict_times.append(predict_seconds)

    mean, std = mean_std(folds)
    return {
        "k": k,
        "folds": folds,
        "mean": mean,
        "std": std,
        "fit_seconds": float(np.mean(fit_times)),
        "predict_seconds": float(np.mean(predict_times)),
    }


def timeit(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - start
