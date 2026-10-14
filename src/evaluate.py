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


def timeit(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - start
