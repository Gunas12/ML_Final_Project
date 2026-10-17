"""Where and why do the models fail?

All numbers are computed on the TEST split, from predictions of models that
were already fitted and selected using train/validation only. Nothing here
refits a model or tunes anything, so it cannot leak test information back
into model selection.

Four views (each returns plain numbers and writes one figure):
  1. regression error by true-price quantile   -> luxury under-prediction
  2. regression error by district              -> sparse-district failure
  3. largest individual regression errors      -> concrete bad listings
  4. classification errors vs distance to the median threshold -> the model
     is only wrong on listings that are genuinely borderline
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import evaluate as ev

QUANTILE_EDGES = (0.0, 0.5, 0.75, 0.9, 0.99, 1.0)
DISTANCE_EDGES = (0.0, 0.1, 0.25, 0.5, 1.0, np.inf)   # |price - T| / T


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def region_labels(frame, prefix: str = "location=") -> np.ndarray:
    """District name per row, recovered from the one-hot `location=...` columns."""
    columns = [c for c in frame.columns if str(c).startswith(prefix)]
    if not columns:
        return np.full(len(frame), "Unknown", dtype=object)
    block = np.asarray(frame[columns], dtype=float)
    names = np.array([str(c)[len(prefix):] for c in columns], dtype=object)
    labels = names[block.argmax(axis=1)]
    labels[block.max(axis=1) <= 0] = "Unknown"
    return labels


def _bin_labels(edges) -> list:
    return [f"p{int(100 * lo)}-{int(100 * hi)}" for lo, hi in zip(edges[:-1], edges[1:])]


def error_by_price_quantile(y_true, y_pred, edges=QUANTILE_EDGES) -> list:
    """MAE, mean signed error (bias) and median relative error per price band."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    cuts = np.quantile(y_true, edges)
    rows = []
    for i, label in enumerate(_bin_labels(edges)):
        upper = y_true <= cuts[i + 1] if i == len(edges) - 2 else y_true < cuts[i + 1]
        mask = (y_true >= cuts[i]) & upper
        if not mask.any():
            continue
        residual = y_pred[mask] - y_true[mask]
        rows.append(
            {
                "band": label,
                "n": int(mask.sum()),
                "price_min": float(y_true[mask].min()),
                "price_max": float(y_true[mask].max()),
                "mae": float(np.abs(residual).mean()),
                "bias": float(residual.mean()),
                "median_rel_error": float(np.median(np.abs(residual) / y_true[mask])),
            }
        )
    return rows


def error_by_region(y_true, y_pred, regions, train_regions, min_test: int = 20) -> list:
    """Per-district test error alongside how many TRAINING listings the district has."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    regions = np.asarray(regions, dtype=object)
    train_names, train_counts = np.unique(np.asarray(train_regions, dtype=object), return_counts=True)
    train_n = dict(zip(train_names, train_counts))
    rows = []
    for name in np.unique(regions):
        mask = regions == name
        if mask.sum() < min_test:
            continue
        rel = np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]
        rows.append(
            {
                "region": str(name),
                "n_test": int(mask.sum()),
                "n_train": int(train_n.get(name, 0)),
                "mae": float(np.abs(y_pred[mask] - y_true[mask]).mean()),
                "median_rel_error": float(np.median(rel)),
            }
        )
    rows.sort(key=lambda r: r["median_rel_error"], reverse=True)
    return rows


def sparse_vs_dense(region_rows: list, sparse_below: int = 200) -> dict:
    """Pooled median relative error for rarely vs frequently seen districts."""
    def pooled(rows):
        if not rows:
            return {"regions": 0, "n_test": 0, "median_rel_error": float("nan")}
        weights = np.array([r["n_test"] for r in rows], dtype=float)
        values = np.array([r["median_rel_error"] for r in rows])
        return {
            "regions": len(rows),
            "n_test": int(weights.sum()),
            "median_rel_error": float(np.average(values, weights=weights)),
        }

    sparse = [r for r in region_rows if r["n_train"] < sparse_below]
    dense = [r for r in region_rows if r["n_train"] >= sparse_below]
    return {"threshold_train_rows": sparse_below, "sparse": pooled(sparse), "dense": pooled(dense)}


def largest_errors(y_true, y_pred, frame, k: int = 10) -> list:
    """The k worst test predictions with a few readable attributes."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    regions = region_labels(frame)
    order = np.argsort(-np.abs(y_pred - y_true))[:k]
    out = []
    for i in order:
        row = {
            "true_price": float(y_true[i]),
            "predicted": float(y_pred[i]),
            "abs_error": float(abs(y_pred[i] - y_true[i])),
            "region": str(regions[i]),
        }
        for column in ("area", "rooms"):
            if column in frame.columns:
                row[column] = float(np.asarray(frame[column], dtype=float)[i])
        out.append(row)
    return out


def classification_error_by_distance(y_price, y_tier, y_pred, threshold, edges=DISTANCE_EDGES) -> list:
    """Error rate by how far the true price sits from the tier threshold."""
    y_price = np.asarray(y_price, dtype=float)
    wrong = np.asarray(y_tier) != np.asarray(y_pred)
    distance = np.abs(y_price - threshold) / threshold
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (distance >= lo) & (distance < hi)
        if mask.any():
            hi_text = "inf" if np.isinf(hi) else f"{hi:.2f}"
            rows.append(
                {
                    "distance": f"{lo:.2f}-{hi_text}",
                    "n": int(mask.sum()),
                    "error_rate": float(wrong[mask].mean()),
                }
            )
    return rows


def top_features(values, names, k: int = 10, absolute: bool = False) -> list:
    """Top-k (name, value) pairs of an importance / weight vector."""
    values = np.asarray(values, dtype=float)
    ranking = np.abs(values) if absolute else values
    order = np.argsort(-ranking)[:k]
    return [(str(names[i]), float(values[i])) for i in order]


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _save(figure, out_dir, name) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _fig_quantiles(bands_by_model, out_dir) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [b["band"] for b in next(iter(bands_by_model.values()))]
    x = np.arange(len(labels))
    width = 0.8 / len(bands_by_model)
    for j, (name, bands) in enumerate(bands_by_model.items()):
        axes[0].bar(x + j * width, [b["mae"] for b in bands], width, label=name)
        axes[1].bar(x + j * width, [b["bias"] for b in bands], width, label=name)
    for axis, title, ylabel in (
        (axes[0], "Mean absolute error by true-price band", "MAE (AZN)"),
        (axes[1], "Mean signed error (prediction - truth)", "bias (AZN)"),
    ):
        axis.set_xticks(x + width * (len(bands_by_model) - 1) / 2)
        axis.set_xticklabels(labels)
        axis.set_xlabel("true-price percentile band")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.3, axis="y")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].legend()
    return _save(figure, out_dir, "fig_error_price_quantile.png")


def _fig_pred_vs_actual(y_true, y_pred, name, out_dir) -> str:
    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.scatter(y_true, np.maximum(y_pred, 1.0), s=3, alpha=0.2)
    low, high = float(np.min(y_true)), float(np.max(y_true))
    axis.plot([low, high], [low, high], color="red", linewidth=1, label="perfect")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("true price (AZN, log scale)")
    axis.set_ylabel("predicted price (AZN, log scale)")
    axis.set_title(f"Predicted vs. actual ({name})")
    axis.legend()
    axis.grid(alpha=0.3)
    return _save(figure, out_dir, "fig_error_pred_vs_actual.png")


def _fig_regions(region_rows, out_dir) -> str:
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    n_train = [max(r["n_train"], 1) for r in region_rows]
    error = [r["median_rel_error"] for r in region_rows]
    axis.scatter(n_train, error, s=18)
    for r in region_rows[:5]:
        axis.annotate(r["region"], (max(r["n_train"], 1), r["median_rel_error"]), fontsize=7)
    axis.set_xscale("log")
    axis.set_xlabel("training listings in district (log scale)")
    axis.set_ylabel("median relative error on test")
    axis.set_title("Error vs. district sample size")
    axis.grid(alpha=0.3)
    return _save(figure, out_dir, "fig_error_regions.png")


def _fig_classification(outputs, y_price, y_tier, threshold, out_dir) -> str:
    names = list(outputs)
    figure, axes = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 3.8))
    for axis, name in zip(axes, names):
        matrix = ev.confusion_matrix(y_tier, outputs[name]["pred"])
        axis.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, f"{value:,}", ha="center", va="center", fontsize=9)
        axis.set_xticks([0, 1], ["standard", "premium"])
        axis.set_yticks([0, 1], ["standard", "premium"])
        axis.set_xlabel("predicted")
        axis.set_ylabel("actual")
        axis.set_title(name, fontsize=9)
    for name in names:
        rows = classification_error_by_distance(y_price, y_tier, outputs[name]["pred"], threshold)
        axes[-1].plot([r["distance"] for r in rows], [r["error_rate"] for r in rows], marker="o", label=name)
    axes[-1].set_xlabel("|price - threshold| / threshold")
    axes[-1].set_ylabel("error rate")
    axes[-1].set_title("Errors concentrate near the threshold", fontsize=9)
    axes[-1].tick_params(axis="x", rotation=30)
    axes[-1].legend(fontsize=7)
    axes[-1].grid(alpha=0.3)
    return _save(figure, out_dir, "fig_error_classification.png")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run(
    X_train,
    X_test,
    y_price_test,
    y_tier_test,
    regression_predictions: dict,
    classification_outputs: dict,
    tier_threshold: float,
    out_dir: str = "report/figures",
    feature_names=None,
    tree_importances=None,
    svm_weights=None,
) -> dict:
    """Run all four error views and save the figures.

    X_train / X_test            pandas DataFrames (need `location=` columns,
                                optionally `area` and `rooms`)
    regression_predictions      {model name: predicted prices on the test set}
    classification_outputs      {model name: {"pred": labels, "score": scores}}
    tier_threshold              training-median price used to build the tiers
    tree_importances, svm_weights
                                optional vectors aligned with feature_names,
                                summarised next to the failure analysis
    """
    y_price_test = np.asarray(y_price_test, dtype=float)
    y_tier_test = np.asarray(y_tier_test)
    regions_test = region_labels(X_test)
    regions_train = region_labels(X_train)

    bands = {
        name: error_by_price_quantile(y_price_test, pred)
        for name, pred in regression_predictions.items()
    }
    primary = next(iter(regression_predictions))
    region_rows = error_by_region(
        y_price_test, regression_predictions[primary], regions_test, regions_train
    )
    results = {
        "regression_primary_model": primary,
        "price_quantile_bands": bands,
        "regions": region_rows,
        "sparse_vs_dense": sparse_vs_dense(region_rows),
        "largest_errors": largest_errors(
            y_price_test, regression_predictions[primary], X_test
        ),
        "classification_by_distance": {
            name: classification_error_by_distance(
                y_price_test, y_tier_test, out["pred"], tier_threshold
            )
            for name, out in classification_outputs.items()
        },
        "confusion_matrices": {
            name: ev.confusion_matrix(y_tier_test, out["pred"]).tolist()
            for name, out in classification_outputs.items()
        },
        "figures": {},
    }
    if feature_names is not None and tree_importances is not None:
        results["top_tree_importances"] = top_features(tree_importances, feature_names)
    if feature_names is not None and svm_weights is not None:
        results["top_svm_weights"] = top_features(svm_weights, feature_names, absolute=True)

    results["figures"]["price_quantile"] = _fig_quantiles(bands, out_dir)
    results["figures"]["pred_vs_actual"] = _fig_pred_vs_actual(
        y_price_test, regression_predictions[primary], primary, out_dir
    )
    if region_rows:
        results["figures"]["regions"] = _fig_regions(region_rows, out_dir)
    results["figures"]["classification"] = _fig_classification(
        classification_outputs, y_price_test, y_tier_test, tier_threshold, out_dir
    )
    return results
