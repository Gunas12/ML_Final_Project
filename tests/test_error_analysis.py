"""Tests for src/error_analysis.py on small synthetic frames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import error_analysis as ea


def _frame(n=400, seed=42):
    rng = np.random.default_rng(seed)
    regions = rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.35, 0.05])
    frame = pd.DataFrame(
        {
            "area": rng.uniform(30, 200, n),
            "rooms": rng.integers(1, 6, n).astype(float),
            "location=A": (regions == "A").astype(float),
            "location=B": (regions == "B").astype(float),
            "location=C": (regions == "C").astype(float),
        }
    )
    price = np.exp(10 + 0.01 * frame["area"].to_numpy() + rng.normal(0, 0.4, n))
    return frame, price, regions


def test_region_labels_recovers_names():
    frame, _, regions = _frame()
    assert list(ea.region_labels(frame)) == list(regions)


def test_region_labels_without_location_columns_is_unknown():
    labels = ea.region_labels(pd.DataFrame({"area": [1.0, 2.0]}))
    assert list(labels) == ["Unknown", "Unknown"]


def test_quantile_bands_partition_all_rows():
    _, price, _ = _frame()
    bands = ea.error_by_price_quantile(price, price * 0.9)
    assert sum(b["n"] for b in bands) == len(price)
    assert all(b["bias"] < 0 for b in bands)       # always under-predicting


def test_quantile_bands_expose_luxury_underprediction():
    _, price, _ = _frame()
    pred = np.minimum(price, np.quantile(price, 0.9))   # model cannot exceed p90
    top = ea.error_by_price_quantile(price, pred)[-1]
    assert top["band"] == "p99-100" and top["bias"] < 0


def test_region_rows_report_training_counts_and_skip_tiny_regions():
    frame, price, regions = _frame()
    rows = ea.error_by_region(price, price * 1.1, regions, regions, min_test=20)
    by_name = {r["region"]: r for r in rows}
    assert by_name["A"]["n_train"] == int((regions == "A").sum())
    assert by_name["A"]["median_rel_error"] == pytest.approx(0.1)
    assert "C" not in by_name or by_name["C"]["n_test"] >= 20


def test_sparse_vs_dense_splits_on_training_size():
    rows = [
        {"region": "x", "n_test": 50, "n_train": 10, "mae": 1, "median_rel_error": 0.5},
        {"region": "y", "n_test": 50, "n_train": 1000, "mae": 1, "median_rel_error": 0.1},
    ]
    out = ea.sparse_vs_dense(rows, sparse_below=200)
    assert out["sparse"]["median_rel_error"] == pytest.approx(0.5)
    assert out["dense"]["median_rel_error"] == pytest.approx(0.1)


def test_largest_errors_sorted_descending():
    frame, price, _ = _frame()
    pred = price.copy()
    pred[7] = price[7] * 10
    worst = ea.largest_errors(price, pred, frame, k=3)
    assert worst[0]["abs_error"] == pytest.approx(abs(pred[7] - price[7]))
    assert worst[0]["abs_error"] >= worst[1]["abs_error"] >= worst[2]["abs_error"]
    assert "area" in worst[0] and "region" in worst[0]


def test_classification_errors_concentrate_near_threshold():
    rng = np.random.default_rng(0)
    price = rng.uniform(50_000, 500_000, 2000)
    threshold = 200_000.0
    tier = (price > threshold).astype(int)
    noisy = price * rng.normal(1.0, 0.15, price.size)
    pred = (noisy > threshold).astype(int)
    rows = ea.classification_error_by_distance(price, tier, pred, threshold)
    assert rows[0]["error_rate"] > rows[-1]["error_rate"]


def test_top_features_orders_by_magnitude():
    names = ["a", "b", "c"]
    assert ea.top_features([0.1, -0.9, 0.5], names, k=2, absolute=True)[0][0] == "b"
    assert ea.top_features([0.1, -0.9, 0.5], names, k=1)[0][0] == "c"


def test_run_writes_all_figures_and_results(tmp_path):
    frame, price, _ = _frame()
    threshold = float(np.median(price))
    tier = (price > threshold).astype(int)
    rng = np.random.default_rng(1)
    out = ea.run(
        frame, frame, price, tier,
        {"model": price * rng.normal(1, 0.2, price.size)},
        {"clf": {"pred": tier, "score": tier.astype(float)}},
        threshold, out_dir=str(tmp_path),
        feature_names=list(frame.columns),
        tree_importances=np.ones(frame.shape[1]) / frame.shape[1],
        svm_weights=np.arange(frame.shape[1], dtype=float),
    )
    for key in ("price_quantile_bands", "regions", "largest_errors",
                "classification_by_distance", "confusion_matrices",
                "top_tree_importances", "top_svm_weights"):
        assert key in out
    for path in out["figures"].values():
        assert (tmp_path / path.split("/")[-1].split("\\")[-1]).exists()
