"""Tests for the src/run_all.py orchestrator (smoke test on tiny synthetic data)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import run_all


def _raw_listings(n=700, seed=42) -> pd.DataFrame:
    """A tiny frame with the same raw bina.az column names the pipeline reads."""
    rng = np.random.default_rng(seed)
    area = rng.uniform(30, 220, n).round()
    district = rng.choice(["Səbail r.", "Nəsimi r.", "Xətai r.", "Binəqədi r."], n, p=[0.4, 0.3, 0.2, 0.1])
    premium = {"Səbail r.": 1.6, "Nəsimi r.": 1.3, "Xətai r.": 1.0, "Binəqədi r.": 0.8}
    price = (area * 1500 * np.array([premium[d] for d in district]) * rng.lognormal(0, 0.25, n)).round()
    return pd.DataFrame(
        {
            "price": price, "currency_x": "AZN", "location": district, "city": "bakı",
            "Kateqoriya": rng.choice(["Mənzil", "Həyət evi"], n), "Təmir": rng.choice(["Təmirli", "Təmirsiz"], n),
            "Çıxarış": rng.choice(["Çıxarış var", "Çıxarış yoxdur"], n), "İpoteka": rng.choice(["İpoteka var", "yox"], n),
            "Binanın növü": rng.choice(["Yeni tikili", "Köhnə tikili"], n),
            "lat": 40.4 + rng.normal(0, 0.03, n), "lng": 49.85 + rng.normal(0, 0.03, n),
            "Sahə": area, "Otaq sayı": rng.integers(1, 6, n), "Mərtəbə": [f"{rng.integers(1, 9)}/9" for _ in range(n)],
            "Torpaq sahəsi": np.nan, "unit_price": price / area, "total_price": price,
        }
    )


def test_parse_args_defaults_and_flags():
    args = run_all.parse_args([])
    assert not args.fast and not args.bonus and args.folds is None
    args = run_all.parse_args(["--fast", "--bonus", "--folds", "4"])
    assert args.fast and args.bonus and args.folds == 4


def test_final_configs_follow_validation_choices():
    tree = {"task_A": {"final_config": {"max_depth": 8, "min_samples_leaf": 100}},
            "task_B": {"final_config": {"max_depth": 6, "min_samples_leaf": 20}, "final_criterion": "entropy"}}
    linear = {"best_lambda": 1e-3, "selected_model_type": "linear", "best_gamma": None, "n_components": 256}
    cfg = run_all.final_configs(tree, linear, fast=False)
    assert cfg["tree_regression"] == {"max_depth": 8, "min_samples_leaf": 100}
    assert cfg["tree_classification"]["criterion"] == "entropy"
    assert cfg["svm"] == {"lambda_": 1e-3, "n_iters": 20_000}
    rff = {**linear, "selected_model_type": "rff", "best_gamma": 0.1}
    cfg = run_all.final_configs(tree, rff, fast=True)
    assert cfg["svm"]["gamma"] == 0.1 and cfg["svm"]["n_iters"] == 2_000


def test_jsonable_drops_models_and_converts_numpy():
    out = run_all._jsonable({"a": np.float64(1.5), "b": np.arange(3), "model": object(), ("t", "n"): 1})
    assert out == {"a": 1.5, "b": [0, 1, 2], "t | n": 1}
    json.dumps(out)


def test_fast_mode_subsamples_but_keeps_seeded_rows():
    X = pd.DataFrame({"a": np.arange(100.0)})
    a, _ = run_all._subsample(X, np.arange(100), 10, seed=1)
    b, _ = run_all._subsample(X, np.arange(100), 10, seed=1)
    assert len(a) == 10 and a.equals(b)


def test_run_all_fast_end_to_end(tmp_path, monkeypatch):
    csv = tmp_path / "listings.csv"
    _raw_listings().to_csv(csv, index=False)
    monkeypatch.chdir(tmp_path)
    results = run_all.main(["--fast", "--data", str(csv)])

    assert results["mode"] == "fast" and results["folds"] == 3
    sizes = results["split_sizes"]
    assert sizes["train"] > sizes["validation"] and sizes["test"] > 0
    assert (tmp_path / "report" / "results_table.tex").exists()
    assert (tmp_path / "results" / "results.json").exists()
    for figure in ("fig_error_price_quantile.png", "fig_error_classification.png", "fig_timing_scaling.png"):
        assert (tmp_path / "report" / "figures" / figure).exists()
    json.loads((tmp_path / "results" / "results.json").read_text(encoding="utf-8"))
    # every model scored the same test rows
    keys = [k for k in results["test"] if k[0] == "classification"]
    assert len(keys) >= 4


def test_run_all_is_reproducible(tmp_path, monkeypatch):
    csv = tmp_path / "listings.csv"
    _raw_listings().to_csv(csv, index=False)
    monkeypatch.chdir(tmp_path)
    a = run_all.main(["--fast", "--data", str(csv)])
    b = run_all.main(["--fast", "--data", str(csv)])
    for key in a["test"]:
        assert a["test"][key]["metrics"] == b["test"][key]["metrics"]
