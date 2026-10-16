"""Tests for src/benchmark.py on small synthetic data (fast)."""
from __future__ import annotations

import numpy as np
import pytest

from src import benchmark as bm
from src.decision_tree import DecisionTree

CONFIGS = {
    "tree_regression": {"max_depth": 4, "min_samples_leaf": 10},
    "tree_classification": {"criterion": "gini", "max_depth": 4, "min_samples_leaf": 10},
    "svm": {"lambda_": 1e-2, "n_iters": 500},
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(400, 5))
    price = np.exp(11 + X[:, 0] + 0.5 * X[:, 1] + 0.2 * rng.normal(size=400))
    tier = (price > np.median(price)).astype(int)
    return X, price, tier


def test_build_specs_pairs_ours_with_sklearn():
    specs = bm.build_specs(CONFIGS, n_train=400)
    names = {(s["task"], s["name"]) for s in specs}
    assert ("regression", "NumPy DecisionTree") in names
    assert ("regression", "sklearn DecisionTreeRegressor") in names
    assert ("classification", "NumPy Pegasos SVM") in names
    assert ("classification", "sklearn SVC(linear)") in names
    assert {s["group"] for s in specs} == {"ours", "sklearn"}


def test_rbf_specs_added_when_gamma_present():
    cfg = {**CONFIGS, "svm": {**CONFIGS["svm"], "gamma": 0.5, "n_components": 32}}
    names = {s["name"] for s in bm.build_specs(cfg, n_train=400)}
    assert "NumPy Pegasos SVM + RFF" in names and "sklearn RBFSampler+SGD" in names


def test_factories_return_fresh_models():
    spec = bm.build_specs(CONFIGS, 400)[0]
    assert spec["factory"]() is not spec["factory"]()


def test_rff_pegasos_wrapper_fits_and_predicts(data):
    X, _, tier = data
    for gamma in (None, 0.5):
        model = bm.RFFPegasosSVM(1e-2, 800, gamma, 32).fit(X, tier)
        assert model.predict(X).shape == (400,)
        assert model.decision_function(X).shape == (400,)
        assert np.mean(model.predict(X) == tier) > 0.7


def test_cross_validate_all_covers_every_spec(data):
    X, price, tier = data
    specs = bm.build_specs(CONFIGS, 400)
    cv = bm.cross_validate_all(specs, X, price, tier, k=3)
    assert len(cv) == len(specs)
    for result in cv.values():
        assert result["k"] == 3 and result["fit_seconds"] >= 0
    assert cv[("classification", "NumPy DecisionTree")]["mean"]["F1"] > 0.6


def test_test_evaluate_all_returns_metrics_and_predictions(data):
    X, price, tier = data
    specs = bm.build_specs(CONFIGS, 300)
    out = bm.test_evaluate_all(specs, X[:300], price[:300], tier[:300], X[300:], price[300:], tier[300:])
    reg = out[("regression", "NumPy DecisionTree")]
    assert reg["pred"].shape == (100,) and "RMSE" in reg["metrics"]
    clf = out[("classification", "NumPy Pegasos SVM")]
    assert clf["score"].shape == (100,) and "ROC-AUC" in clf["metrics"]


def test_matched_trees_have_the_same_structure(data):
    X, price, _ = data
    specs = {s["name"]: s for s in bm.build_specs(CONFIGS, 400) if s["task"] == "regression"}
    ours = specs["NumPy DecisionTree"]["factory"]().fit(X, price)
    theirs = specs["sklearn DecisionTreeRegressor"]["factory"]().fit(X, price)
    report = bm.tree_structure_comparison(ours, theirs, X, "regression")
    assert report["prediction_agreement"] > 0.99
    assert report["our_depth"] == report["sklearn_depth"]


def test_timing_study_and_plot(data, tmp_path):
    X, price, tier = data
    specs = bm.build_specs(CONFIGS, 400)[:3]
    timing = bm.timing_study(specs, X, price, tier, fractions=(0.25, 1.0))
    assert all(len(series) == 2 and series[0]["n"] < series[1]["n"] for series in timing.values())
    assert (tmp_path / "fig_timing_scaling.png").parent.exists()
    assert bm.plot_timing(timing, str(tmp_path)).endswith("fig_timing_scaling.png")


def test_results_table_tex_contains_both_tasks_and_pm(data):
    X, price, tier = data
    specs = bm.build_specs(CONFIGS, 300)
    cv = bm.cross_validate_all(specs, X[:300], price[:300], tier[:300], k=3)
    test = bm.test_evaluate_all(specs, X[:300], price[:300], tier[:300], X[300:], price[300:], tier[300:])
    tex = bm.results_table_tex(test, cv)
    assert "tab:results_task_a" in tex and "tab:results_task_b" in tex
    assert r"$\pm$" in tex and "NumPy DecisionTree" in tex and "sklearn" in tex
    assert tex.count(r"\begin{table*}") == tex.count(r"\end{table*}") == 2


def test_print_results_runs(data, capsys):
    X, price, tier = data
    specs = bm.build_specs(CONFIGS, 300)
    cv = bm.cross_validate_all(specs, X[:300], price[:300], tier[:300], k=3)
    test = bm.test_evaluate_all(specs, X[:300], price[:300], tier[:300], X[300:], price[300:], tier[300:])
    bm.print_results(test, cv)
    assert "Task A" in capsys.readouterr().out
