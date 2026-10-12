import json
import os

import numpy as np
import pytest

from src import tree_experiments as te


def make_tier(y, threshold=None):
    """Stand-in for data_prep.make_tier_label (same rule: premium = price > TRAIN median)."""
    thr = float(np.median(y)) if threshold is None else threshold
    return (np.asarray(y) > thr).astype(int), thr


def make_splits(n=1500, d=8, seed=0, poison_test=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    price = np.exp(10 + 0.5 * X[:, 0] + 0.4 * X[:, 1] * (X[:, 2] > 0) + 0.2 * rng.normal(size=n))
    idx = rng.permutation(n)
    a, b = int(0.7 * n), int(0.85 * n)
    tr, va, tst = idx[:a], idx[a:b], idx[b:]
    X_te, y_te = X[tst].copy(), price[tst].copy()
    if poison_test:
        X_te[:] = np.nan
        y_te[:] = np.nan
    return X[tr], price[tr], X[va], price[va], X_te, y_te


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    out = tmp_path_factory.mktemp("figs")
    splits = make_splits(poison_test=True)            # NaN test set: run() must never touch it
    res = te.run(splits, out_dir=str(out), feature_names=[f"f{j}" for j in range(8)],
                 make_tier=make_tier, fast=True, verbose=False)
    return res, str(out), splits


def test_run_does_not_touch_test_set(results):
    res, _, _ = results                                # would have raised on NaN otherwise
    assert np.isfinite(res["task_A"]["final_val"]["val_rmse"])


def test_figures_are_written(results):
    res, out, _ = results
    for path in res["figures"].values():
        assert os.path.isfile(path) and os.path.getsize(path) > 5000
        assert os.path.dirname(path) == out


def test_result_is_json_serialisable(results):
    json.dumps(results[0])


def test_final_config_is_best_on_validation(results):
    res, _, _ = results
    a = res["task_A"]
    best_val = min(r["val_rmse"] for r in a["leaf_sweep"])
    assert np.isclose(a["final_val"]["val_rmse"], best_val)
    assert a["final_config"] == {"max_depth": a["final_val"]["max_depth"],
                                 "min_samples_leaf": a["final_val"]["min_samples_leaf"]}
    b = res["task_B"]
    crit = b["final_criterion"]
    best_f1 = max(r["val_f1"] for r in b["by_criterion"][crit]["leaf_sweep"])
    assert np.isclose(b["final_val"]["val_f1"], best_f1)
    assert b["final_val"]["val_f1"] >= max(r["val_f1"] for r in b["criterion_comparison"]) - 1e-12


def test_overfitting_story_is_visible(results):
    """Deeper trees fit train better; validation error is NOT monotone (it should stop improving)."""
    res, _, _ = results
    rows = res["task_A"]["depth_sweep"]
    train = [r["train_rmse"] for r in rows]
    assert train[-1] <= train[0]                       # unlimited tree has the lowest train error
    assert rows[-1]["val_rmse"] > min(r["val_rmse"] for r in rows) - 1e-12


def test_tier_threshold_uses_train_median_only(results):
    res, _, splits = results
    # train has < 4000 rows here, so fast mode does not subsample it
    assert np.isclose(res["tier_threshold"], np.median(splits[1]))


def test_learning_curve_has_growing_sizes(results):
    res, _, _ = results
    for key in ("task_A", "task_B"):
        sizes = [r["n_train"] for r in res[key]["learning_curve"]]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_importances_sum_to_at_most_one_and_find_signal(results):
    res, _, _ = results
    imp = dict(res["task_A"]["importances"])
    assert sum(imp.values()) <= 1.0 + 1e-9
    assert max(imp, key=imp.get) in ("f0", "f1", "f2")


def test_deterministic(tmp_path):
    s = make_splits(seed=3)
    r1 = te.run(s, out_dir=str(tmp_path / "a"), make_tier=make_tier, fast=True, verbose=False)
    r2 = te.run(s, out_dir=str(tmp_path / "b"), make_tier=make_tier, fast=True, verbose=False)
    for k in ("task_A", "task_B"):
        assert r1[k]["final_config"] == r2[k]["final_config"]
    assert np.isclose(r1["task_A"]["final_val"]["val_rmse"], r2["task_A"]["final_val"]["val_rmse"])


def test_final_test_scores(tmp_path):
    s = make_splits(seed=1)
    res = te.run(s, out_dir=str(tmp_path), make_tier=make_tier, fast=True, verbose=False)
    out = te.final_test_scores(s, res, make_tier=make_tier)
    assert out["task_A"]["RMSE"] > 0 and -1 <= out["task_A"]["R2"] <= 1
    b = out["task_B"]
    assert 0.5 < b["AUC"] <= 1.0 and 0 <= b["F1"] <= 1
    assert np.array(b["confusion_matrix"]).sum() == len(s[5])


# ------------------------------------------------------- LaTeX report helper
def test_write_latex_with_and_without_test_scores(tmp_path):
    s = make_splits(seed=4)
    res = te.run(s, out_dir=str(tmp_path / "f"), feature_names=[f"feat_{j}" for j in range(8)],
                 make_tier=make_tier, fast=True, verbose=False)

    m1, _ = te.write_latex(res, None, str(tmp_path / "t1"))                        # before the final test run
    t1 = open(m1, encoding="utf-8").read()
    assert "\\newcommand{\\treeAtestRMSE}{n/a}" in t1 and "nan" not in t1.lower().replace("\\newcommand", "")

    test = te.final_test_scores(s, res, make_tier=make_tier)
    m2, t2p = te.write_latex(res, test, str(tmp_path / "t2"))
    macros, tables = open(m2, encoding="utf-8").read(), open(t2p, encoding="utf-8").read()
    assert "n/a" not in macros                                                     # every macro is filled
    for name in ("treeAdepth", "treeAvalRMSE", "treeBtestFone", "treeBtestAUC", "treeCritGap", "treeAtop"):
        assert f"\\newcommand{{\\{name}}}" in macros
    assert "\\begin{table}" not in macros                                          # macros file is preamble-safe
    assert tables.count("\\begin{table}") == tables.count("\\end{table}") == 2
    assert macros.count("{") == macros.count("}") and tables.count("{") == tables.count("}")
    assert "\\label{tab:tree_criteria}" in tables and "\\label{tab:tree_final}" in tables
    assert "feat\\_0" in macros                                                    # underscores are escaped


def test_tex_escaping():
    assert te._tex("a_b & 50% $x$ #1") == r"a\_b \& 50\% \$x\$ \#1"