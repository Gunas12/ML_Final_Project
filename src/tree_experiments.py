"""
tree_experiments.py — P4a experiments for the from-scratch DecisionTree.

What this module produces (all numbers come from VALIDATION data only):
  1. max_depth sweep           (Task A regression, Task B classification)
  2. min_samples_leaf sweep    (the "one more hyperparameter" the statement asks for)
  3. Gini vs entropy           (Task B)
  4. learning curves           (train vs validation score vs training-set size)
  5. impurity-decrease feature importances of the final trees
  6. final configurations, chosen on validation evidence ONLY

The test set is never touched by `run()`. Call `final_test_scores()` ONCE at
the very end (run_all.py does this) to score the chosen configurations.
`write_latex()` then turns the numbers into LaTeX tables and \newcommand macros
for the report, so no number is ever copied by hand.

Interface (agreed contract):
    run(splits, out_dir="report/figures") -> dict
    splits = (X_tr, y_tr, X_val, y_val, X_te, y_te) as returned by
             data_prep.train_val_test_split, with y = price.
    Task B labels (premium/standard) are derived HERE with
    data_prep.make_tier_label using the TRAIN median only.

Model-selection rule
    Task A: lowest validation RMSE.   Task B: highest validation F1.
    Ties go to the simpler tree (fewer leaves).
    Procedure: sweep max_depth (min_samples_leaf=1) -> take best depth ->
    sweep min_samples_leaf at that depth -> take best.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")                      # no display needed (works on servers / CI)
import matplotlib.pyplot as plt
import numpy as np

from . import data_prep
from . import evaluate as ev
from .decision_tree import DecisionTree

DEPTHS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, None]          # None = unlimited
LEAVES = [1, 2, 5, 10, 20, 50, 100, 200]
LC_FRACTIONS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
CRITERIA_B = ("gini", "entropy")

FAST_DEPTHS = [2, 4, 6, 8, None]
FAST_LEAVES = [1, 5, 20, 50]
FAST_LC = [0.1, 0.3, 0.6, 1.0]
FAST_MAX_TRAIN = 4000


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------
def _reg_scores(model, X, y, prefix: str) -> dict:
    p = model.predict(X)
    return {f"{prefix}_rmse": ev.rmse(y, p), f"{prefix}_mae": ev.mae(y, p), f"{prefix}_r2": ev.r2(y, p)}


def _pos_proba(model, X) -> np.ndarray:
    """P(premium = 1); zeros if the fitted tree never saw class 1."""
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(X.shape[0])
    return model.predict_proba(X)[:, classes.index(1)]


def _clf_scores(model, X, y, prefix: str) -> dict:
    p = model.predict(X)
    _, _, f1 = ev.precision_recall_f1(y, p)
    return {f"{prefix}_f1": f1, f"{prefix}_acc": ev.accuracy(y, p),
            f"{prefix}_auc": ev.roc_auc(y, _pos_proba(model, X))}


def _fit_eval(task, criterion, cfg, Xtr, ytr, Xva, yva) -> dict:
    model = DecisionTree(task=task, criterion=criterion, **cfg)
    _, fit_s = ev.timeit(model.fit, Xtr, ytr)
    scorer = _reg_scores if task == "regression" else _clf_scores
    row = {**cfg, "criterion": criterion, "n_leaves": model.n_leaves_, "depth": model.depth_,
           "fit_seconds": fit_s}
    row.update(scorer(model, Xtr, ytr, "train"))
    row.update(scorer(model, Xva, yva, "val"))
    return row


def _pick_best(rows, metric: str, higher_is_better: bool) -> dict:
    """Best validation row; ties (within 1e-12) go to the tree with fewer leaves."""
    sign = 1.0 if higher_is_better else -1.0
    return max(rows, key=lambda r: (round(sign * r[metric], 12), -r["n_leaves"]))


def _sweep(task, criterion, name, values, base, data):
    rows = []
    for v in values:
        rows.append(_fit_eval(task, criterion, {**base, name: v}, *data))
    return rows


def _tune(task, criterion, data, depths, leaves, metric, higher):
    """Coordinate search: depth first (leaf=1), then min_samples_leaf at the best depth."""
    depth_rows = _sweep(task, criterion, "max_depth", depths, {"min_samples_leaf": 1}, data)
    best_depth = _pick_best(depth_rows, metric, higher)["max_depth"]
    leaf_rows = _sweep(task, criterion, "min_samples_leaf", leaves, {"max_depth": best_depth}, data)
    best = _pick_best(leaf_rows, metric, higher)
    final_cfg = {"max_depth": best["max_depth"], "min_samples_leaf": best["min_samples_leaf"]}
    return {"depth_sweep": depth_rows, "leaf_sweep": leaf_rows, "final_config": final_cfg, "final_row": best}


def _learning_curve(task, criterion, cfg, fractions, Xtr, ytr, Xva, yva, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ytr))                 # nested subsets: bigger sets contain smaller ones
    rows = []
    for f in fractions:
        idx = order[: max(2, int(round(f * len(ytr))))]
        r = _fit_eval(task, criterion, cfg, Xtr[idx], ytr[idx], Xva, yva)
        r["n_train"] = int(len(idx))
        rows.append(r)
    return rows


def _importances(task, criterion, cfg, Xtr, ytr, names, top=15):
    model = DecisionTree(task=task, criterion=criterion, **cfg).fit(Xtr, ytr)
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top]
    return [(names[j], float(imp[j])) for j in order if imp[j] > 0]


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _xlabels(vals):
    return ["∞" if v is None else str(v) for v in vals]


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def _mark_best(ax, xs, rows, metric, higher):
    vals = [r[metric] for r in rows]
    i = int(np.argmax(vals) if higher else np.argmin(vals))
    ax.plot(xs[i], vals[i], marker="*", markersize=14, color="gold", markeredgecolor="black", zorder=5)


def _fig_sweep(path, name, a_rows, b_rows_by_crit, xvals, xlabel):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xs = list(range(len(xvals)))
    ax = axes[0]
    ax.plot(xs, [r["train_rmse"] for r in a_rows], "o-", label="train")
    ax.plot(xs, [r["val_rmse"] for r in a_rows], "s-", label="validation")
    _mark_best(ax, xs, a_rows, "val_rmse", higher=False)
    _style(ax, "Task A: regression (lower is better)", xlabel, "RMSE")
    ax.set_xticks(xs, _xlabels(xvals))
    ax.legend()
    ax = axes[1]
    for crit, ls in zip(CRITERIA_B, ("-", "--")):
        rows = b_rows_by_crit[crit]
        ax.plot(xs, [r["train_f1"] for r in rows], "o" + ls, alpha=0.6, label=f"train ({crit})")
        ax.plot(xs, [r["val_f1"] for r in rows], "s" + ls, label=f"validation ({crit})")
        _mark_best(ax, xs, rows, "val_f1", higher=True)
    _style(ax, "Task B: classification (higher is better)", xlabel, "F1 (premium)")
    ax.set_xticks(xs, _xlabels(xvals))
    ax.legend(fontsize=8)
    fig.suptitle(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_learning_curve(path, a_rows, b_rows, a_cfg, b_cfg, b_crit):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    n = [r["n_train"] for r in a_rows]
    ax = axes[0]
    ax.plot(n, [r["train_rmse"] for r in a_rows], "o-", label="train")
    ax.plot(n, [r["val_rmse"] for r in a_rows], "s-", label="validation")
    _style(ax, f"Task A  (depth={a_cfg['max_depth']}, leaf={a_cfg['min_samples_leaf']})", "training samples", "RMSE")
    ax.legend()
    n = [r["n_train"] for r in b_rows]
    ax = axes[1]
    ax.plot(n, [r["train_f1"] for r in b_rows], "o-", label="train")
    ax.plot(n, [r["val_f1"] for r in b_rows], "s-", label="validation")
    _style(ax, f"Task B  ({b_crit}, depth={b_cfg['max_depth']}, leaf={b_cfg['min_samples_leaf']})", "training samples", "F1 (premium)")
    ax.legend()
    fig.suptitle("Learning curves of the final trees", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_importance(path, imp_a, imp_b, b_crit):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, imp, title in ((axes[0], imp_a, "Task A (regression, MSE)"), (axes[1], imp_b, f"Task B (classification, {b_crit})")):
        names = [n for n, _ in imp][::-1]
        vals = [v for _, v in imp][::-1]
        ax.barh(names, vals)
        _style(ax, title, "impurity-decrease importance", "")
        ax.grid(axis="y", alpha=0)
    fig.suptitle("Feature importances of the final trees", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main entry points
# ---------------------------------------------------------------------------
def run(splits, out_dir: str = "report/figures", feature_names=None, make_tier=None,
        fast: bool = False, verbose: bool = True) -> dict:
    """
    Run all tree experiments on TRAIN/VALIDATION data and save figures to out_dir.
    The test arrays inside `splits` are deliberately ignored.
    Returns a JSON-friendly dict (see keys at the bottom).
    """
    X_tr, y_tr, X_val, y_val = (np.asarray(a) for a in splits[:4])   # NOTE: splits[4:] (test) never used
    make_tier = make_tier or data_prep.make_tier_label
    os.makedirs(out_dir, exist_ok=True)
    names = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X_tr.shape[1])]
    seed = data_prep.SEED

    depths, leaves, fractions = (FAST_DEPTHS, FAST_LEAVES, FAST_LC) if fast else (DEPTHS, LEAVES, LC_FRACTIONS)
    if fast and len(y_tr) > FAST_MAX_TRAIN:                     # quick development runs only
        idx = np.sort(np.random.default_rng(seed).choice(len(y_tr), FAST_MAX_TRAIN, replace=False))
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    # Task B labels: threshold = median of the TRAIN prices only
    yb_tr, threshold = make_tier(y_tr)
    yb_val, _ = make_tier(y_val, threshold=threshold)
    yb_tr, yb_val = np.asarray(yb_tr), np.asarray(yb_val)

    data_a = (X_tr, y_tr, X_val, y_val)
    data_b = (X_tr, yb_tr, X_val, yb_val)

    # ---- Task A -----------------------------------------------------------
    A = _tune("regression", "mse", data_a, depths, leaves, "val_rmse", higher=False)
    # ---- Task B (both criteria) ------------------------------------------
    B = {c: _tune("classification", c, data_b, depths, leaves, "val_f1", higher=True) for c in CRITERIA_B}
    comparison = [{"criterion": c, **B[c]["final_config"], "val_f1": B[c]["final_row"]["val_f1"],
                   "val_auc": B[c]["final_row"]["val_auc"], "val_acc": B[c]["final_row"]["val_acc"],
                   "n_leaves": B[c]["final_row"]["n_leaves"], "fit_seconds": B[c]["final_row"]["fit_seconds"]}
                  for c in CRITERIA_B]
    best_crit = max(comparison, key=lambda r: (round(r["val_f1"], 12), -r["n_leaves"]))["criterion"]
    final_b_cfg = B[best_crit]["final_config"]

    # ---- learning curves + importances of the FINAL trees -------------------
    lc_a = _learning_curve("regression", "mse", A["final_config"], fractions, *data_a, seed)
    lc_b = _learning_curve("classification", best_crit, final_b_cfg, fractions, *data_b, seed)
    imp_a = _importances("regression", "mse", A["final_config"], X_tr, y_tr, names)
    imp_b = _importances("classification", best_crit, final_b_cfg, X_tr, yb_tr, names)

    # ---- figures ------------------------------------------------------------
    figs = {
        "depth_sweep": os.path.join(out_dir, "fig_tree_depth_sweep.png"),
        "leaf_sweep": os.path.join(out_dir, "fig_tree_min_leaf_sweep.png"),
        "learning_curve": os.path.join(out_dir, "fig_tree_learning_curve.png"),
        "importance": os.path.join(out_dir, "fig_tree_importance.png"),
    }
    _fig_sweep(figs["depth_sweep"], "Effect of max_depth (min_samples_leaf = 1)", A["depth_sweep"],
               {c: B[c]["depth_sweep"] for c in CRITERIA_B}, depths, "max_depth  (∞ = unlimited)")
    _fig_sweep(figs["leaf_sweep"], "Effect of min_samples_leaf (at each task's best max_depth)", A["leaf_sweep"],
               {c: B[c]["leaf_sweep"] for c in CRITERIA_B}, leaves, "min_samples_leaf")
    _fig_learning_curve(figs["learning_curve"], lc_a, lc_b, A["final_config"], final_b_cfg, best_crit)
    _fig_importance(figs["importance"], imp_a, imp_b, best_crit)

    if verbose:
        fa, fb = A["final_row"], B[best_crit]["final_row"]
        print("\n=== Decision tree experiments (validation only) ===")
        print(f"Task A final: depth={A['final_config']['max_depth']}  min_samples_leaf={A['final_config']['min_samples_leaf']}"
              f"  -> val RMSE {fa['val_rmse']:.4g}  MAE {fa['val_mae']:.4g}  R2 {fa['val_r2']:.4f}  leaves {fa['n_leaves']}")
        print("Task B  Gini vs entropy (each at its own best depth / min_samples_leaf):")
        print(ev.format_table({f"{r['criterion']} (d={r['max_depth']}, leaf={r['min_samples_leaf']})":
                               {"val F1": r["val_f1"], "val AUC": r["val_auc"], "val acc": r["val_acc"], "leaves": r["n_leaves"]}
                               for r in comparison}))
        print(f"Task B final: {best_crit}, depth={final_b_cfg['max_depth']}, min_samples_leaf={final_b_cfg['min_samples_leaf']}")
        print("Figures:", ", ".join(os.path.basename(p) for p in figs.values()))

    return {
        "tier_threshold": float(threshold),
        "task_A": {"depth_sweep": A["depth_sweep"], "leaf_sweep": A["leaf_sweep"],
                   "final_config": A["final_config"], "final_val": A["final_row"],
                   "learning_curve": lc_a, "importances": imp_a},
        "task_B": {"by_criterion": {c: {"depth_sweep": B[c]["depth_sweep"], "leaf_sweep": B[c]["leaf_sweep"],
                                        "final_config": B[c]["final_config"], "final_val": B[c]["final_row"]}
                                    for c in CRITERIA_B},
                   "criterion_comparison": comparison, "final_criterion": best_crit,
                   "final_config": final_b_cfg, "final_val": B[best_crit]["final_row"],
                   "learning_curve": lc_b, "importances": imp_b},
        "figures": figs,
    }


def final_test_scores(splits, results: dict, make_tier=None) -> dict:
    """
    Fit the FINAL configurations (chosen on validation in `run`) on the training
    set and score them on the TEST set. Call this ONCE, at the very end.
    `results` is the dict returned by run().
    """
    X_tr, y_tr, _, y_val, X_te, y_te = (np.asarray(a) for a in splits)
    make_tier = make_tier or data_prep.make_tier_label
    cfg_a = results["task_A"]["final_config"]
    cfg_b = results["task_B"]["final_config"]
    crit = results["task_B"]["final_criterion"]

    model_a = DecisionTree(task="regression", criterion="mse", **cfg_a)
    _, fit_a = ev.timeit(model_a.fit, X_tr, y_tr)
    pred_a, pred_a_s = ev.timeit(model_a.predict, X_te)

    yb_tr, threshold = make_tier(y_tr)
    yb_te, _ = make_tier(y_te, threshold=threshold)
    yb_te = np.asarray(yb_te)
    model_b = DecisionTree(task="classification", criterion=crit, **cfg_b)
    _, fit_b = ev.timeit(model_b.fit, X_tr, np.asarray(yb_tr))
    pred_b = model_b.predict(X_te)
    prec, rec, f1 = ev.precision_recall_f1(yb_te, pred_b)

    return {
        "task_A": {"config": cfg_a, "RMSE": ev.rmse(y_te, pred_a), "MAE": ev.mae(y_te, pred_a),
                   "R2": ev.r2(y_te, pred_a), "fit_seconds": fit_a, "predict_seconds": pred_a_s,
                   "n_leaves": model_a.n_leaves_},
        "task_B": {"config": {"criterion": crit, **cfg_b}, "Precision": prec, "Recall": rec, "F1": f1,
                   "Accuracy": ev.accuracy(yb_te, pred_b), "AUC": ev.roc_auc(yb_te, _pos_proba(model_b, X_te)),
                   "confusion_matrix": ev.confusion_matrix(yb_te, pred_b).tolist(),
                   "fit_seconds": fit_b, "n_leaves": model_b.n_leaves_},
    }


# ---------------------------------------------------------------------------
# report helpers: LaTeX tables + macros (generated, never edited by hand)
# ---------------------------------------------------------------------------
_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex(text) -> str:
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in str(text))


def _num(x, digits: int = 3) -> str:
    """Format a number for LaTeX text: 7,612 for large values, 0.844 for small ones."""
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.{digits}f}"


def _depth_txt(d) -> str:
    return "unlimited" if d is None else str(d)


def write_latex(results: dict, test_scores: dict | None = None, out_dir: str = "report") -> tuple:
    """
    Write two files into `out_dir` from run() (and optionally final_test_scores()) output:
      tree_macros.tex  -> \\newcommand macros (\\treeAdepth, \\treeBtestFone, ...)
                          put  \\input{tree_macros.tex}  in the PREAMBLE of report.tex
      tree_tables.tex  -> two float tables (tab:tree_criteria, tab:tree_final)
                          put  \\input{tree_tables.tex}  in the Results section
    Re-run the pipeline and every number in the report updates. Without `test_scores`
    the test columns/macros read "n/a". Returns (macros_path, tables_path).
    """
    A, B = results["task_A"], results["task_B"]
    fa, fb = A["final_val"], B["final_val"]
    ta = (test_scores or {}).get("task_A", {})
    tb = (test_scores or {}).get("task_B", {})
    unl = next((r for r in A["depth_sweep"] if r["max_depth"] is None), A["depth_sweep"][-1])
    cmp_by = {r["criterion"]: r for r in B["criterion_comparison"]}
    gap = abs(cmp_by["gini"]["val_f1"] - cmp_by["entropy"]["val_f1"])

    def top(imp, k=3):
        return ", ".join(_tex(n) for n, _ in imp[:k]) or "n/a"

    macros = {
        "treeTierThreshold": _num(results["tier_threshold"]),
        "treeAdepth": _depth_txt(A["final_config"]["max_depth"]),
        "treeAleaf": str(A["final_config"]["min_samples_leaf"]),
        "treeAleaves": str(fa["n_leaves"]),
        "treeAvalRMSE": _num(fa["val_rmse"]), "treeAvalMAE": _num(fa["val_mae"]), "treeAvalRtwo": _num(fa["val_r2"]),
        "treeAtrainRMSE": _num(fa["train_rmse"]),
        "treeAunlimTrainRMSE": _num(unl["train_rmse"]), "treeAunlimValRMSE": _num(unl["val_rmse"]),
        "treeAtestRMSE": _num(ta.get("RMSE")), "treeAtestMAE": _num(ta.get("MAE")), "treeAtestRtwo": _num(ta.get("R2")),
        "treeAtop": top(A["importances"]),
        "treeBcrit": _tex(B["final_criterion"]),
        "treeBdepth": _depth_txt(B["final_config"]["max_depth"]),
        "treeBleaf": str(B["final_config"]["min_samples_leaf"]),
        "treeBleaves": str(fb["n_leaves"]),
        "treeBvalFone": _num(fb["val_f1"]), "treeBvalAUC": _num(fb["val_auc"]), "treeBvalAcc": _num(fb["val_acc"]),
        "treeBtrainFone": _num(fb["train_f1"]),
        "treeBtestPrec": _num(tb.get("Precision")), "treeBtestRec": _num(tb.get("Recall")),
        "treeBtestFone": _num(tb.get("F1")), "treeBtestAcc": _num(tb.get("Accuracy")), "treeBtestAUC": _num(tb.get("AUC")),
        "treeBtop": top(B["importances"]),
        "treeGiniFone": _num(cmp_by["gini"]["val_f1"]), "treeEntropyFone": _num(cmp_by["entropy"]["val_f1"]),
        "treeGiniAUC": _num(cmp_by["gini"]["val_auc"]), "treeEntropyAUC": _num(cmp_by["entropy"]["val_auc"]),
        "treeCritGap": _num(gap, 4),
    }

    header = "% AUTO-GENERATED by tree_experiments.write_latex -- do not edit by hand; re-run the pipeline instead."
    macro_lines = [header] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]

    rows = []
    for r in B["criterion_comparison"]:
        rows.append(f"{_tex(r['criterion'])} & {_depth_txt(r['max_depth'])} & {r['min_samples_leaf']} & "
                    f"{_num(r['val_f1'])} & {_num(r['val_auc'])} & {_num(r['val_acc'])} & {r['n_leaves']} \\\\")
    lines = [header, "", "\\begin{table}[t]", "  \\centering",
             "  \\caption{Gini vs.\\ entropy for the decision tree on Task~B (validation set; each criterion at its own best "
             "\\texttt{max\\_depth} and \\texttt{min\\_samples\\_leaf}).}",
             "  \\label{tab:tree_criteria}", "  \\begin{tabular}{@{}lrrrrrr@{}}", "    \\toprule",
             "    Criterion & Depth & Leaf & F1 & AUC & Acc. & Leaves \\\\", "    \\midrule"]
    lines += [f"    {row}" for row in rows]
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]

    cfg_a = f"depth {_depth_txt(A['final_config']['max_depth'])}, leaf {A['final_config']['min_samples_leaf']}"
    cfg_b = (f"{_tex(B['final_criterion'])}, depth {_depth_txt(B['final_config']['max_depth'])}, "
             f"leaf {B['final_config']['min_samples_leaf']}")
    lines += ["", "\\begin{table}[t]", "  \\centering",
              "  \\caption{Final decision-tree configurations (chosen on validation only) and their scores.}",
              "  \\label{tab:tree_final}", "  \\begin{tabular}{@{}llrr@{}}", "    \\toprule",
              "    Metric & Configuration & Validation & Test \\\\", "    \\midrule",
              f"    A: RMSE & {cfg_a} & {_num(fa['val_rmse'])} & {_num(ta.get('RMSE'))} \\\\",
              f"    A: $R^2$ & & {_num(fa['val_r2'])} & {_num(ta.get('R2'))} \\\\",
              f"    B: F1 & {cfg_b} & {_num(fb['val_f1'])} & {_num(tb.get('F1'))} \\\\",
              f"    B: AUC & & {_num(fb['val_auc'])} & {_num(tb.get('AUC'))} \\\\",
              "    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""]

    os.makedirs(out_dir, exist_ok=True)
    macros_path = os.path.join(out_dir, "tree_macros.tex")
    tables_path = os.path.join(out_dir, "tree_tables.tex")
    with open(macros_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(macro_lines) + "\n")
    with open(tables_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return macros_path, tables_path