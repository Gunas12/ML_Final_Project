"""One command reproduces every number, table and figure in the report.

    python -m src.run_all            # full run (all data, final settings)
    python -m src.run_all --fast     # small subset, for development only
    python -m src.run_all --bonus    # also run optional bonus modules, if present

Pipeline order (owner in brackets):
  1  load -> clean -> features -> split -> tier label -> standardise  [Member 1]
  2  EDA figures                                                      [Member 1]
  3  decision-tree experiments (validation only)                      [Member 2]
  4  SVM experiments (validation only)                                [Member 3]
  5  matched sklearn baselines + cross-validation + timing            [Member 4]
  6  error analysis                                                   [Member 4]
  7  results tables (console + LaTeX + JSON)                          [Member 4]

Test-set discipline: steps 3-4 select every hyperparameter on the validation
split. Step 5 cross-validates on train+validation and scores the test split
exactly once with the final configurations.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from . import benchmark, data_prep, eda, error_analysis, evaluate
from . import svm_experiments, tree_experiments

SEED = data_prep.SEED
FIGURES = Path("report") / "figures"
REPORT_DIR = Path("report")
RESULTS_DIR = Path("results")
BONUS_MODULES = ("bonus_linear", "bonus_ensembles", "bonus_boosting", "bonus_clustering")

# SVM iteration budgets mirror svm_experiments.run (fast / full)
SVM_ITERS = {True: 2_000, False: 20_000}
FAST_ROWS = {"train": 6_000, "val": 2_000, "test": 2_000}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce all results of the ML final project.")
    parser.add_argument("--fast", action="store_true", help="small subset + fewer folds (development only)")
    parser.add_argument("--bonus", action="store_true", help="also run optional bonus modules if present")
    parser.add_argument("--data", default=data_prep.DATA_PATH, help="path to the raw bina.az CSV")
    parser.add_argument("--folds", type=int, default=None, help="CV folds (default 5, or 3 with --fast)")
    return parser.parse_args(argv)


def _subsample(X, y, rows: int, seed: int):
    """Seeded row subsample used only by --fast."""
    if len(y) <= rows:
        return X, y
    idx = np.sort(np.random.default_rng(seed).choice(len(y), rows, replace=False))
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx], np.asarray(y)[idx]


def prepare_data(path: str, fast: bool) -> dict:
    """Step 1: everything up to labelled, split, standardised arrays."""
    cleaned = data_prep.clean(data_prep.load_raw(path))
    X, y_price, feature_names = data_prep.make_features(cleaned)
    X_tr, y_tr, X_val, y_val, X_te, y_te = data_prep.train_val_test_split(X, y_price, seed=SEED)
    if fast:
        X_tr, y_tr = _subsample(X_tr, y_tr, FAST_ROWS["train"], SEED)
        X_val, y_val = _subsample(X_val, y_val, FAST_ROWS["val"], SEED + 1)
        X_te, y_te = _subsample(X_te, y_te, FAST_ROWS["test"], SEED + 2)
    tier_tr, threshold = data_prep.make_tier_label(y_tr)          # threshold: TRAIN median only
    tier_val, _ = data_prep.make_tier_label(y_val, threshold=threshold)
    tier_te, _ = data_prep.make_tier_label(y_te, threshold=threshold)
    scaled = data_prep.standardize(X_tr, X_val, X_te)
    return {
        "n_rows": len(y_price), "feature_names": feature_names, "threshold": threshold,
        "X_tr": X_tr, "X_val": X_val, "X_te": X_te,
        "y_tr": np.asarray(y_tr), "y_val": np.asarray(y_val), "y_te": np.asarray(y_te),
        "tier_tr": np.asarray(tier_tr), "tier_val": np.asarray(tier_val), "tier_te": np.asarray(tier_te),
        "scaled": scaled,
    }


def final_configs(tree_results: dict, svm_results: dict, fast: bool) -> dict:
    """Turn the validation-selected settings into model configurations."""
    keep = ("max_depth", "min_samples_split", "min_samples_leaf", "min_impurity_decrease")
    cfg_a = {k: v for k, v in tree_results["task_A"]["final_config"].items() if k in keep}
    cfg_b = {k: v for k, v in tree_results["task_B"]["final_config"].items() if k in keep}
    cfg_b["criterion"] = tree_results["task_B"]["final_criterion"]
    svm = {"lambda_": svm_results["best_lambda"], "n_iters": SVM_ITERS[fast]}
    if svm_results["selected_model_type"] == "rff":
        svm.update(gamma=svm_results["best_gamma"], n_components=svm_results["n_components"])
    return {"tree_regression": cfg_a, "tree_classification": cfg_b, "svm": svm}


def _jsonable(value):
    """Recursively convert numpy / path objects; drop fitted models."""
    if isinstance(value, dict):
        return {str(k) if not isinstance(k, tuple) else " | ".join(k): _jsonable(v)
                for k, v in value.items() if k not in ("model", "rff", "pred", "score")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def run_bonus() -> dict:
    """Optional bonus modules. A failure here never touches the core results."""
    outcomes = {}
    for name in BONUS_MODULES:
        try:
            module = importlib.import_module(f"src.{name}")
        except ImportError:
            outcomes[name] = "not present"
            continue
        try:
            outcomes[name] = _jsonable(module.run())
        except Exception as error:   # noqa: BLE001 - bonus must not break the core run
            outcomes[name] = f"failed: {error}"
    return outcomes


def main(argv=None) -> dict:
    args = parse_args(argv)
    fast = args.fast
    folds = args.folds or (3 if fast else 5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- 1. data ----------------------------------------------------------
    d = prepare_data(args.data, fast)
    print(f"Data rows: {d['n_rows']:,}   features: {len(d['feature_names'])}   mode: {'FAST' if fast else 'FULL'}")
    print(f"Split sizes: train={len(d['y_tr']):,}  validation={len(d['y_val']):,}  test={len(d['y_te']):,}")
    print(f"Tier threshold (train median): {d['threshold']:,.2f} AZN   seed={SEED}")

    # ---- 2. EDA -----------------------------------------------------------
    eda_figures = eda.run((d["X_tr"], d["y_tr"], d["X_val"], d["y_val"], d["X_te"], d["y_te"]), out_dir=FIGURES)

    # ---- 3. tree experiments (validation only) ------------------------------
    tree_results = tree_experiments.run(
        (d["X_tr"], d["y_tr"], d["X_val"], d["y_val"], d["X_te"], d["y_te"]),
        out_dir=str(FIGURES), feature_names=d["feature_names"], fast=fast,
    )

    # ---- 4. SVM experiments (validation only) -------------------------------
    s_tr, s_val, s_te = d["scaled"]
    svm_results = svm_experiments.run(
        (s_tr, d["tier_tr"], s_val, d["tier_val"], s_te, d["tier_te"]),
        out_dir=str(FIGURES), fast=fast, random_state=SEED,
    )

    # ---- 5. matched benchmark ------------------------------------------------
    configs = final_configs(tree_results, svm_results, fast)
    print("\nFinal configurations (chosen on validation only):")
    for name, cfg in configs.items():
        print(f"  {name}: {cfg}")
    specs = benchmark.build_specs(configs, n_train=len(d["y_tr"]), seed=SEED)

    X_trval = np.vstack([np.asarray(d["X_tr"], dtype=float), np.asarray(d["X_val"], dtype=float)])
    cv = benchmark.cross_validate_all(
        specs, X_trval,
        np.concatenate([d["y_tr"], d["y_val"]]), np.concatenate([d["tier_tr"], d["tier_val"]]),
        k=folds, seed=SEED,
    )
    test = benchmark.test_evaluate_all(
        specs, d["X_tr"], d["y_tr"], d["tier_tr"], d["X_te"], d["y_te"], d["tier_te"]
    )
    benchmark.print_results(test, cv)

    structure = {
        task: benchmark.tree_structure_comparison(
            test[(task, "NumPy DecisionTree")]["model"],
            test[(task, f"sklearn DecisionTree{'Regressor' if task == 'regression' else 'Classifier'}")]["model"],
            d["X_te"], task,
        )
        for task in ("regression", "classification")
    }
    print("\nTree structure vs sklearn (matched hyperparameters, test rows):")
    for report in structure.values():
        print("  " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in report.items()))

    timing = benchmark.timing_study(specs, d["X_tr"], d["y_tr"], d["tier_tr"],
                                    fractions=(0.1, 0.5, 1.0) if fast else (0.1, 0.25, 0.5, 1.0))
    timing_figure = benchmark.plot_timing(timing, str(FIGURES))
    table_path = REPORT_DIR / "results_table.tex"
    table_path.write_text(benchmark.results_table_tex(test, cv) + "\n", encoding="utf-8")

    # ---- 6. error analysis (uses the already-fitted models, no refitting) ---
    ours_reg = test[("regression", "NumPy DecisionTree")]
    ours_clf = test[("classification", "NumPy DecisionTree")]
    svm_key = next(k for k in test if k[0] == "classification" and k[1].startswith("NumPy Pegasos"))
    svm_model = test[svm_key]["model"]
    linear_weights = np.asarray(svm_model.svm_.w) if svm_model.gamma is None else None
    errors = error_analysis.run(
        d["X_tr"], d["X_te"], d["y_te"], d["tier_te"],
        {"NumPy DecisionTree": ours_reg["pred"],
         "sklearn DecisionTreeRegressor": test[("regression", "sklearn DecisionTreeRegressor")]["pred"]},
        {"NumPy DecisionTree": {"pred": ours_clf["pred"], "score": ours_clf["score"]},
         svm_key[1]: {"pred": test[svm_key]["pred"], "score": test[svm_key]["score"]}},
        d["threshold"], out_dir=str(FIGURES), feature_names=d["feature_names"],
        tree_importances=ours_clf["model"].feature_importances_, svm_weights=linear_weights,
    )
    print("\nError analysis (test set):")
    for band in errors["price_quantile_bands"]["NumPy DecisionTree"]:
        print(f"  price {band['band']:<8} n={band['n']:>5}  MAE={band['mae']:>10,.0f}  bias={band['bias']:>11,.0f}")
    sd = errors["sparse_vs_dense"]
    print(f"  median rel. error  sparse districts: {sd['sparse']['median_rel_error']:.3f}  dense: {sd['dense']['median_rel_error']:.3f}")

    # ---- 7. report artefacts -------------------------------------------------
    tree_test = tree_experiments.final_test_scores(
        (d["X_tr"], d["y_tr"], d["X_val"], d["y_val"], d["X_te"], d["y_te"]), tree_results
    )
    tree_experiments.write_latex(tree_results, tree_test, out_dir=str(REPORT_DIR))

    results = {
        "seed": SEED, "mode": "fast" if fast else "full", "folds": folds,
        "split_sizes": {"train": len(d["y_tr"]), "validation": len(d["y_val"]), "test": len(d["y_te"])},
        "tier_threshold": d["threshold"], "final_configs": configs,
        "test": test, "cv": cv, "tree_structure": structure, "timing": timing,
        "tree_experiments": tree_results, "svm_experiments": svm_results, "error_analysis": errors,
        "figures": {**eda_figures, "timing": timing_figure},
    }
    if args.bonus:
        results["bonus"] = run_bonus()
        print("\nBonus modules:", results["bonus"])
    (RESULTS_DIR / "results.json").write_text(json.dumps(_jsonable(results), indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {table_path}, {RESULTS_DIR / 'results.json'} and figures in {FIGURES}/")
    return results


if __name__ == "__main__":
    main()
