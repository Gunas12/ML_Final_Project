"""Fair benchmark of our NumPy models against matched scikit-learn baselines.

Fairness rules enforced here
----------------------------
* the same train / validation / test rows for every model;
* hyperparameters of the sklearn baselines are matched to ours (baselines.py);
* cross-validation uses ONLY train+validation rows; the test split is scored
  exactly once, with the final configurations;
* scalers are fitted on training rows only (inside each CV fold as well);
* the tier threshold (training median) is fixed before any labelling.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone

from . import baselines, data_prep
from . import evaluate as ev
from .decision_tree import DecisionTree
from .svm import PegasosSVM, RandomFourierFeatures

SEED = 42


# --------------------------------------------------------------------------
# glue: our SVM (+ optional random Fourier features) behind one fit/predict API
# --------------------------------------------------------------------------
class RFFPegasosSVM:
    """Random Fourier features followed by our Pegasos solver.

    Thin composition of Member 3's two classes so the RBF variant can be used
    wherever a single estimator is expected (cross-validation, timing).
    With gamma=None it is just the linear PegasosSVM.
    """

    def __init__(self, lambda_=1e-3, n_iters=20_000, gamma=None, n_components=256, random_state=SEED):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if self.gamma is not None:
            self.rff_ = RandomFourierFeatures(self.n_components, self.gamma, self.random_state)
            X = self.rff_.fit_transform(X)
        self.svm_ = PegasosSVM(self.lambda_, self.n_iters, self.random_state).fit(X, y)
        return self

    def _map(self, X):
        X = np.asarray(X, dtype=float)
        return self.rff_.transform(X) if self.gamma is not None else X

    def decision_function(self, X):
        return self.svm_.decision_function(self._map(X))

    def predict(self, X):
        return self.svm_.predict(self._map(X))


# --------------------------------------------------------------------------
# model specifications
# --------------------------------------------------------------------------
def _standardize_pair(X_train, X_val):
    return data_prep.standardize(X_train, X_val)


def build_specs(configs: dict, n_train: int, seed: int = SEED) -> list:
    """One spec per model: name, group (ours/sklearn), task, factory, scale.

    configs = {"tree_regression": {...}, "tree_classification": {...,
               "criterion": ...}, "svm": {"lambda_": ..., "n_iters": ...,
               optional "gamma", "n_components"}}
    """
    tree_reg = {k: v for k, v in configs["tree_regression"].items() if k != "criterion"}
    tree_clf = {k: v for k, v in configs["tree_classification"].items() if k != "criterion"}
    svm = configs["svm"]
    skl = baselines.build_baselines(
        configs["tree_regression"], configs["tree_classification"], svm, n_train, seed
    )
    svm_label = "NumPy Pegasos SVM" + (" + RFF" if svm.get("gamma") is not None else "")
    specs = [
        {"name": "NumPy DecisionTree", "group": "ours", "task": "regression", "scale": False,
         "factory": lambda: DecisionTree("regression", "mse", random_state=seed, **tree_reg)},
        {"name": "sklearn DecisionTreeRegressor", "group": "sklearn", "task": "regression", "scale": False,
         "factory": lambda: clone(skl["DecisionTreeRegressor"])},
        {"name": "NumPy DecisionTree", "group": "ours", "task": "classification", "scale": False,
         "factory": lambda: DecisionTree("classification", configs["tree_classification"].get("criterion", "gini"),
                                         random_state=seed, **tree_clf)},
        {"name": "sklearn DecisionTreeClassifier", "group": "sklearn", "task": "classification", "scale": False,
         "factory": lambda: clone(skl["DecisionTreeClassifier"])},
        {"name": svm_label, "group": "ours", "task": "classification", "scale": True,
         "factory": lambda: RFFPegasosSVM(svm["lambda_"], svm.get("n_iters", 20_000), svm.get("gamma"),
                                          svm.get("n_components", 256), seed)},
    ]
    for key, label in (
        ("StandardScaler+SVC(linear)", "sklearn SVC(linear)"),
        ("StandardScaler+SGD(hinge)", "sklearn SGD(hinge)"),
        ("StandardScaler+RBFSampler+SGD(hinge)", "sklearn RBFSampler+SGD"),
    ):
        if key in skl:
            specs.append(
                {"name": label, "group": "sklearn", "task": "classification", "scale": False,
                 "factory": (lambda k=key: clone(skl[k]))}
            )
    return specs


# --------------------------------------------------------------------------
# cross-validation and the single test evaluation
# --------------------------------------------------------------------------
def cross_validate_all(specs, X_trval, y_price, y_tier, k: int = 5, seed: int = SEED) -> dict:
    """k-fold CV (mean +/- std) of every spec on train+validation rows only."""
    results = {}
    for spec in specs:
        target = y_price if spec["task"] == "regression" else y_tier
        results[(spec["task"], spec["name"])] = ev.cross_validate(
            spec["factory"], X_trval, target, k=k, seed=seed, task=spec["task"],
            preprocess=_standardize_pair if spec["scale"] else None,
        )
    return results


def test_evaluate_all(specs, X_train, y_price_train, y_tier_train, X_test, y_price_test, y_tier_test) -> dict:
    """Fit on the training split and score the TEST split once per model.

    Returns {(task, name): {"metrics", "fit_seconds", "predict_seconds",
    "pred", "score", "model"}}; predictions are kept for error analysis.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    X_train_scaled, X_test_scaled = data_prep.standardize(X_train, X_test)
    out = {}
    for spec in specs:
        train, test = (X_train_scaled, X_test_scaled) if spec["scale"] else (X_train, X_test)
        model = spec["factory"]()
        if spec["task"] == "regression":
            _, fit_s = ev.timeit(model.fit, train, y_price_train)
            pred, pred_s = ev.timeit(model.predict, test)
            metrics, score = ev.score_regression(y_price_test, pred), None
        else:
            _, fit_s = ev.timeit(model.fit, train, y_tier_train)
            pred, pred_s = ev.timeit(model.predict, test)
            score = ev.positive_scores(model, test)
            metrics = ev.score_classification(y_tier_test, pred, score)
        out[(spec["task"], spec["name"])] = {
            "metrics": metrics, "fit_seconds": fit_s, "predict_seconds": pred_s,
            "pred": np.asarray(pred), "score": score, "model": model,
        }
    return out


# --------------------------------------------------------------------------
# tree structure comparison (P5a)
# --------------------------------------------------------------------------
def tree_structure_comparison(ours, theirs, X, task: str) -> dict:
    """Do our tree and sklearn's tree look and behave alike on the same data?"""
    X = np.asarray(X, dtype=float)
    a, b = np.asarray(ours.predict(X)), np.asarray(theirs.predict(X))
    return {
        "task": task,
        "our_leaves": int(ours.n_leaves_),
        "sklearn_leaves": int(theirs.get_n_leaves()),
        "our_depth": int(ours.depth_),
        "sklearn_depth": int(theirs.get_depth()),
        "prediction_agreement": float(np.mean(np.isclose(a, b))),
        "max_abs_prediction_gap": float(np.max(np.abs(a - b))),
    }


# --------------------------------------------------------------------------
# timing study
# --------------------------------------------------------------------------
def timing_study(specs, X_train, y_price, y_tier, fractions=(0.1, 0.25, 0.5, 1.0), seed: int = SEED) -> dict:
    """Fit time as the training set grows (same random subsets for all models)."""
    X = np.asarray(X_train, dtype=float)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    rows = {}
    for spec in specs:
        target = y_price if spec["task"] == "regression" else y_tier
        series = []
        for fraction in fractions:
            idx = order[: max(int(len(X) * fraction), 10)]
            train = X[idx]
            if spec["scale"]:
                train = data_prep.standardize(train)[0]
            _, seconds = ev.timeit(spec["factory"]().fit, train, target[idx])
            series.append({"n": int(len(idx)), "fit_seconds": float(seconds)})
        rows[f"{spec['task']}: {spec['name']}"] = series
    return rows


def plot_timing(timing: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for name, series in timing.items():
        axis.plot([r["n"] for r in series], [r["fit_seconds"] for r in series], marker="o", label=name)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("training rows (log scale)")
    axis.set_ylabel("fit time (s, log scale)")
    axis.set_title("Fit time vs. training-set size")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=6)
    figure.tight_layout()
    path = os.path.join(out_dir, "fig_timing_scaling.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# results table (console + LaTeX)
# --------------------------------------------------------------------------
def _pm(cv, name, digits=3, scale=1.0):
    return f"{cv['mean'][name] * scale:.{digits}f} $\\pm$ {cv['std'][name] * scale:.{digits}f}"


def _escape(text: str) -> str:
    return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def results_table_tex(test: dict, cv: dict) -> str:
    """Two booktabs tables: Task A (price) and Task B (tier), test + CV."""
    lines = ["% AUTO-GENERATED by benchmark.results_table_tex -- do not edit by hand."]
    reg = [key for key in test if key[0] == "regression"]
    clf = [key for key in test if key[0] == "classification"]
    lines += [
        r"\begin{table*}[t]", r"  \centering",
        r"  \caption{Task A (price regression). Test metrics from one fit on the training split; "
        r"CV columns are mean $\pm$ std over folds on train+validation. RMSE and MAE in AZN.}",
        r"  \label{tab:results_task_a}", r"  \begin{tabular}{@{}lrrrrr@{}}", r"    \toprule",
        r"    Model & Test RMSE & Test MAE & Test $R^2$ & CV RMSE & CV $R^2$ \\", r"    \midrule",
    ]
    for key in reg:
        m, c = test[key]["metrics"], cv[key]
        lines.append(
            f"    {_escape(key[1])} & {m['RMSE']:,.0f} & {m['MAE']:,.0f} & {m['R2']:.3f} & "
            f"{_pm(c, 'RMSE', 0)} & {_pm(c, 'R2')} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}", ""]
    lines += [
        r"\begin{table*}[t]", r"  \centering",
        r"  \caption{Task B (price tier). Test metrics plus cross-validated mean $\pm$ std "
        r"(train+validation).}",
        r"  \label{tab:results_task_b}", r"  \begin{tabular}{@{}lrrrrrr@{}}", r"    \toprule",
        r"    Model & Test F1 & Test ROC-AUC & Test PR-AUC & CV F1 & CV ROC-AUC & Fit (s) \\", r"    \midrule",
    ]
    for key in clf:
        m, c = test[key]["metrics"], cv[key]
        lines.append(
            f"    {_escape(key[1])} & {m['F1']:.3f} & {m['ROC-AUC']:.3f} & {m['PR-AUC']:.3f} & "
            f"{_pm(c, 'F1')} & {_pm(c, 'ROC-AUC')} & {test[key]['fit_seconds']:.2f} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def print_results(test: dict, cv: dict) -> None:
    """Console version of the same two tables."""
    for task, title, test_keys in (
        ("regression", "Task A (price)", ["RMSE", "MAE", "R2", "RMSE_log", "R2_log"]),
        ("classification", "Task B (price tier)", ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]),
    ):
        keys = [k for k in test if k[0] == task]
        print(f"\n{title} - TEST set (single evaluation)")
        ev.compare({k[1]: test[k]["metrics"] for k in keys})
        headline = "RMSE" if task == "regression" else "F1"
        print(f"{title} - {cv[keys[0]]['k']}-fold CV on train+validation: {headline} mean +/- std")
        for k in keys:
            c = cv[k]
            print(f"  {k[1]:<34} {c['mean'][headline]:>12.4f} +/- {c['std'][headline]:.4f}   "
                  f"fit {c['fit_seconds']:.2f}s  predict {c['predict_seconds']:.3f}s")
