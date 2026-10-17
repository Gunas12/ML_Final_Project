"""BONUS P9 (Member 3): AdaBoost from scratch with small trees of OUR DecisionTree.

Binary AdaBoost (labels -> {-1,+1}). Each round resamples the training set with
probabilities equal to the current sample weights (rather than implementing
weighted impurity), fits a depth-limited tree, computes the weighted error on
the ORIGINAL training rows and updates the weights:

    alpha_t = 0.5 ln((1 - err_t) / err_t),   w_i <- w_i exp(-alpha_t y_i h_t(x_i)),  normalise.
"""
from __future__ import annotations

from inspect import signature
from pathlib import Path
from time import perf_counter

import numpy as np

from . import evaluate as ev
from .decision_tree import DecisionTree

SEED = 42


class AdaBoost:
    """Binary discrete AdaBoost with weighted bootstrap trees (not exact SAMME).

    sklearn's comparison fits exact sample-weighted splits instead of bootstrap
    samples. Both use discrete boosting, but identical predictions are not expected.
    """

    def __init__(self, n_estimators=50, max_depth=1, random_state=SEED):
        self.n_estimators, self.max_depth, self.random_state = n_estimators, max_depth, random_state
        self.trees_, self.alphas_, self.classes_ = [], [], None
        self.n_features_in_ = None

    def _validate_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite two-dimensional matrix")
        return X

    def fit(self, X, y):
        for name in ("n_estimators", "max_depth"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.trees_, self.alphas_, self.classes_ = [], [], None
        self.n_features_in_ = None
        X, y = self._validate_X(X), np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional and match X's row count")
        if y.dtype.kind in "fc" and not np.isfinite(y).all():
            raise ValueError("y must contain finite labels")
        if any(value is None or value != value or
               (isinstance(value, (float, complex, np.number)) and not np.isfinite(value))
               for value in y):
            raise ValueError("y must not contain missing or infinite labels")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("AdaBoost here is binary")
        ypm = np.where(y == classes[1], 1.0, -1.0)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        w = np.full(n, 1.0 / n)
        for _ in range(self.n_estimators):
            # A bootstrap can be unlucky. Retry without erasing learned weights.
            for _attempt in range(10):
                idx = rng.choice(n, size=n, replace=True, p=w)
                if len(np.unique(ypm[idx])) < 2:
                    continue
                tree = DecisionTree("classification", "gini", max_depth=self.max_depth,
                                    random_state=int(rng.integers(2**31 - 1))).fit(X[idx], ypm[idx])
                pred = tree.predict(X)
                err = float(np.sum(w[pred != ypm]))
                if err < 0.5:
                    break
            else:
                if not self.trees_:
                    raise ValueError("No weak learner performed better than chance")
                break
            clipped_err = np.clip(err, 1e-12, 1.0 - 1e-12)
            alpha = 0.5 * np.log((1.0 - clipped_err) / clipped_err)
            self.trees_.append(tree)
            self.alphas_.append(float(alpha))
            if err <= 1e-12:  # perfect learner: avoid infinite weight and stop
                break
            # Log-space normalisation avoids numerical overflow/underflow.
            log_w = np.log(np.maximum(w, np.finfo(float).tiny)) - alpha * ypm * pred
            w = np.exp(log_w - log_w.max())
            w /= w.sum()
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        return self

    def staged_decision_function(self, X):
        if self.classes_ is None or not self.trees_:
            raise ValueError("AdaBoost must be fitted before prediction")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X feature count must match the training matrix")
        total = np.zeros(len(X))
        for tree, alpha in zip(self.trees_, self.alphas_):
            total = total + alpha * tree.predict(X)
            yield total.copy()

    def decision_function(self, X):
        out = np.zeros(len(np.asarray(X)))
        for out in self.staged_decision_function(X):
            pass
        return out

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, self.classes_[1], self.classes_[0])


def _sklearn_reference(X_tr, y_tr, X_te, y_te, n_estimators, depth, seed):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    kwargs = dict(estimator=DecisionTreeClassifier(max_depth=depth, criterion="gini"),
                  n_estimators=n_estimators, learning_rate=1.0, random_state=seed)
    # Older sklearn defaults to SAMME.R; newer releases support only SAMME
    # and have removed the algorithm keyword entirely.
    if "algorithm" in signature(AdaBoostClassifier).parameters:
        kwargs["algorithm"] = "SAMME"
    m = AdaBoostClassifier(**kwargs).fit(X_tr, y_tr)
    return ev.score_classification(y_te, m.predict(X_te), m.decision_function(X_te))


def run(fast: bool = False, data_path: str | None = None, out_dir: str | None = None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import data_prep
    from .run_all import prepare_data

    d = prepare_data(data_path or data_prep.DATA_PATH, fast)
    n_est = 20 if fast else 100
    depths = [1, 3]
    curves, val_f1, fit_times = {}, {}, {}
    models = {}
    for depth in depths:
        start = perf_counter()
        m = AdaBoost(n_est, depth).fit(d["X_tr"], d["tier_tr"])
        fit_times[depth] = perf_counter() - start
        models[depth] = m
        tr_err = [float(np.mean(np.where(s > 0, m.classes_[1], m.classes_[0]) == d["tier_tr"])) for s in m.staged_decision_function(d["X_tr"])]
        va_err = [float(np.mean(np.where(s > 0, m.classes_[1], m.classes_[0]) == d["tier_val"])) for s in m.staged_decision_function(d["X_val"])]
        curves[depth] = {"train_error": [1 - a for a in tr_err], "val_error": [1 - a for a in va_err]}
        val_f1[depth] = ev.precision_recall_f1(d["tier_val"], m.predict(d["X_val"]))[2]
    best_depth = max(val_f1, key=val_f1.get)            # chosen on VALIDATION only
    best = models[best_depth]
    test_scores = ev.score_classification(d["tier_te"], best.predict(d["X_te"]), best.decision_function(d["X_te"]))
    sk = _sklearn_reference(d["X_tr"], d["tier_tr"], d["X_te"], d["tier_te"], n_est, best_depth, SEED)

    out = Path(out_dir or ("results/fast/figures" if fast else "report/figures"))
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for depth in depths:
        ax.plot(range(1, len(curves[depth]["val_error"]) + 1), curves[depth]["val_error"], label=f"val, depth {depth}")
        ax.plot(range(1, len(curves[depth]["train_error"]) + 1), curves[depth]["train_error"], "--", label=f"train, depth {depth}")
    ax.set(xlabel="boosting rounds", ylabel="error (1 - accuracy)", title="AdaBoost, Task B")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    figure = out / "fig_bonus_boosting.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)

    rows = [[f"AdaBoost d={best_depth} (ours)", "B", "F1", test_scores["F1"]],
            [f"AdaBoost d={best_depth} (ours)", "B", "ROC-AUC", test_scores["ROC-AUC"]],
            [f"sklearn AdaBoost d={best_depth}", "B", "F1", sk["F1"]]]
    return {"n_estimators": n_est, "n_estimators_fitted": len(best.trees_),
            "best_depth": best_depth, "val_f1_by_depth": val_f1, "curves": curves,
            "fit_seconds_by_depth": fit_times, "reference_algorithm": "SAMME",
            "weighting": "weighted bootstrap (ours), exact sample weights (sklearn)",
            "test": test_scores, "sklearn_test": sk, "figure": str(figure), "summary_rows": rows}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Member 3: NumPy AdaBoost bonus and sklearn comparison")
    parser.add_argument("--fast", action="store_true", help="development subset, not final report results")
    parser.add_argument("--data", default=None, help="raw bina.az CSV path")
    args = parser.parse_args()
    result = run(fast=args.fast, data_path=args.data)
    output = Path("results/fast" if args.fast else "results") / "bonus_boosting.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected depth: {result['best_depth']}; rounds: {result['n_estimators_fitted']}")
    print(f"NumPy AdaBoost test: {result['test']}")
    print(f"sklearn AdaBoost (SAMME) test: {result['sklearn_test']}")
    print(f"Saved: {output}")
