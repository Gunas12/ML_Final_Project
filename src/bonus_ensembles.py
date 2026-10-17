"""BONUS P9 : Random Forest built from OUR decision tree.

Bootstrap sampling + a random feature subset at every split (DecisionTree
max_features) + majority vote / mean. Out-of-bag (OOB) score included.
Used for Task B (classification, mean of leaf probabilities) and Task A
(regression, mean of tree predictions). sklearn is imported only inside
`_sklearn_reference` to compare against RandomForest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import evaluate as ev
from .decision_tree import DecisionTree

SEED = 42


class RandomForest:
    def __init__(self, task="classification", n_trees=30, max_depth=None, min_samples_leaf=1,
                 max_features=None, criterion=None, random_state=SEED):
        self.task, self.n_trees, self.max_depth = task, n_trees, max_depth
        self.min_samples_leaf, self.max_features = min_samples_leaf, max_features
        self.criterion = criterion or ("gini" if task == "classification" else "mse")
        self.random_state = random_state
        self.trees_, self.oob_idx_, self.classes_, self.oob_score_ = [], [], None, None

    def _k(self, p):
        if self.max_features is not None:
            return int(self.max_features)
        return max(1, int(np.sqrt(p))) if self.task == "classification" else max(1, p // 3)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self.trees_, self.oob_idx_ = [], []
        for t in range(self.n_trees):
            boot = rng.integers(0, n, n)
            oob = np.setdiff1d(np.arange(n), boot)
            tree = DecisionTree(self.task, self.criterion, self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                random_state=int(rng.integers(2**31 - 1)), max_features=self._k(p))
            self.trees_.append(tree.fit(X[boot], y[boot]))
            self.oob_idx_.append(oob)
        self._compute_oob(X, y)
        return self

    def _tree_output(self, tree, X):
        if self.task == "regression":
            return tree.predict(X)
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, np.searchsorted(self.classes_, tree.classes_)] = tree.predict_proba(X)
        return proba

    def _compute_oob(self, X, y):
        shape = (len(X),) if self.task == "regression" else (len(X), len(self.classes_))
        total, votes = np.zeros(shape), np.zeros(len(X))
        for tree, idx in zip(self.trees_, self.oob_idx_):
            if len(idx):
                total[idx] += self._tree_output(tree, X[idx])
                votes[idx] += 1
        seen = votes > 0
        avg = total[seen] / votes[seen].reshape((-1,) + (1,) * (total.ndim - 1))
        if self.task == "regression":
            self.oob_score_ = ev.r2(y[seen], avg)
        else:
            self.oob_score_ = float(np.mean(self.classes_[np.argmax(avg, axis=1)] == y[seen]))

    def staged_outputs(self, X):
        """Running mean of the first m trees, m = 1..n_trees (for error-vs-trees curves)."""
        X = np.asarray(X, dtype=float)
        running, out = 0.0, []
        for m, tree in enumerate(self.trees_, start=1):
            running = running + self._tree_output(tree, X)
            out.append(running / m)
        return out

    def predict_proba(self, X):
        return self.staged_outputs(X)[-1]

    def predict(self, X):
        avg = self.staged_outputs(X)[-1]
        return avg if self.task == "regression" else self.classes_[np.argmax(avg, axis=1)]


def _sklearn_reference(X_tr, y_tr, X_te, y_te, task, n_trees, min_leaf, seed, criterion="gini"):
    if task == "classification":
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(n_estimators=n_trees, max_features="sqrt", min_samples_leaf=min_leaf,
                                   criterion=criterion, random_state=seed, n_jobs=1).fit(X_tr, y_tr)
        return ev.score_classification(y_te, m.predict(X_te), m.predict_proba(X_te)[:, 1])
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=n_trees, max_features=1 / 3, min_samples_leaf=min_leaf,
                              random_state=seed, n_jobs=1).fit(X_tr, y_tr)
    return ev.score_regression(y_te, m.predict(X_te))


def run(fast: bool = False, data_path: str | None = None, out_dir: str = "report/figures") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import data_prep
    from .run_all import prepare_data

    d = prepare_data(data_path or data_prep.DATA_PATH, fast)
    n_trees = 8 if fast else 30
    leaf_b, leaf_a = 1, 3

    # ---- Task B ---------------------------------------------------------------
    clf = RandomForest("classification", n_trees, min_samples_leaf=leaf_b, criterion="entropy").fit(d["X_tr"], d["tier_tr"])
    val_stage = clf.staged_outputs(d["X_val"])
    val_err = [1.0 - float(np.mean(clf.classes_[np.argmax(p, axis=1)] == d["tier_val"])) for p in val_stage]
    single = DecisionTree("classification", "entropy", min_samples_leaf=leaf_b, random_state=SEED).fit(d["X_tr"], d["tier_tr"])
    te_proba = clf.predict_proba(d["X_te"])[:, 1]
    forest_b = ev.score_classification(d["tier_te"], clf.predict(d["X_te"]), te_proba)
    tree_b = ev.score_classification(d["tier_te"], single.predict(d["X_te"]), single.predict_proba(d["X_te"])[:, 1])
    sk_b = _sklearn_reference(d["X_tr"], d["tier_tr"], d["X_te"], d["tier_te"], "classification", n_trees, leaf_b, SEED, "entropy")  # matched criterion

    # ---- Task A ---------------------------------------------------------------
    reg = RandomForest("regression", n_trees, min_samples_leaf=leaf_a).fit(d["X_tr"], d["y_tr"])
    val_stage_a = reg.staged_outputs(d["X_val"])
    val_rmse = [ev.rmse(d["y_val"], p) for p in val_stage_a]
    forest_a = ev.score_regression(d["y_te"], reg.predict(d["X_te"]))
    sk_a = _sklearn_reference(d["X_tr"], d["y_tr"], d["X_te"], d["y_te"], "regression", n_trees, leaf_a, SEED)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(range(1, n_trees + 1), val_err, marker="o", ms=3)
    ax[0].set(xlabel="number of trees", ylabel="validation error (1 - accuracy)", title="Random Forest, Task B")
    ax[1].plot(range(1, n_trees + 1), val_rmse, marker="o", ms=3)
    ax[1].set(xlabel="number of trees", ylabel="validation RMSE (AZN)", title="Random Forest, Task A")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    figure = out / "fig_bonus_forest.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)

    rows = [["Forest (ours)", "B", "F1", forest_b["F1"]], ["Forest (ours)", "B", "ROC-AUC", forest_b["ROC-AUC"]],
            ["sklearn RF", "B", "F1", sk_b["F1"]], ["Tree (ours)", "B", "F1", tree_b["F1"]],
            ["Forest (ours)", "A", "RMSE", forest_a["RMSE"]], ["Forest (ours)", "A", "R2", forest_a["R2"]],
            ["sklearn RF", "A", "RMSE", sk_a["RMSE"]]]
    return {"n_trees": n_trees, "oob_accuracy_B": clf.oob_score_, "oob_r2_A": reg.oob_score_,
            "val_error_by_trees_B": val_err, "val_rmse_by_trees_A": val_rmse,
            "forest_B": forest_b, "tree_B": tree_b, "sklearn_B": sk_b,
            "forest_A": forest_a, "sklearn_A": sk_a, "figure": str(figure), "summary_rows": rows}
