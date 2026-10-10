"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

STATUS (step 1 of the plan): impurity functions + best-split search are
implemented. _grow / _predict_one / predict / predict_proba are the NEXT steps.

Conventions
-----------
* A split sends a sample LEFT if  x[feature] <= threshold,  otherwise RIGHT.
* "Impurity decrease" of a split at a node with n samples (N = training-set size):

      decrease = (n / N) * ( I(parent) - n_L/n * I(left) - n_R/n * I(right) )

  i.e. the node-level gain weighted by the node's share of the training data.
  This is the same definition scikit-learn uses for `min_impurity_decrease`
  and it makes `feature_importances_` (sum of decreases per feature) easy.
"""

from __future__ import annotations

import numpy as np


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    _VALID_CRITERIA = {"classification": ("gini", "entropy"), "regression": ("mse",)}

    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None
        # set in fit():
        self._n_total: int | None = None      # size of the training set (for weighting)
        self.classes_ = None                  # sorted class labels (classification)
        self.n_features_: int | None = None

    # --- parameter validation -----------------------------------------------
    def _check_params(self) -> None:
        if self.task not in self._VALID_CRITERIA:
            raise ValueError(f"task must be 'classification' or 'regression', got {self.task!r}")
        if self.criterion not in self._VALID_CRITERIA[self.task]:
            raise ValueError(
                f"criterion {self.criterion!r} is not valid for task {self.task!r}; "
                f"use one of {self._VALID_CRITERIA[self.task]}"
            )
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be None or >= 1")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if self.min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be >= 0")

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """
        Impurity of ONE node.
          gini     : 1 - sum_k p_k^2
          entropy  : - sum_k p_k * log2(p_k)      (0*log0 treated as 0)
          mse      : mean((y - mean(y))^2)        (variance)
        """
        y = np.asarray(y)
        if y.size == 0:
            return 0.0
        if self.task == "regression":
            y = y.astype(float)
            return float(np.mean((y - y.mean()) ** 2))
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()                  # every p > 0 here
        if self.criterion == "gini":
            return float(1.0 - np.sum(p * p))
        return float(-np.sum(p * np.log2(p)))

    def _impurity_from_counts(self, counts: np.ndarray, n: np.ndarray) -> np.ndarray:
        """
        Vectorised classification impurity.
        counts: (m, K) class counts of m candidate child nodes; n: (m,) their sizes.
        """
        p = counts / n[:, None]
        if self.criterion == "gini":
            return 1.0 - np.sum(p * p, axis=1)
        safe = np.where(p > 0, p, 1.0)             # log2(1) = 0  ->  0*log0 = 0
        return -np.sum(p * np.log2(safe), axis=1)

    # --- best split ---------------------------------------------------------
    def _best_split(self, X, y):
        """
        Search all features & candidate thresholds; return the split that
        maximizes impurity decrease as (feature, threshold, decrease),
        or None if NO valid split exists.

        Candidate thresholds are midpoints between consecutive DISTINCT sorted
        values of a feature. A split is valid only if both children have at
        least `min_samples_leaf` samples. We sort each feature once and use
        cumulative class counts (classification) or cumulative sums of y and
        y^2 (regression), so one feature costs O(n log n), not O(n^2).

        Note: a valid split with zero gain is still returned (scikit-learn does
        the same); `_grow` decides with `min_impurity_decrease` whether to use it.
        Ties are broken deterministically: lowest feature index, then lowest threshold.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, n_features = X.shape
        if n < 2:
            return None
        n_total = self._n_total if self._n_total is not None else n

        parent_imp = self._impurity(y)
        if parent_imp <= 0.0:                      # pure node: nothing left to separate
            return None

        classification = self.task == "classification"
        if classification:
            classes, y_enc = np.unique(y, return_inverse=True)
            y_enc = y_enc.reshape(-1)
            k = len(classes)
            total_counts = np.bincount(y_enc, minlength=k).astype(float)
        else:
            y = y.astype(float)
            y = y - y.mean()                       # centering avoids cancellation in E[y^2]-E[y]^2
            tot, tot2 = y.sum(), (y ** 2).sum()

        # position i means: first i+1 sorted samples go left
        n_left = np.arange(1, n)
        n_right = n - n_left
        size_ok = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
        if not size_ok.any():
            return None

        best = None
        best_gain = -np.inf
        for j in range(n_features):
            order = np.argsort(X[:, j], kind="mergesort")
            xs = X[order, j]
            valid = (xs[:-1] < xs[1:]) & size_ok   # never split between equal values
            if not valid.any():
                continue

            if classification:
                onehot = np.zeros((n, k))
                onehot[np.arange(n), y_enc[order]] = 1.0
                left = np.cumsum(onehot, axis=0)[:-1]
                right = total_counts - left
                imp_l = self._impurity_from_counts(left, n_left)
                imp_r = self._impurity_from_counts(right, n_right)
            else:
                ys = y[order]
                cs = np.cumsum(ys)[:-1]
                cs2 = np.cumsum(ys ** 2)[:-1]
                mean_l = cs / n_left
                mean_r = (tot - cs) / n_right
                imp_l = np.maximum(cs2 / n_left - mean_l ** 2, 0.0)
                imp_r = np.maximum((tot2 - cs2) / n_right - mean_r ** 2, 0.0)

            child_imp = (n_left * imp_l + n_right * imp_r) / n
            gain = np.where(valid, parent_imp - child_imp, -np.inf)
            i = int(np.argmax(gain))               # first maximum -> lowest threshold on ties
            if gain[i] > best_gain:                # strict '>' -> lowest feature index on ties
                thr = (xs[i] + xs[i + 1]) / 2.0
                if thr >= xs[i + 1]:               # float rounding guard (adjacent floats)
                    thr = xs[i]
                best_gain = float(gain[i])
                best = (j, float(thr))

        if best is None:
            return None
        decrease = max(best_gain, 0.0) * n / n_total
        return best[0], best[1], float(decrease)

    # --- grow / predict (NEXT STEPS) ----------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        raise NotImplementedError("TODO: recursive grow")

    def fit(self, X, y):
        self._check_params()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self._n_total = X.shape[0]
        self.n_features_ = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        raise NotImplementedError("TODO: traverse to a leaf")

    def predict(self, X):
        raise NotImplementedError("TODO: predict")

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        raise NotImplementedError("TODO: predict_proba (classification)")