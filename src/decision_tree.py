"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

STATUS: impurity, best split, recursive growth, prediction, predict_proba,
feature_importances_, depth_/n_leaves_, export_text are implemented.

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
import pandas as pd


class Node:
    """One node of the tree."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node   (x[feature] <= threshold)
        self.right = None        # Node   (x[feature] >  threshold)
        self.value = None        # classification: class COUNTS (K,)   regression: mean
        self.n_samples = 0       # training samples that reached this node
        self.impurity = 0.0      # impurity of this node
        self.decrease = 0.0      # weighted impurity decrease of this node's split (internal only)

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

    @staticmethod
    def _validate_target(y, *, task: str):
        values = np.asarray(y)
        if values.ndim != 1:
            raise ValueError("y must be a one-dimensional array")
        if values.size == 0:
            raise ValueError("y must not be empty")
        if np.any(pd.isna(values)):
            raise ValueError("y must not contain NaN or missing values")
        if task == "regression":
            try:
                values = values.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("y must be numeric for regression") from exc
            if not np.all(np.isfinite(values)):
                raise ValueError("y must not contain NaN or infinite values")
            return values
        try:
            numeric = values.astype(float)
        except (TypeError, ValueError):
            numeric = None
        else:
            if not np.all(np.isfinite(numeric)):
                raise ValueError("classification labels must be finite")
        try:
            np.unique(values)
        except TypeError as exc:
            raise ValueError("classification labels must be mutually comparable") from exc
        return values

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        """
        Impurity of ONE node.
          gini     : 1 - sum_k p_k^2
          entropy  : - sum_k p_k * log2(p_k)      (0*log0 treated as 0)
          mse      : mean((y - mean(y))^2)        (variance)
        """
        y = self._validate_target(y, task=self.task)
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
        y = self._validate_target(y, task=self.task)
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

    # --- grow ---------------------------------------------------------------
    def _make_node(self, y) -> Node:
        node = Node()
        node.n_samples = int(y.shape[0])
        node.impurity = self._impurity(y)
        if self.task == "classification":
            node.value = np.bincount(y, minlength=len(self.classes_)).astype(float)
        else:
            node.value = float(np.mean(y))
        return node

    def _grow(self, X, y, depth: int) -> Node:
        """
        Recursively build the tree. A node becomes a LEAF when
          * depth >= max_depth                      (if max_depth is set)
          * n < min_samples_split
          * n < 2 * min_samples_leaf                (no valid split can exist)
          * the node is pure (impurity == 0)
          * no valid split exists
          * the best split's decrease < min_impurity_decrease
        For classification `y` holds integer class codes 0..K-1 (see fit).
        """
        node = self._make_node(y)
        n = node.n_samples
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n < self.min_samples_split
            or n < 2 * self.min_samples_leaf
            or node.impurity <= 0.0
        ):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node
        feature, threshold, decrease = split
        if decrease + 1e-12 < self.min_impurity_decrease:
            return node

        mask = X[:, feature] <= threshold
        node.feature, node.threshold, node.decrease = feature, threshold, decrease
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        self._check_params()
        X = np.asarray(X, dtype=float)
        y = self._validate_target(y, task=self.task)
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] == 0:
            raise ValueError("cannot fit on an empty dataset")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinity; clean the data first")
        self._n_total = X.shape[0]
        self.n_features_ = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
            y_fit = np.searchsorted(self.classes_, y)      # labels -> codes 0..K-1
        else:
            y_fit = y.astype(float)
        self.root = self._grow(X, y_fit, depth=0)
        return self

    # --- predict ------------------------------------------------------------
    def _check_fitted(self, X) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("this DecisionTree is not fitted yet; call fit(X, y) first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(f"X must have shape (n, {self.n_features_})")
        return X

    def _predict_one(self, x, node: Node):
        """Walk ONE sample down to a leaf and return the leaf value."""
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def _leaf_values(self, X) -> np.ndarray:
        """Leaf value for every row of X (all rows are routed down together)."""
        n = X.shape[0]
        k = len(self.classes_) if self.task == "classification" else None
        out = np.empty((n, k)) if k else np.empty(n)
        stack = [(self.root, np.arange(n))]
        while stack:
            node, idx = stack.pop()
            if idx.size == 0:
                continue
            if node.is_leaf():
                out[idx] = node.value
                continue
            go_left = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[go_left]))
            stack.append((node.right, idx[~go_left]))
        return out

    def predict(self, X):
        X = self._check_fitted(X)
        vals = self._leaf_values(X)
        if self.task == "classification":
            return self.classes_[np.argmax(vals, axis=1)]   # ties -> smallest class
        return vals

    def predict_proba(self, X):
        """Classification only: class probabilities = leaf class counts / leaf size."""
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification")
        X = self._check_fitted(X)
        counts = self._leaf_values(X)
        return counts / counts.sum(axis=1, keepdims=True)

    # --- inspection ---------------------------------------------------------
    def _nodes(self):
        """All nodes (depth-first), as (node, depth) pairs."""
        if self.root is None:
            return []
        out, stack = [], [(self.root, 0)]
        while stack:
            node, d = stack.pop()
            out.append((node, d))
            if not node.is_leaf():
                stack.append((node.right, d + 1))
                stack.append((node.left, d + 1))
        return out

    @property
    def depth_(self) -> int:
        """Depth of the deepest leaf (a single-leaf tree has depth 0)."""
        return max((d for _, d in self._nodes()), default=0)

    @property
    def n_nodes_(self) -> int:
        return len(self._nodes())

    @property
    def n_leaves_(self) -> int:
        return sum(1 for node, _ in self._nodes() if node.is_leaf())

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        Impurity-decrease importance: sum of the (weighted) decreases of all
        splits on a feature, normalised to sum to 1. All zeros if no split.
        """
        if self.root is None:
            raise RuntimeError("this DecisionTree is not fitted yet; call fit(X, y) first")
        imp = np.zeros(self.n_features_)
        for node, _ in self._nodes():
            if not node.is_leaf():
                imp[node.feature] += node.decrease
        total = imp.sum()
        return imp / total if total > 0 else imp

    def export_text(self, feature_names=None, max_depth: int | None = None) -> str:
        """Human-readable rules (useful for the report and for debugging)."""
        if self.root is None:
            raise RuntimeError("this DecisionTree is not fitted yet; call fit(X, y) first")
        names = feature_names or [f"x{j}" for j in range(self.n_features_)]
        lines = []

        def leaf_text(node):
            if self.task == "classification":
                return f"class: {self.classes_[int(np.argmax(node.value))]}  (n={node.n_samples}, counts={node.value.astype(int).tolist()})"
            return f"value: {node.value:.4g}  (n={node.n_samples})"

        def rec(node, depth):
            pad = "|   " * depth
            if node.is_leaf() or (max_depth is not None and depth >= max_depth):
                lines.append(f"{pad}|--- {leaf_text(node)}")
                return
            f, thr = names[node.feature], node.threshold
            lines.append(f"{pad}|--- {f} <= {thr:.4g}")
            rec(node.left, depth + 1)
            lines.append(f"{pad}|--- {f} >  {thr:.4g}")
            rec(node.right, depth + 1)

        rec(self.root, 0)
        return "\n".join(lines)

    def __repr__(self) -> str:
        params = (f"task={self.task!r}, criterion={self.criterion!r}, max_depth={self.max_depth}, "
                  f"min_samples_split={self.min_samples_split}, min_samples_leaf={self.min_samples_leaf}, "
                  f"min_impurity_decrease={self.min_impurity_decrease}")
        if self.root is None:
            return f"DecisionTree({params}) [not fitted]"
        return f"DecisionTree({params}) [depth={self.depth_}, leaves={self.n_leaves_}, nodes={self.n_nodes_}]"