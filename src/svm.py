"""
svm.py — a soft-margin SVM built FROM SCRATCH via PEGASOS (NumPy only).

You do NOT need a QP / SMO solver. Pegasos minimises the regularised
hinge-loss primal by stochastic sub-gradient descent:

    J(w, b) = (lambda/2) * ||w||^2
              + (1/n) * sum_i  max(0, 1 - y_i (w . x_i + b))

with labels y_i in {-1, +1}. Per-step update with learning rate
eta_t = 1 / (lambda * t):

    if  y_i (w . x_i + b) < 1:
        w <- (1 - eta_t*lambda) w + eta_t*y_i*x_i      (hinge active)
    else:
        w <- (1 - eta_t*lambda) w                      (only regularise)

(Handle b with its own sub-gradient step; many implementations fold a bias
feature into x instead — your choice, just be consistent.)

Non-linear extension (pick ONE and justify it in the report):
  - kernelised Pegasos (keep per-sample coefficients alpha_i), OR
  - random Fourier features (map X -> phi(X), then run LINEAR Pegasos).

scikit-learn's SVC is a BASELINE only — not allowed inside this class.
Remember to STANDARDIZE features (see data_prep.standardize) before training.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    def __init__(
        self,
        lambda_: float = 1e-4,     # regularisation strength (1/(n*C) flavour)
        n_iters: int = 20_000,     # number of sub-gradient steps
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.w = None              # weight vector
        self.b = 0.0               # bias
        self.classes_ = None       # [negative_label, positive_label]
        self.class_to_pm1_ = None  # original label -> {-1.0, +1.0}

    def _to_pm1(self, y):
        """Encode a one-dimensional binary target as ``{-1.0, +1.0}``.

        ``numpy.unique`` gives the encoding a deterministic order: the first
        class maps to ``-1`` and the second class maps to ``+1``.  The class
        order and the explicit mapping are retained so :meth:`predict` can
        return labels in their original dtype.
        """
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y must be a one-dimensional array of labels")
        if labels.size == 0:
            raise ValueError("y must contain at least one label")

        try:
            classes = np.unique(labels)
        except TypeError as exc:
            raise ValueError("labels must be mutually comparable") from exc

        if classes.size != 2:
            raise ValueError(
                f"PegasosSVM requires exactly two classes; got {classes.size}"
            )

        self.classes_ = classes.copy()
        self.class_to_pm1_ = {
            self.classes_[0].item(): -1.0,
            self.classes_[1].item(): 1.0,
        }
        return np.where(labels == self.classes_[0], -1.0, 1.0)

    def fit(self, X, y):
        """
        Run Pegasos. Sample one (or a mini-batch of) example(s) per step,
        use eta_t = 1/(lambda*t), and apply the update above.
        Return self.
        """
        raise NotImplementedError("TODO: Pegasos training loop")

    def decision_function(self, X):
        """Return the signed margin  w . x + b  for each row."""
        if self.w is None:
            raise RuntimeError("PegasosSVM must be fitted before prediction")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")

        w = np.asarray(self.w, dtype=float)
        if w.ndim != 1:
            raise RuntimeError("fitted weight vector must be one-dimensional")
        if X.shape[1] != w.shape[0]:
            raise ValueError(
                "X has a different number of features than the fitted model "
                f"({X.shape[1]} != {w.shape[0]})"
            )

        return X @ w + float(self.b)

    def predict(self, X):
        """Return class labels (map sign of decision_function back to your labels)."""
        if self.classes_ is None or len(self.classes_) != 2:
            raise RuntimeError("PegasosSVM must be fitted before prediction")

        margins = self.decision_function(X)
        # A point exactly on the hyperplane is assigned to the positive class.
        class_indices = (margins >= 0.0).astype(int)
        return self.classes_[class_indices]
