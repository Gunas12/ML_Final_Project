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


class RandomFourierFeatures:
    """Approximate an RBF kernel with a randomized cosine feature map."""

    def __init__(
        self,
        n_components: int = 256,
        gamma: float = 1.0,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self.W_ = None
        self.offset_ = None
        self.n_features_in_ = None

    def fit(self, X):
        """Sample and store the frequencies and offsets for the RFF map."""
        if (
            isinstance(self.n_components, (bool, np.bool_))
            or not isinstance(self.n_components, (int, np.integer))
            or self.n_components <= 0
        ):
            raise ValueError("n_components must be a positive integer")
        if (
            isinstance(self.gamma, (bool, np.bool_))
            or not isinstance(self.gamma, (int, float, np.integer, np.floating))
            or not np.isfinite(self.gamma)
            or self.gamma <= 0
        ):
            raise ValueError("gamma must be a finite positive number")

        X = self._validate_X(X, require_non_empty=True)
        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(2.0 * float(self.gamma))

        self.n_features_in_ = X.shape[1]
        self.W_ = rng.normal(
            loc=0.0,
            scale=scale,
            size=(self.n_features_in_, int(self.n_components)),
        )
        self.offset_ = rng.uniform(
            low=0.0,
            high=2.0 * np.pi,
            size=int(self.n_components),
        )
        return self

    def transform(self, X):
        """Map samples into the fitted random Fourier feature space."""
        if self.W_ is None or self.offset_ is None or self.n_features_in_ is None:
            raise RuntimeError("RandomFourierFeatures must be fitted before transform")

        X = self._validate_X(X, require_non_empty=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the fitted mapping "
                f"({X.shape[1]} != {self.n_features_in_})"
            )

        normalization = np.sqrt(2.0 / int(self.n_components))
        return normalization * np.cos(X @ self.W_ + self.offset_)

    def fit_transform(self, X):
        """Fit the random mapping and transform ``X`` in one call."""
        return self.fit(X).transform(X)

    @staticmethod
    def _validate_X(X, require_non_empty: bool):
        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric values") from exc

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if require_non_empty and (X.shape[0] == 0 or X.shape[1] == 0):
            raise ValueError("X must not be empty")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values")
        return X


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
        self.loss_history_ = []    # objective values at monitored steps
        self.loss_steps_ = []      # step numbers corresponding to the losses

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
        Run single-sample Pegasos with ``eta_t = 1 / (lambda_ * t)``.

        The full objective is recorded at step zero, at approximately 100
        evenly spaced checkpoints, and at the final step.  This keeps loss
        monitoring useful without evaluating the entire training set after
        every stochastic update.

        Return self.
        """
        if (
            isinstance(self.lambda_, (bool, np.bool_))
            or not isinstance(self.lambda_, (int, float, np.integer, np.floating))
            or not np.isfinite(self.lambda_)
            or self.lambda_ <= 0
        ):
            raise ValueError("lambda_ must be a finite positive number")
        if (
            isinstance(self.n_iters, (bool, np.bool_))
            or not isinstance(self.n_iters, (int, np.integer))
            or self.n_iters <= 0
        ):
            raise ValueError("n_iters must be a positive integer")

        try:
            X = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric values") from exc
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix")
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional label array")
        if X.shape[0] == 0 or X.shape[1] == 0 or y.size == 0:
            raise ValueError("X and y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples "
                f"({X.shape[0]} != {y.shape[0]})"
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or infinite values")
        if self._has_non_finite_labels(y):
            raise ValueError("y must not contain NaN or infinite values")

        y_pm1 = self._to_pm1(y)
        n_samples, n_features = X.shape
        lambda_ = float(self.lambda_)

        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0
        self.loss_steps_ = [0]
        self.loss_history_ = [self._objective(X, y_pm1, lambda_)]

        rng = np.random.default_rng(self.random_state)
        checkpoint_interval = max(1, int(self.n_iters) // 100)

        for t in range(1, int(self.n_iters) + 1):
            eta_t = 1.0 / (lambda_ * t)
            sample_idx = int(rng.integers(n_samples))
            x_i = X[sample_idx]
            y_i = y_pm1[sample_idx]
            margin = y_i * (np.dot(self.w, x_i) + self.b)

            self.w *= 1.0 - eta_t * lambda_
            if margin < 1.0:
                self.w += eta_t * y_i * x_i
                self.b += eta_t * y_i

            if t % checkpoint_interval == 0 or t == self.n_iters:
                self.loss_steps_.append(t)
                self.loss_history_.append(self._objective(X, y_pm1, lambda_))

        return self

    @staticmethod
    def _has_non_finite_labels(y) -> bool:
        """Return whether a label array contains NaN, infinity, NaT, or None."""
        if y.dtype.kind in "fc":
            return not bool(np.all(np.isfinite(y)))
        if y.dtype.kind in "mM":
            return bool(np.any(np.isnat(y)))
        if y.dtype.kind != "O":
            return False

        for label in y:
            if label is None:
                return True
            try:
                is_finite = np.isfinite(label)
                if np.ndim(is_finite) == 0 and not bool(is_finite):
                    return True
            except (TypeError, ValueError):
                # Ordinary object/string labels do not define finiteness.
                continue
        return False

    def _objective(self, X, y_pm1, lambda_: float) -> float:
        """Compute the regularised mean hinge-loss objective."""
        margins = y_pm1 * (X @ self.w + self.b)
        hinge_loss = np.maximum(0.0, 1.0 - margins)
        regularization = 0.5 * lambda_ * np.dot(self.w, self.w)
        return float(regularization + np.mean(hinge_loss))

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
