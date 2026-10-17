"""BONUS P8 (Member 4): Ridge regression and epsilon-SVR, both FROM SCRATCH (NumPy).

* RidgeScratch   closed form  w = (Xc'Xc + lambda I)^-1 Xc'yc  (intercept is not penalised)
* SVRScratch     epsilon-insensitive loss, L2 penalty, Pegasos-style sub-gradient steps

Both are Task A (price) models. They predict log(price) from standardised
features and are scored in AZN after exp(). lambda / epsilon are chosen on the
VALIDATION split only; the test split is scored once with the final setting.
scikit-learn appears only inside `_sklearn_reference` as a sanity check.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import evaluate as ev

SEED = 42


class RidgeScratch:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        gram = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.w = np.linalg.solve(gram, Xc.T @ yc)
        self.b = float(y_mean - x_mean @ self.w)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b


class SVRScratch:
    """min  lambda/2 ||w||^2 + 1/n sum max(0, |y - (w.x + b)| - eps)   (sub-gradient)."""

    def __init__(self, lambda_: float = 1e-3, epsilon: float = 0.1, n_iters: int = 50_000,
                 eta0: float = 1.0, random_state: int = SEED):
        self.lambda_, self.epsilon, self.n_iters = lambda_, epsilon, n_iters
        self.eta0, self.random_state = eta0, random_state
        self.w = None
        self.b = 0.0
        self.loss_history_ = []

    def _objective(self, X, y):
        resid = np.abs(y - (X @ self.w + self.b))
        return 0.5 * self.lambda_ * float(self.w @ self.w) + float(np.mean(np.maximum(0.0, resid - self.epsilon)))

    def fit(self, X, y):
        if self.lambda_ <= 0 or self.epsilon < 0 or self.n_iters < 1:
            raise ValueError("lambda_ > 0, epsilon >= 0 and n_iters >= 1 are required")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.w, self.b = np.zeros(X.shape[1]), float(np.mean(y))
        # eta_t = 1/(lambda (t + t0)); t0 makes eta_1 = eta0 so the first steps stay finite
        t0 = max(1.0 / (self.lambda_ * self.eta0) - 1.0, 0.0)
        every = max(1, self.n_iters // 50)
        self.loss_history_ = [self._objective(X, y)]
        for t in range(1, self.n_iters + 1):
            eta = 1.0 / (self.lambda_ * (t + t0))
            i = int(rng.integers(len(y)))
            resid = y[i] - (X[i] @ self.w + self.b)
            self.w *= 1.0 - eta * self.lambda_
            if abs(resid) > self.epsilon:
                sign = 1.0 if resid > 0 else -1.0
                self.w += eta * sign * X[i]
                self.b += eta * sign
            if t % every == 0:
                self.loss_history_.append(self._objective(X, y))
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b


def _scores(price_true, std_pred, mu, sd, lo, hi):
    """Score in AZN. A linear model can extrapolate wildly on extreme feature values and
    exp() would then overflow, so the standardised log-prediction is clipped to the range
    seen in TRAIN targets (lo, hi) before it is turned back into a price."""
    return ev.score_regression(price_true, np.exp(np.clip(std_pred, lo, hi) * sd + mu))


def _sklearn_reference(X, y_log, alpha, our_w):
    """Sanity check only: our closed form must agree with sklearn's Ridge."""
    from sklearn.linear_model import Ridge
    ref = Ridge(alpha=alpha).fit(X, y_log)
    return float(np.max(np.abs(ref.coef_ - our_w)))


def run(fast: bool = False, data_path: str | None = None, out_dir: str = "report/figures") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import data_prep
    from .decision_tree import DecisionTree
    from .run_all import prepare_data

    d = prepare_data(data_path or data_prep.DATA_PATH, fast)
    s_tr, s_val, s_te = d["scaled"]
    y_tr, y_val, y_te = d["y_tr"], d["y_val"], d["y_te"]
    ly_tr = np.log(y_tr)
    mu, sd = float(ly_tr.mean()), float(ly_tr.std())
    t_tr = (ly_tr - mu) / sd                       # standardised log-price (train statistics only)
    lo, hi = float(t_tr.min()), float(t_tr.max())

    # ---- ridge: lambda sweep on validation ---------------------------------
    alphas = [0.01, 0.1, 1, 10, 100, 1_000, 10_000]
    ridge_val = []
    for a in alphas:
        m = RidgeScratch(a).fit(s_tr, t_tr)
        ridge_val.append(_scores(y_val, m.predict(s_val), mu, sd, lo, hi)["RMSE_log"])
    best_alpha = alphas[int(np.argmin(ridge_val))]
    ridge = RidgeScratch(best_alpha).fit(s_tr, t_tr)
    ridge_test = _scores(y_te, ridge.predict(s_te), mu, sd, lo, hi)
    ridge_val_scores = _scores(y_val, ridge.predict(s_val), mu, sd, lo, hi)
    agreement = _sklearn_reference(s_tr, t_tr, best_alpha, ridge.w)

    # ---- SVR: lambda sweep on validation -----------------------------------
    n_iters = 5_000 if fast else 60_000
    lambdas = [1e-3, 1e-2, 1e-1]
    svr_val = []
    for lam in lambdas:
        m = SVRScratch(lam, 0.1, n_iters).fit(s_tr, t_tr)
        svr_val.append(_scores(y_val, m.predict(s_val), mu, sd, lo, hi)["RMSE_log"])
    best_lambda = lambdas[int(np.argmin(svr_val))]
    svr = SVRScratch(best_lambda, 0.1, n_iters).fit(s_tr, t_tr)
    svr_test = _scores(y_te, svr.predict(s_te), mu, sd, lo, hi)

    # ---- reference: our tree (Task A) with the same split -------------------
    tree = DecisionTree("regression", "mse", max_depth=8, min_samples_leaf=10, random_state=SEED)
    tree_test = ev.score_regression(y_te, tree.fit(d["X_tr"], y_tr).predict(d["X_te"]))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].semilogx(alphas, ridge_val, marker="o")
    ax[0].set(xlabel="ridge alpha (log scale)", ylabel="validation RMSE (log price)", title="Ridge: alpha sweep")
    ax[1].plot(np.linspace(0, n_iters, len(svr.loss_history_)), svr.loss_history_)
    ax[1].set(xlabel="Pegasos-style step", ylabel="objective", title=f"SVR objective (lambda={best_lambda:g})")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    figure = out / "fig_bonus_linear.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)

    rows = [["Ridge (ours)", "A", "RMSE", ridge_test["RMSE"]], ["Ridge (ours)", "A", "R2", ridge_test["R2"]],
            ["SVR (ours)", "A", "RMSE", svr_test["RMSE"]], ["SVR (ours)", "A", "R2", svr_test["R2"]],
            ["Tree (ours)", "A", "RMSE", tree_test["RMSE"]], ["Tree (ours)", "A", "R2", tree_test["R2"]]]
    return {"best_ridge_alpha": best_alpha, "best_svr_lambda": best_lambda, "ridge_test": ridge_test,
            "ridge_val": ridge_val_scores, "svr_test": svr_test, "tree_test": tree_test,
            "max_abs_coef_gap_vs_sklearn": agreement, "figure": str(figure), "summary_rows": rows}
