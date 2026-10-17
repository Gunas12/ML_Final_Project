"""Regression tests for independent RFF lambda/gamma selection."""
import numpy as np
import pytest

from src import run_all, svm_experiments as experiments


@pytest.mark.parametrize("rff_wins", [True, False])
def test_full_grid_selects_correct_lambda_and_scores_test_once(tmp_path, monkeypatch, rff_wins):
    calls, scored = [], []
    X = np.zeros((4, 2))
    splits = (X, np.array([0, 1, 0, 1]), X[:3], np.array([0, 1, 0]),
              X[:2], np.array([0, 1]))

    class Model:
        def predict(self, X):
            return np.ones(len(X))

    class Mapping:
        def transform(self, X):
            return X

    def result(lam, score, gamma=None):
        return dict(lambda_=lam, **{"lambda": lam}, validation_score=score, gamma=gamma,
                    model=Model(), rff=Mapping(), final_loss=0.2, weight_norm=2,
                    margin_width=1, support_count=2, fit_time_seconds=0.01)

    linear = result(0.1, 0.8)

    def fake_linear(*args):
        assert args[4] == [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        assert args[5] == 400_000
        return [linear], linear

    def fake_gamma(*args):
        gammas, components, lam, iters = args[4:8]
        assert components == 1024 and iters == 400_000
        rows = []
        for gamma in gammas:
            calls.append((lam, gamma))
            score = 0.9 if rff_wins and (lam, gamma) == (3e-5, 0.01) else 0.7
            rows.append(result(lam, score, gamma))
        return rows, max(rows, key=lambda row: (row["validation_score"], -row["gamma"]))

    monkeypatch.setattr(experiments, "lambda_sweep", fake_linear)
    monkeypatch.setattr(experiments, "gamma_sweep", fake_gamma)
    for plot in ("plot_convergence", "plot_lambda_results", "plot_gamma_results"):
        monkeypatch.setattr(experiments, plot, lambda data, path: str(path))

    def score(y, prediction):
        scored.append(len(y))
        return 0.6

    out = experiments.run(splits, out_dir=tmp_path, score_fn=score)
    assert len(calls) == len(set(calls)) == len(out["rff_grid"]) == 16
    assert {lam for lam, _ in calls} == {1e-5, 3e-5, 1e-4, 3e-4}
    assert {gamma for _, gamma in calls} == {0.003, 0.01, 0.03, 0.1}
    assert scored == [2]
    assert out["n_iters"] == 400_000
    assert out["best_linear_lambda"] == 0.1
    if rff_wins:
        assert out["selected_model_type"] == "rff"
        assert out["best_lambda"] == 3e-5
        assert out["best_gamma"] == 0.01
        assert all(row["lambda"] == 3e-5 for row in out["gamma_results"])
    else:
        assert out["selected_model_type"] == "linear"
        assert out["best_lambda"] == 0.1 and out["best_gamma"] is None


def test_final_benchmark_reuses_actual_selected_iteration_budget():
    tree = {"task_A": {"final_config": {}},
            "task_B": {"final_config": {}, "final_criterion": "gini"}}
    svm = {"best_lambda": 3e-5, "selected_model_type": "rff", "best_gamma": 0.01,
           "n_components": 1024, "n_iters": 12345}
    assert run_all.final_configs(tree, svm, fast=False)["svm"]["n_iters"] == 12345
    assert run_all.SVM_ITERS is experiments.ITERATION_BUDGETS
