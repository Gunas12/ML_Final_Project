from __future__ import annotations

from pathlib import Path
from time import perf_counter

from . import data_prep, evaluate
from .baselines import fit_baselines
from .decision_tree import DecisionTree
from .svm import PegasosSVM


SEED = 42


def _regression_metrics(y_true, y_pred):
    return {
        "RMSE": evaluate.rmse(y_true, y_pred),
        "MAE": evaluate.mae(y_true, y_pred),
        "R2": evaluate.r2(y_true, y_pred),
    }


def _classification_metrics(y_true, y_pred, scores):
    precision, recall, f1 = evaluate.precision_recall_f1(y_true, y_pred)
    return {
        "Accuracy": evaluate.accuracy(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": evaluate.roc_auc(y_true, scores),
    }


def _plot_results(output_directory, regression_results, classification_results, svm):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    regression_names = list(regression_results)
    axes[0].bar(
        regression_names,
        [regression_results[name]["RMSE"] for name in regression_names],
    )
    axes[0].set_title("Test RMSE")
    axes[0].tick_params(axis="x", rotation=25)
    classifier_names = list(classification_results)
    axes[1].bar(
        classifier_names,
        [classification_results[name]["F1"] for name in classifier_names],
    )
    axes[1].set_title("Test F1")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=35)
    figure.tight_layout()
    comparison_path = output_directory / "fig_model_comparison.png"
    figure.savefig(comparison_path, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(svm.loss_steps_, svm.loss_history_, color="tab:blue")
    axis.set_xlabel("Pegasos iteration")
    axis.set_ylabel("Regularized hinge-loss objective")
    axis.set_title("Pegasos convergence")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    convergence_path = output_directory / "fig_pegasos_convergence.png"
    figure.savefig(convergence_path, dpi=150)
    plt.close(figure)
    return {
        "model_comparison": str(comparison_path),
        "pegasos_convergence": str(convergence_path),
    }


def main() -> dict:
    raw = data_prep.load_raw()
    cleaned = data_prep.clean(raw)
    X, y_price, feature_names = data_prep.make_features(cleaned)
    X_train, y_train, X_val, y_val, X_test, y_test = (
        data_prep.train_val_test_split(X, y_price, seed=SEED)
    )
    y_tier_train, tier_threshold = data_prep.make_tier_label(y_train)
    y_tier_val, _ = data_prep.make_tier_label(y_val, threshold=tier_threshold)
    y_tier_test, _ = data_prep.make_tier_label(y_test, threshold=tier_threshold)
    X_train_scaled, X_val_scaled, X_test_scaled = data_prep.standardize(
        X_train, X_val, X_test
    )

    regression_tree = DecisionTree(
        task="regression",
        criterion="mse",
        max_depth=8,
        min_samples_leaf=100,
        random_state=SEED,
    )
    classification_tree = DecisionTree(
        task="classification",
        criterion="gini",
        max_depth=8,
        min_samples_leaf=100,
        random_state=SEED,
    )
    start = perf_counter()
    regression_tree.fit(X_train, y_train)
    regression_tree_seconds = perf_counter() - start
    start = perf_counter()
    classification_tree.fit(X_train, y_tier_train)
    classification_tree_seconds = perf_counter() - start

    svm = PegasosSVM(lambda_=1e-3, n_iters=10_000, random_state=SEED)
    start = perf_counter()
    svm.fit(X_train_scaled, y_tier_train)
    svm_seconds = perf_counter() - start

    baseline_models = fit_baselines(
        X_train, y_train, y_tier_train, random_state=SEED
    )
    regression_results = {
        "NumPy DecisionTree": _regression_metrics(
            y_test, regression_tree.predict(X_test)
        )
    }
    classification_results = {
        "NumPy DecisionTree": _classification_metrics(
            y_tier_test,
            classification_tree.predict(X_test),
            classification_tree.predict_proba(X_test)[:, 1],
        ),
        "NumPy PegasosSVM": _classification_metrics(
            y_tier_test,
            svm.predict(X_test_scaled),
            svm.decision_function(X_test_scaled),
        ),
    }

    baseline_regressor = baseline_models["DecisionTreeRegressor"]
    regression_results["sklearn DecisionTreeRegressor"] = _regression_metrics(
        y_test, baseline_regressor.predict(X_test)
    )
    for name in (
        "DecisionTreeClassifier",
        "StandardScaler+SVC",
        "StandardScaler+SGD hinge",
    ):
        model = baseline_models[name]
        predictions = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)
        classification_results[f"sklearn {name}"] = _classification_metrics(
            y_tier_test, predictions, scores
        )

    validation_results = {
        "regression_tree": _regression_metrics(
            y_val, regression_tree.predict(X_val)
        ),
        "classification_tree": _classification_metrics(
            y_tier_val,
            classification_tree.predict(X_val),
            classification_tree.predict_proba(X_val)[:, 1],
        ),
        "pegasos_svm": _classification_metrics(
            y_tier_val,
            svm.predict(X_val_scaled),
            svm.decision_function(X_val_scaled),
        ),
    }

    print(f"Data rows: {len(y_price):,}")
    print(f"Features after one-hot encoding: {len(feature_names):,}")
    print(f"Split sizes: train={len(y_train):,}, validation={len(y_val):,}, test={len(y_test):,}")
    print(f"Training price median: {tier_threshold:,.2f} AZN")
    print("\nValidation metrics")
    evaluate.compare(
        {
            "NumPy DecisionTree regression": validation_results["regression_tree"],
            "NumPy DecisionTree classification": validation_results[
                "classification_tree"
            ],
            "NumPy PegasosSVM": validation_results["pegasos_svm"],
        }
    )
    print("\nTest regression metrics")
    evaluate.compare(regression_results)
    print("\nTest classification metrics")
    evaluate.compare(classification_results)
    print(
        "\nTraining time (seconds): "
        f"regression tree={regression_tree_seconds:.2f}, "
        f"classification tree={classification_tree_seconds:.2f}, "
        f"Pegasos SVM={svm_seconds:.2f}"
    )

    figures = _plot_results(
        Path("report") / "figures",
        regression_results,
        classification_results,
        svm,
    )
    return {
        "seed": SEED,
        "feature_names": feature_names,
        "tier_threshold": tier_threshold,
        "validation": validation_results,
        "test_regression": regression_results,
        "test_classification": classification_results,
        "models": {
            "regression_tree": regression_tree,
            "classification_tree": classification_tree,
            "pegasos_svm": svm,
            "baselines": baseline_models,
        },
        "figures": figures,
    }


if __name__ == "__main__":
    main()
