# Contribution Report

This report summarizes each team member's responsibilities and contributions
to the machine learning final project.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Mursalov, Punhan | Punhan0 | Data preparation, leakage cleaning, EDA, data documentation, clustering bonus |
| Mammadova, Gunash | Gunas12 | Decision tree from scratch, tree experiments, tree tests |
| Muradov, Ali | Ali-Muradov | Pegasos SVM, random Fourier features, SVM experiments, AdaBoost bonus review and tests |
| Mammadli, Esmira | Esmira | Metrics, baselines, CV and timing, error analysis, `run_all`, pinned requirements |

## Per-member summary
**Mursalov, Punhan —** Implemented the data loading, cleaning, feature
construction, tier-label, stratified split, and train-only standardization
workflow in `src/data_prep.py`. Added exploratory plots in `src/eda.py`,
data-preparation tests, and documentation of leakage exclusions and cleaning
choices in `docs/data_decisions.md`, `data/README.md`, and `README.md`. Wrote
the PCA / k-means / DBSCAN bonus in `src/bonus_clustering.py`.
Report: sections II and III, data limitations; slides 1-2.

**Mammadova, Gunash —** Implemented Gini, entropy and MSE impurity,
the cumulative-count best-split search, recursive growth with the four stopping
rules, `predict`, `predict_proba` and impurity-based feature importances in
`src/decision_tree.py`, with tests against scikit-learn in
`tests/test_decision_tree.py`. Ran the depth / leaf sweeps and Gini-vs-entropy
experiment in `src/tree_experiments.py`. Report: Method A, Results A, tree
limitations; slide 3.

**Muradov, Ali —** Implemented label encoding, the
Pegasos training loop with objective tracking, `margin_width`, `support_mask`
and the random Fourier feature map in `src/svm.py`, and the validation-based
experiment pipeline in `src/svm_experiments.py`. Report: Method B, Results B,
SVM limitations; slide 4.

Also reviewed and integrated the teammate-provided joint
lambda/gamma RFF grid and AdaBoost draft, added regression tests for independent
RFF selection and consistent iteration budgets, and hardened AdaBoost input
validation, weight updates, retries and early stopping. The bonus uses the
team's NumPy DecisionTree; sklearn is only a comparison. Verified SVM and
AdaBoost on the real dataset and tested both Python 3.11 and the team's pinned
Python 3.14 environment.

**Mammadli, Esmira —** Implemented the NumPy metrics, k-fold / stratified k-fold and cross-validation
in `src/evaluate.py`, the matched sklearn baselines (`src/baselines.py`), the
benchmark with a single test evaluation and timing (`src/benchmark.py`), the
error analysis (`src/error_analysis.py`), the single entry point
(`src/run_all.py`), and the pinned requirements. Report: Introduction,
Experimental Setup, Results C, Discussion, Conclusion and merge; slides 5-7.

## AI assistance
Claude assisted with the final repository review, bonus-model and joint RFF-grid
drafts, shared pipeline fixes, README, report and slides.

OpenAI Codex also assisted Member 3 with the SVM/AdaBoost review, integration,
safety fixes, regression tests, real-data verification and documentation.
The initial joint-grid and AdaBoost drafts were supplied by the teammate.

## Effort allocation and signatures

Individual effort percentages and signatures are pending team confirmation.
