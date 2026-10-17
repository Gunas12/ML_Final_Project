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

**Mammadli, Esmira —** Extended `src/evaluate.py` (PR-AUC, k-fold and
stratified k-fold, `cross_validate`, timing helper) on top of Punhan's basic
metrics (RMSE, MAE, R2, confusion matrix, precision/recall/F1, ROC-AUC), and
integrated the matched scikit-learn baselines (`src/baselines.py`), the
benchmark with one test evaluation and timing (`src/benchmark.py`), the error
analysis (`src/error_analysis.py`), the single entry point (`src/run_all.py`
with `--fast` and `--bonus`), the ridge / epsilon-SVR bonus
(`src/bonus_linear.py`), their tests, and the pinned requirements. I ran,
tested, debugged and committed these files, ran the final `python -m
src.run_all --bonus` (250 tests passed), and checked the report and slide
numbers against that run. **The first versions of these files, the ridge/SVR
bonus and the first drafts of the report and slides were written by Claude
(see AI assistance); I did not write them from scratch.**
Report: Introduction, Experimental Setup, the error analysis in the Discussion,
Conclusion, Tools and the merge of the team's sections; slides 5-7.
Bonus code I can explain at the defence: [Esmira: keep `bonus_linear.py` only if true].

## AI assistance
Claude (Anthropic) was used for:
- the first versions of `src/baselines.py`, `src/benchmark.py`,
  `src/error_analysis.py`, the `src/run_all.py` pipeline, the extensions of
  `src/evaluate.py` (PR-AUC, k-fold, cross-validation), `src/bonus_linear.py`
  (ridge, epsilon-SVR), and their tests (`tests/test_baselines.py`,
  `tests/test_benchmark.py`, `tests/test_error_analysis.py`,
  `tests/test_bonus_models.py`, `tests/test_run_all.py`,
  `tests/test_core_has_no_sklearn.py`), all written with Member 4, who ran,
  tested and committed them;
- the final repository review, shared pipeline fixes, and drafts of the report,
  the slides and this file.

OpenAI Codex also assisted Member 3 with the SVM/AdaBoost review, integration,
safety fixes, regression tests, real-data verification and documentation.
The initial joint-grid and AdaBoost drafts were supplied by the teammate.

Members 1 and 2: [add your own AI use, or write "none"].

## Bonus ownership (who explains it at the defence)
| Bonus | Code | Member who explains it |
|-------|------|------------------------|
| P8 Ridge / SVR | `src/bonus_linear.py` | [ ] |
| P9 Random forest | `src/bonus_ensembles.py` | [ ] |
| P9 AdaBoost | `src/bonus_boosting.py` | [ ] |
| P10 PCA / K-means / DBSCAN | `src/bonus_clustering.py` | [ ] |

## Effort allocation and signatures
Percentages are agreed by the team; each member checks their own paragraph
above and signs it themselves.

| Member | Effort (%) | Signature | Date |
|--------|-----------:|-----------|------|
| Mursalov, Punhan | [ ] | | |
| Mammadova, Gunash | [ ] | | |
| Muradov, Ali | [ ] | | |
| Mammadli, Esmira | [ ] | | |
