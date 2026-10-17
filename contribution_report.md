# Contribution Report

This file summarizes who did what in the MLE-AI-201 final project
(bina.az price regression and price-tier classification). The claims below
can be checked against `git log` / `git shortlog -sn --all` (Git author names:
Esmira / Esmira Mammadli, Punhan0, Ali Muradov / Ali-Muradov, Gunas12 /
Gunash Mammadova).

## Team

| Member (Surname, Name) | GitHub handle | Main responsibilities                                                                                                                                                            |
| ---------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mursalov, Punhan       | Punhan0       | Data loading and cleaning, leakage columns, features, stratified split, tier label, standardization, EDA, data documentation, README, PCA / k-means / DBSCAN bonus               |
| Mammadova, Gunash      | Gunas12       | Decision tree from scratch (impurity, split search, growth, importances), tree tests, tree experiments, random forest bonus                                                      |
| Muradov, Ali           | Ali-Muradov   | Pegasos SVM, random Fourier features, margin / support-vector diagnostics, SVM experiments and tests, AdaBoost bonus                                                             |
| Mammadli, Esmira       | Esmira        | Metrics and cross-validation, sklearn baselines, benchmark and timing, error analysis, `run_all`, pinned requirements, ridge / SVR bonus, merging report and slides, release tag |

## Per-member summary

**Mursalov, Punhan —** Implemented the whole data workflow in
`src/data_prep.py`: loading, cleaning (one row per listing, area-unit
conversion, plausibility filters), feature construction, the median tier label
(training split only), the stratified 70/15/15 split and train-only
standardization. Wrote the EDA figures in
`src/eda.py`, `tests/test_data_prep.py`, the cleaning log `docs/data_decisions.md`,
`data/README.md` and the project `README.md`, and the PCA / k-means / DBSCAN
bonus in `src/bonus_clustering.py`. Report: sections II and III and
the data limitations; slides 1-2.

**Mammadova, Gunash —** Implemented Gini, entropy and MSE impurity, the
cumulative-count best-split search, recursive growth with the four stopping
rules, `predict`, `predict_proba` and impurity-based feature importances in
`src/decision_tree.py`, with tests against scikit-learn
in `tests/test_decision_tree.py`. Ran the depth and
leaf sweeps and the Gini-vs-entropy comparison in `src/tree_experiments.py` and wrote the random forest bonus in `src/bonus_ensembles.py`. Report: Method A, Results A, tree limitations; slide 3.

**Muradov, Ali —** Implemented label encoding, the Pegasos training loop with
objective tracking, `margin_width`, `support_mask` and the random Fourier
feature map in `src/svm.py`, and the
validation-based experiments (lambda sweep, gamma sweep, convergence, margin
plots) in `src/svm_experiments.py`. Added the joint lambda/gamma RFF
grid and the AdaBoost bonus in `src/bonus_boosting.py` with their tests
(`tests/test_svm.py`, `tests/test_svm_joint_grid.py`,
`tests/test_bonus_boosting.py`). Report: Method B, Results B, SVM limitations;
slide 4.

**Mammadli, Esmira —** Extended `src/evaluate.py` (PR-AUC, k-fold and
stratified k-fold, `cross_validate`, timing helper) and wrote the matched
scikit-learn baselines (`src/baselines.py`), the benchmark with a single test
evaluation (`src/benchmark.py`), the error analysis (`src/error_analysis.py`),
the one-command pipeline `src/run_all.py` (`--fast`, `--bonus`), the ridge /
epsilon-SVR bonus (`src/bonus_linear.py`), their tests, and the pinned
`requirements.txt`. Ran the final `python -m src.run_all` (250 tests pass),
checked every number in the report and slides against that run, merged all
sections and slides, and created the release tag. Report: Introduction,
Experimental Setup, Results C, Discussion (error analysis), Conclusion, Tools;
slides 5-7.

## AI assistance

AI assistants (Claude by Anthropic, and OpenAI Codex) were used only in a
limited, assisting role: small code corrections and help preparing the LaTeX
files of the report and the slides. The algorithms, experiments, analysis and
conclusions are the team's own work, and each member can explain their own
modules. This matches the "Tools and Acknowledgements" section of the report.

| Member            | AI use                                                               |
| ----------------- | -------------------------------------------------------------------- |
| Mursalov, Punhan  | small corrections only                                               |
| Mammadova, Gunash | small corrections only                                               |
| Muradov, Ali      | small corrections only                                               |
| Mammadli, Esmira  | small corrections and help with the LaTeX files of report and slides |

## Bonus ownership (who explains it at the defence)

| Bonus                      | Code                      | Member who explains it |
| -------------------------- | ------------------------- | ---------------------- |
| P8 Ridge / SVR             | `src/bonus_linear.py`     | Mammadli, Esmira       |
| P9 Random forest           | `src/bonus_ensembles.py`  | Mammadova, Gunash      |
| P9 AdaBoost                | `src/bonus_boosting.py`   | Muradov, Ali           |
| P10 PCA / k-means / DBSCAN | `src/bonus_clustering.py` | Mursalov, Punhan       |

## Rough effort split

Proposed split, to be confirmed by the team before signing (should sum to
100%). Esmira has the largest share because of integration, evaluation, the
final run and the merge of report and slides.

| Member            | Approx. % |
| ----------------- | --------: |
| Mursalov, Punhan  |        20 |
| Mammadova, Gunash |        20 |
| Muradov, Ali      |        25 |
| Mammadli, Esmira  |        35 |

## Signatures

Each member signs only their own line, with their own git name.

- Signed: **********\_\_********** (Mursalov, Punhan), date: ****\_\_****
- Signed: **********\_\_********** (Mammadova, Gunash), date: ****\_\_****
- Signed: **********\_\_********** (Muradov, Ali), date: ****\_\_****
- Signed: **********\_\_********** (Mammadli, Esmira), date: ****\_\_****
