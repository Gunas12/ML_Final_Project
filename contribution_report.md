# Contribution Report

> **DRAFT built from the Git history. Every member must check their own part,
> correct anything that is wrong, agree the percentages with the team, and sign
> with their own name. Nobody may sign for another member.** Delete this note
> after signing.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Mursalov, Punhan | Punhan0 | Data preparation, leakage cleaning, EDA, data documentation, clustering bonus |
| Mammadova, Gunash | Gunas12 | Decision tree from scratch, tree experiments, tree tests |
| Muradov, Ali | Ali-Muradov (commits as "Ali Muradov") | Pegasos SVM, random Fourier features, SVM experiments, AdaBoost bonus review and tests |
| Mammadli, Esmira | Esmira | Metrics, baselines, CV and timing, error analysis, `run_all`, pinned requirements |

## Per-member summary
**Mursalov, Punhan —** Implemented the data loading, cleaning, feature
construction, tier-label, stratified split, and train-only standardization
workflow in `src/data_prep.py`. Added exploratory plots in `src/eda.py`,
data-preparation tests, and documentation of leakage exclusions and cleaning
choices in `docs/data_decisions.md`, `data/README.md`, and `README.md`. Wrote
the PCA / k-means / DBSCAN bonus in `src/bonus_clustering.py` (merged via #6).
Report: sections II and III, data limitations; slides 1-2.

**Mammadova, Gunash —** *(check against your commits: `13f6718`, `03b66c8`,
`772940f`, `49f4aa7`, `a0ca621`)* Implemented Gini, entropy and MSE impurity,
the cumulative-count best-split search, recursive growth with the four stopping
rules, `predict`, `predict_proba` and impurity-based feature importances in
`src/decision_tree.py`, with tests against scikit-learn in
`tests/test_decision_tree.py`. Ran the depth / leaf sweeps and Gini-vs-entropy
experiment in `src/tree_experiments.py`. Report: Method A, Results A, tree
limitations; slide 3.

**Muradov, Ali —** *(check against your commits: `b311521`, `6a438c4`,
`ac76327`, `48c9b37`, `15d678e`, `aabda20`)* Implemented label encoding, the
Pegasos training loop with objective tracking, `margin_width`, `support_mask`
and the random Fourier feature map in `src/svm.py`, and the validation-based
experiment pipeline in `src/svm_experiments.py`. Report: Method B, Results B,
SVM limitations; slide 4.

Latest update (`1187ccf`): reviewed and integrated the teammate-provided joint
lambda/gamma RFF grid and AdaBoost draft, added regression tests for independent
RFF selection and consistent iteration budgets, and hardened AdaBoost input
validation, weight updates, retries and early stopping. The bonus uses the
team's NumPy DecisionTree; sklearn is only a comparison. Verified SVM and
AdaBoost on the real dataset and tested both Python 3.11 and the team's pinned
Python 3.14 environment. Personal review and signature are still required.

**Mammadli, Esmira —** *(check against your commits: `ffbc72b`...`2048fb2`)*
Implemented the NumPy metrics, k-fold / stratified k-fold and cross-validation
in `src/evaluate.py`, the matched sklearn baselines (`src/baselines.py`), the
benchmark with a single test evaluation and timing (`src/benchmark.py`), the
error analysis (`src/error_analysis.py`), the single entry point
(`src/run_all.py`), and the pinned requirements. Report: Introduction,
Experimental Setup, Results C, Discussion, Conclusion and merge; slides 5-7.

## AI assistance (honest disclosure)
An AI assistant (Claude) was used late in the project to review the repository
and to draft: the three extra bonus modules `src/bonus_linear.py`,
`src/bonus_ensembles.py`, `src/bonus_boosting.py` with `tests/test_bonus_models.py`,
the `max_features` option of `DecisionTree`, the joint lambda-gamma RFF sweep in
`src/svm_experiments.py`, the `--fast` output separation in `src/run_all.py`,
the README rewrite, the report text and the slides. **Whoever puts their name on
one of these parts must have read it and be able to explain it at the demo.**

OpenAI Codex also assisted Member 3 with the SVM/AdaBoost review, integration,
safety fixes, regression tests, real-data verification and documentation.
The joint-grid and AdaBoost drafts were supplied by the teammate; this is not
a claim that the additions were written manually without AI assistance.

List here which member takes responsibility for which bonus (P8 Ridge/SVR,
P9 Random Forest and AdaBoost):

- P8 (`bonus_linear.py`): ____________
- P9 forest (`bonus_ensembles.py`): ____________
- P9 AdaBoost (`bonus_boosting.py`): Muradov, Ali (draft; awaiting personal review/signature)

## Rough effort split
Agree these numbers as a team (they should sum to ~100%) and note any imbalance
and why.

| Member            | Approx. % |
|-------------------|-----------|
| Mursalov, Punhan  | ____ %    |
| Mammadova, Gunash | ____ %    |
| Muradov, Ali      | ____ %    |
| Mammadli, Esmira  | ____ %    |

Signed: Punhan Mursalov, ____ Oct 2026 (re-sign after checking the text above)

Signed: ______________________, ____ Oct 2026

Signed: ______________________, ____ Oct 2026

Signed: ______________________, ____ Oct 2026
