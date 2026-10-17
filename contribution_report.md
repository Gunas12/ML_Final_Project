# Contribution Report

Keep this file in the repo root. It is checked at the demo. Be honest and
specific — "helped with everything" is not acceptable. One row per member,
plus a short per-member paragraph.

## Team
| Member (Surname, Name) | GitHub handle | Main responsibilities |
|------------------------|---------------|-----------------------|
| Mursalov, Punhan       | —             | Data preparation, leakage cleaning, EDA, documentation |
| Muradov, Ali | Ali-Muradov | Pegasos SVM, RFF selection/tests, AdaBoost bonus review and integration |
| ---                    | ---           | ---                   |
| ---                    | ---           | ---                   |

## Per-member summary
**Mursalov, Punhan —** Implemented the data loading, cleaning, feature
construction, tier-label, stratified split, and train-only standardization
workflow in `src/data_prep.py`. Added exploratory plots in `src/eda.py`,
data-preparation tests, and documentation of leakage exclusions and cleaning
choices in `docs/data_decisions.md`, `data/README.md`, and `README.md`.

**Surname, Name —** ...

**Muradov, Ali — DRAFT, awaiting personal review/signature.** Responsible for
the NumPy Pegasos SVM and RFF map (`src/svm.py`), validation-only experiments
(`src/svm_experiments.py`), and associated tests. Reviewed and integrated the
teammate-provided joint lambda/gamma grid and AdaBoost draft. Added independent
RFF-selection regression tests, a shared SVM iteration-budget contract, and
AdaBoost input checks, stable weight updates, retry/early-stop handling and
discrete sklearn comparison compatibility. AdaBoost uses the team's scratch
DecisionTree, not sklearn for its own training. Updated the Member 3 report
sections and presentation notes and checked real-data results.

AI/provenance disclosure for this update: the teammate reported using Claude
for the initial joint-grid and bonus drafts. OpenAI Codex assisted with review,
integration, fixes, tests, real-data runs and documentation. This is not a claim
that these additions were written manually without assistance. Ali must review
and understand the code before accepting responsibility and signing.

**Surname, Name —** ...

## Rough effort split
Give an approximate percentage per member (should sum to ~100%) and note any
imbalance and why.

| Member          | Approx. % |
|-----------------|-----------|
| Mursalov, Punhan | 25%       |
| Muradov, Ali | To be agreed by the team |
| ---             | ---       |
| ---             | ---       |

Signed: Punhan Mursalov, 14 Oct 2026
