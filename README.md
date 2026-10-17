# MLE-AI-201 Machine Learning Final Project

Price prediction on bina.az real-estate sale listings with two models written
from scratch in NumPy, a **decision tree** (CART) and a **Pegasos SVM** with
random Fourier features, compared with matched scikit-learn baselines.

- **Task A, regression:** predict the listing price (AZN).
- **Task B, classification:** predict `premium` (1) or `standard` (0); premium
  means the price is strictly above the **training-split median** (210,000 AZN).
  The same threshold labels validation and test rows.

Team: Esmira Mammadli, Gunash Mammadova, Ali Muradov, Punhan Mursalov
(who did what: [contribution_report.md](contribution_report.md)).

## Headline results (test set, scored once, seed 42)

| Model | Task A RMSE (AZN) | Task B F1 | Task B ROC-AUC |
|-------|------------------:|----------:|---------------:|
| NumPy DecisionTree (ours) | 358,672 | 0.919 | 0.920 |
| sklearn DecisionTree | 358,999 | 0.918 | 0.919 |
| NumPy Pegasos SVM + RFF (ours) | n/a | 0.838 | 0.924 |
| sklearn SVC (linear) | n/a | 0.839 | 0.923 |
| sklearn SGD (hinge) | n/a | 0.831 | 0.905 |
| sklearn RBFSampler + SGD | n/a | 0.833 | 0.924 |

5-fold CV on train+validation (mean ± std): tree RMSE 438,746 ± 29,122; tree
F1 0.915 ± 0.003; our SVM F1 0.843 ± 0.005 (fit 5.3 s per fold) versus
sklearn linear SVC 0.843 ± 0.008 (fit 245.4 s per fold). Full tables and
discussion are in the report (`report/report.tex`). Optional bonus models
(ridge / epsilon-SVR, random forest, AdaBoost, PCA / k-means / DBSCAN) are
reported separately.

## Dataset

Kaggle **binaaz-sale-project** by `sehriyarmemmedli`
(<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>).
Place the CSV at `data/bina_az_sale.csv` (UTF-8). The file is not committed;
see [data/README.md](data/README.md). The supplied copy has 100,775 rows and 51
columns; after cleaning 64,350 listings and 171 features remain (split
45,045 / 9,652 / 9,653). Every cleaning step is logged in
[docs/data_decisions.md](docs/data_decisions.md). `unit_price` and
`total_price` are derived from the target and are removed as leakage; IDs,
URLs, free text, scrape time, view counts and currency labels are dropped too.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` is pinned from the tested environment. scikit-learn is
used only for baselines, tests and bonus reference comparisons, never inside
the tree, the SVM or the metrics (checked by `tests/test_core_has_no_sklearn.py`).

## Running

```bash
pytest -q                      # all tests (250) should pass
python -m src.run_all          # all numbers, tables and figures (full run)
python -m src.run_all --bonus  # plus bonus models (run after the core part works)
python -m src.run_all --fast   # small subset for development only
```

The full run is deterministic (seed 42 in `data_prep.SEED`) and writes:
figures to `report/figures/` (`fig_*.png`), LaTeX tables to
`report/results_table.tex`, `report/tree_tables.tex`, `report/tree_macros.tex`
and `report/bonus_table.tex`, and all numbers to `results/results.json`. The
SVC baseline dominates the run time (about 4 minutes per fold). Figures are
not committed, so run `run_all` before building the report.

Build the report and slides after the run:

```bash
cd report
pdflatex report.tex && pdflatex report.tex     # report/report.pdf
pdflatex slides.tex && pdflatex slides.tex     # slides, copy to presentation/slides.pdf
```

Stand-alone clustering bonus: `python src/bonus_clustering.py` (options
`--clusters`, `--data`, `--figures`).

## Repository structure

```text
.
├── data/README.md            dataset instructions (CSV itself is not committed)
├── docs/data_decisions.md    every cleaning decision with row counts
├── presentation/slides.pdf   final slides
├── report/                   report.tex, slides.tex, generated tables, figures/
├── results/results.json      all numbers of the last full run
├── scripts/pin_requirements.py
├── src/
│   ├── data_prep.py          load, clean, features, split, tier label, standardize
│   ├── eda.py                EDA figures
│   ├── decision_tree.py      CART tree (Gini / entropy / MSE), NumPy only
│   ├── svm.py                Pegasos SVM + random Fourier features, NumPy only
│   ├── evaluate.py           metrics, k-fold, cross-validation, timing
│   ├── baselines.py          matched scikit-learn baselines
│   ├── benchmark.py          single test evaluation, CV table, timing
│   ├── tree_experiments.py   depth / leaf sweeps, Gini vs entropy, learning curve
│   ├── svm_experiments.py    lambda / gamma sweeps, convergence, margin
│   ├── error_analysis.py     errors by price band, district, near-threshold
│   ├── run_all.py            one-command pipeline
│   └── bonus_*.py            ridge/SVR, random forest, AdaBoost, clustering
├── tests/                    pytest suite
├── contribution_report.md
└── requirements.txt
```

## Rules followed

- Tree, SVM and metrics are our own NumPy code; sklearn is a baseline only.
- Median threshold, scaler statistics and all hyperparameters come from
  train / validation only; the test split is used once at the end.
- Every random operation uses seed 42; two runs give identical results.
