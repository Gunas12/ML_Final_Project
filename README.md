# MLE-AI-201 Machine Learning Final Project

Price prediction and price-tier classification on the bina.az real-estate
dataset. A decision tree (CART) and a Pegasos SVM (with random Fourier
features) are implemented **from scratch in NumPy** and compared with
scikit-learn baselines under matched hyperparameters.

## Tasks

- **Task A, regression:** predict the listing price in AZN (decision tree).
- **Task B, classification:** predict standard (`0`) versus premium (`1`),
  where premium means price strictly above the median price of the *training*
  split. The training threshold is reused for validation and test labels
  (decision tree and SVM).

Split: deterministic 70% / 15% / 15% train/validation/test, stratified by tier,
seed **42** (`data_prep.SEED`). Hyperparameters are chosen on validation only;
the test split is scored once per model with the final configuration.

## Dataset

Download Kaggle's **binaaz-sale-project** dataset by `sehriyarmemmedli` and
place the CSV at `data/bina_az_sale.csv` (see [data/README.md](data/README.md)).
The raw data is excluded from version control.

Cleaning, outlier policy, feature selection and leakage exclusions are in
[docs/data_decisions.md](docs/data_decisions.md). `unit_price` and
`total_price` are target-derived and never used as features. The dump contains
100,775 rows but only 64,454 distinct listings; after de-duplication and
plausibility filters **64,350 listings** remain (171 features).

## Setup

Python 3.12+ (the pins in `requirements.txt` were generated on 3.14).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running

```bash
pytest -q                       # all tests (about 1 minute)
python -m src.run_all           # FULL run: every number, table and figure
python -m src.run_all --fast    # tiny subset, development only
python -m src.run_all --bonus   # additionally run bonus modules that exist
```

- A full run is slow (the from-scratch tree and `SVC(linear)` dominate); expect
  roughly 30-40 minutes (about 50 with `--bonus`).
- `run_all` also produces the EDA figures, so no extra step is needed.
- Full-run outputs: `report/results_table.tex`, `report/tree_tables.tex`,
  `report/tree_macros.tex`, `report/figures/fig_*.png` and
  `results/results.json`. Figures are git-ignored, so run the full pipeline
  before compiling the report.
- `--fast` writes **only** to `results/fast/`, so it can never overwrite the
  report numbers.
- Bonus clustering (PCA, k-means, DBSCAN) can also be run on its own:
  `python src/bonus_clustering.py` (`--clusters`, `--data`, `--figures`).

Build the report after a full run:

```bash
cd report && pdflatex report.tex && pdflatex report.tex
```

## Repository structure

```text
.
├── data/README.md            # dataset instructions (CSV itself is not committed)
├── docs/data_decisions.md    # every cleaning decision and its row impact
├── presentation/             # slides.pdf goes here
├── report/                   # report.tex, generated tables, figures/
├── scripts/pin_requirements.py
├── src/
│   ├── data_prep.py          # load, clean, features, split, tier label, standardize
│   ├── eda.py
│   ├── decision_tree.py      # CART from scratch (Gini / entropy / MSE)
│   ├── svm.py                # Pegasos + random Fourier features from scratch
│   ├── evaluate.py           # metrics, k-fold, cross-validation (own NumPy code)
│   ├── tree_experiments.py   # depth / leaf sweeps, Gini vs entropy, learning curve
│   ├── svm_experiments.py    # lambda and gamma sweeps, convergence, margin
│   ├── baselines.py          # sklearn baselines (comparison only)
│   ├── benchmark.py          # matched CV, single test evaluation, timing, tables
│   ├── error_analysis.py
│   ├── bonus_clustering.py   # BONUS P10: PCA, k-means, DBSCAN
│   ├── bonus_linear.py       # BONUS P8: Ridge and epsilon-SVR from scratch
│   ├── bonus_ensembles.py    # BONUS P9: Random Forest from our tree (bootstrap, OOB)
│   ├── bonus_boosting.py     # BONUS P9: AdaBoost with small trees of our own tree
│   └── run_all.py            # single entry point
├── tests/
├── contribution_report.md
└── requirements.txt
```

scikit-learn is used only in `baselines.py`, `benchmark.py`, `data_prep.py`
(stratified split helper), the bonus modules (reference comparisons only) and the tests; the decision
tree, SVM and metrics never import it (`tests/test_core_has_no_sklearn.py`).
