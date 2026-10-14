# MLE-AI-201 Machine Learning Final Project

This project predicts sale prices and derived price tiers for listings from
the bina.az real-estate dataset. It implements a decision tree and a Pegasos
support-vector machine in NumPy and compares them with scikit-learn baselines.

## Tasks

- **Regression:** predict the listing price in AZN.
- **Classification:** predict standard (`0`) versus premium (`1`), where
  premium means the price is strictly above the median price in the training
  split. The training threshold is reused for validation and test labels.

The models are evaluated with a deterministic 70% / 15% / 15%
train/validation/test split. The split, SVM standardization, and model
experiments use random seed **42**.

## Dataset

Download Kaggle's **binaaz-sale-project** dataset by `sehriyarmemmedli` and
place the extracted CSV at `data/bina_az_sale.csv`. Dataset details and
version/date information are maintained in [data/README.md](data/README.md).
The raw dataset is excluded from version control.

Cleaning, missing-value handling, outlier policy, feature selection, and
leakage exclusions are documented in
[docs/data_decisions.md](docs/data_decisions.md). In particular,
`unit_price` and `total_price` are target-derived and are excluded from model
features.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running

Run the integrated baseline and from-scratch model pipeline:

```powershell
python -m src.run_all
```

Generate exploratory figures from prepared splits:

```python
from src.data_prep import clean, load_raw, make_features, train_val_test_split
from src.eda import run

cleaned = clean(load_raw())
X, y_price, feature_names = make_features(cleaned)
splits = train_val_test_split(X, y_price, seed=42)
figures = run(splits, out_dir="report/figures")
```

The EDA output comprises price and log-price distributions, area versus
price, room counts, regional and coordinate effects, and the training price
tier balance. Figures are named `fig_eda_*.png`.

Run the focused preparation and model tests:

```powershell
python -m pytest tests/test_data_prep.py tests/test_eda.py tests/test_decision_tree.py tests/test_svm.py tests/test_svm_experiments.py -q
```

## Repository structure

```text
.
├── data/
│   └── README.md
├── docs/
│   └── data_decisions.md
├── presentation/
├── report/
│   ├── figures/
│   └── report.tex
├── src/
│   ├── baselines.py
│   ├── data_prep.py
│   ├── decision_tree.py
│   ├── eda.py
│   ├── evaluate.py
│   ├── run_all.py
│   ├── svm.py
│   ├── svm_experiments.py
│   └── tree_experiments.py
├── tests/
├── contribution_report.md
└── requirements.txt
```

The LaTeX report source is [report/report.tex](report/report.tex); keep the
team contributions in [contribution_report.md](contribution_report.md).
