# Data

The **bina.az sale** dataset is **not committed** to the repo (see `.gitignore`).
Each team member downloads it locally.

## Where to get it
Kaggle: **"binaaz-sale-project"** by `sehriyarmemmedli`
<https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project>

## How to place it
1. Download the dataset archive from Kaggle.
2. Extract it so the CSV lives **directly in this folder**, e.g.:
   ```
   data/bina_az_sale.csv
   ```
3. If your filename differs, set the path once at the top of
   `src/data_prep.py` (the `DATA_PATH` constant) — do not scatter paths
   across the codebase.

## Dataset copy used by this project
- **Kaggle dataset:** `sehriyarmemmedli/binaaz-sale-project`
- **Local filename:** `bina_az_sale.csv`
- **Copy/download date recorded in this workspace:** 14 Oct 2026
- **Observed CSV:** 100,775 rows, 51 columns; listing prices are in AZN

## Notes / gotchas
- Column names are in **Azerbaijani** (e.g. `Sahə`, `Otaq sayı`, `Mərtəbə`,
  `Təmir`, `Kateqoriya`, …). Keep the UTF-8 encoding when you read the file.
- **Leakage warning:** `unit_price` and `total_price` are derived from the
  target `price` in this copy, so both are dropped before training. The full
  cleaning rationale is in `docs/data_decisions.md`.
- The CSV's local file timestamp is 14 Oct 2026. Record the actual Kaggle
  download date instead if it differs from the date this copy was placed here.
