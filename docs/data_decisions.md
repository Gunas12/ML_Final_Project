# Data cleaning and preprocessing decisions

## Dataset and target

The project uses Kaggle's **binaaz-sale-project** dataset by
`sehriyarmemmedli`. The local CSV contains 100,775 sale listings and 51
columns. The `price` target and listing prices are in AZN. The checked-in
`data/README.md` records the source, local filename, and date the dataset
copy was placed in the workspace.

## Cleaning decisions

- Parse `price` as numeric, retain only finite positive values, and, when a
  currency field is available, retain AZN listings. The supplied file records
  AZN throughout.
- Remove exact duplicate rows. No rows are removed solely for a large or
  small but valid price, area, or coordinate; high-value listings are a
  legitimate part of this market and are retained for evaluation.
- Parse `Sahə`, `Torpaq sahəsi`, and `Otaq sayı` into numeric area, land-area,
  and room-count features. Parse `Mərtəbə` into the listed floor and total
  building floors.
- Impute missing numeric feature values with that feature's median after
  feature construction. Represent missing categorical values as `Unknown`
  before one-hot encoding.
- Exclude listing IDs, URLs, free-text descriptions, scrape timestamps,
  currency labels, and view counts from model features. These fields are not
  stable property characteristics and some may encode collection timing or
  exposure.
- Drop `unit_price` and `total_price`: they are derived from the target price
  and would disclose the answer to both the regression and tier classifiers.
- Keep valid price outliers rather than applying a fixed cap or IQR filter.
  The highly skewed price distribution is also inspected on a logarithmic
  scale in EDA.

## Targets, partitioning, and reproducibility

Task A predicts AZN `price`. Task B labels a listing premium (`1`) only when
its price is strictly greater than the median training price; all other
listings, including ties, are standard (`0`). The same training-median
threshold is applied unchanged to validation and test data.

The data is divided into 70% training, 15% validation, and 15% test sets with
random seed 42. Stratification preserves the proportions above and below the
median price across partitions. Numeric features used by the SVM are
standardized using training means and standard deviations only; constant
training columns use unit scale.

Categorical property fields and region are one-hot encoded. Geographic
coordinates, area, room count, floor, and land area remain numeric. All
processing rules are implemented in `src/data_prep.py`.
