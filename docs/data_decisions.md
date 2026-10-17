# Data cleaning and preprocessing decisions

## Dataset and target

The project uses Kaggle's **binaaz-sale-project** dataset by
`sehriyarmemmedli`. The local CSV contains 100,775 scraped rows (64,454 distinct
listings) and 51 columns. The `price` target and listing prices are in AZN. The checked-in
`data/README.md` records the source, local filename, and date the dataset
copy was placed in the workspace.

## Cleaning decisions

Cleaning is done in this order (`clean()` in `src/data_prep.py`). Row counts
are for the supplied file: 100,775 raw rows -> **64,350 rows**.

1. **Price and currency.** Parse `price` as numeric, keep finite positive
   values, and (when a currency field exists) keep AZN listings. The supplied
   file is AZN throughout, so nothing is removed here. Remove exact duplicate
   rows (none were exact).
2. **One row per listing (100,775 -> 64,454).** The dump has only 64,454
   distinct listings (`estate_rel_url`; the `rel_url` column is just a
   search-page link). Rows of one listing differ only in views/scrape time, so
   `drop_duplicates()` keeps them. Left in, 6,311 listings appeared in both
   train and test (about 48% of test rows had their listing in train), which
   inflated the test score (regression R2 0.325 -> 0.113 once removed). We keep
   the most recently scraped row per listing (3,356 listings changed price
   between scrapes; the latest price is used). This step runs BEFORE the
   plausibility filters so that an old row is never resurrected after its
   newest row is dropped as an outlier.
3. **Area units to m2.** `Sahə` is "120 m²" for flats/houses but "12 sot" for
   the 3,315 land (`Torpaq`) listings, and `Torpaq sahəsi` is always in sot.
   1 sot = 100 m2 (hektar = 10,000 m2 is handled defensively). Reading "12 sot"
   as 12 m2 made land plots 100x too small, so all areas are converted to m2.
4. **Other numeric columns.** `Otaq sayı` -> `rooms`. `Mərtəbə` has the form
   "7 / 9" (floor / building floors); both numbers are extracted
   (`building_floors` was previously all NaN because the pattern did not allow
   spaces after the slash). No row has a floor above its building's height.
5. **Plausibility filters (fixed constants, 64,454 -> 64,350, 104 rows).**
   - Price must lie in [5,000 ; 20,000,000] AZN. The data contains prices of
     11 and 73 AZN for 90 m2 flats and a 600,000,000 AZN listing for a 200 m2
     flat; these are typos or placeholders, and a handful of them dominated
     RMSE. The limits sit in the extreme tails (0.1% quantile = 4,000 AZN,
     99.9% quantile = 10,000,000 AZN) and were fixed from the distribution and
     common sense, not tuned on any validation/test score.
   - Area must lie in [8 ; 1,000,000] m2 (removes 4-7 m2 "objects" and
     20,000,000 m2 typos). `land_area` outside [1 ; 1,000,000] m2 becomes
     missing and is imputed.
   - Coordinates outside Azerbaijan (lat 38-42.5, lng 44-51) become missing
     (one row had lat 32.7 / lng 12.6) and are imputed.
     Genuine expensive listings (up to 20M AZN) and large plots are kept.

- Impute missing numeric feature values with that feature's median after
  feature construction. Represent missing categorical values as `Unknown`
  before one-hot encoding.
- Exclude listing IDs, URLs, free-text descriptions, scrape timestamps,
  currency labels, and view counts from model features. These fields are not
  stable property characteristics and some may encode collection timing or
  exposure.
- Drop `unit_price` and `total_price`: they are derived from the target price
  and would disclose the answer to both the regression and tier classifiers.
- Outliers: only implausible values are removed (step 5); valid expensive
  listings stay. The highly skewed price distribution is also inspected on a
  logarithmic scale in EDA.
- Categorical columns that are more than 95% empty are not used as features.
  `Binanın növü` is 99.8% missing (only 101 of 64,454 rows filled), so it is
  skipped. `İpoteka` (67% missing, "var" or empty) is kept: empty means no
  mortgage offered and is encoded as `Unknown`.
- Known limitation: about 700 rows share identical price, area, rooms, floor,
  coordinates, district and category with another listing under a different
  URL. They may be re-posted ads or identical flats of one developer in one
  building, so they are NOT removed; they can slightly favour near-duplicate
  matches between train and test.

## Targets, partitioning, and reproducibility

Task A predicts AZN `price`. Task B labels a listing premium (`1`) only when
its price is strictly greater than the median training price; all other
listings, including ties, are standard (`0`). The same training-median
threshold is applied unchanged to validation and test data.

The data is divided into 70% training, 15% validation, and 15% test sets with
random seed 42. Stratification preserves the proportions above and below the
median price across partitions (the strata use the median of the full target
only to balance the splits; the label threshold used by every model is the
training median). After step 2 no listing appears in more than one partition
(checked by `tests/test_data_prep.py`). Split sizes: train 45,045, validation
9,652, test 9,653. Numeric features used by the SVM are
standardized using training means and standard deviations only; constant
training columns use unit scale.

Categorical property fields and region are one-hot encoded. Geographic
coordinates, area, room count, floor, and land area remain numeric. All
processing rules are implemented in `src/data_prep.py`.
