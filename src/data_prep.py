from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
DATA_PATH = "data/bina_az_sale.csv"
LEAKAGE_COLUMNS = ["unit_price", "total_price"]
LISTING_ID_COLUMN = "estate_rel_url"       # one value per real listing (rel_url is only a page link)
SCRAPE_TIME_COLUMN = "datetime_scrape_x"   # when the row was scraped

# Plausibility limits. They are FIXED constants chosen from the price/area
# distribution and common sense (never tuned on a test score) and are
# documented in docs/data_decisions.md.
MIN_PRICE = 5_000.0            # AZN; below this the price is a typo / placeholder
MAX_PRICE = 20_000_000.0       # AZN; above this the price is a typo
MIN_AREA_M2 = 8.0              # square metres
MAX_AREA_M2 = 1_000_000.0      # 100 ha
MAX_LAND_AREA_M2 = 1_000_000.0
LAT_RANGE = (38.0, 42.5)       # Azerbaijan bounding box
LNG_RANGE = (44.0, 51.0)
MAX_MISSING_SHARE = 0.95       # categorical columns emptier than this carry no information

_NUMERIC_FEATURES = (
    "lat",
    "lng",
    "area",
    "land_area",
    "rooms",
    "floor_number",
    "building_floors",
)
_CATEGORICAL_FEATURES = (
    "location",
    "city",
    "Kateqoriya",
    "Təmir",
    "Çıxarış",
    "İpoteka",
    "Binanın növü",
)


def _numeric_values(values):
    numeric = pd.to_numeric(values, errors="coerce")
    missing = numeric.isna() & values.notna()
    if missing.any():
        text = values.loc[missing].astype(str)
        extracted = text.str.extract(r"([-+]?\d[\d\s,]*(?:\.\d+)?)", expand=False)
        parsed = pd.to_numeric(
            extracted.str.replace(r"[\s,]", "", regex=True), errors="coerce"
        )
        numeric.loc[missing] = parsed
    return numeric


def _area_in_m2(values):
    """
    Parse area text into square metres. The dump mixes units: '120 m²' for
    flats/houses but '12 sot' for land (1 sot = 100 m²). Reading '12 sot' as
    12 m² would make land plots look 100x smaller than they are.
    """
    number = _numeric_values(values)
    text = values.astype("string").str.lower()
    is_sot = text.str.contains("sot", na=False).to_numpy()
    is_ha = text.str.contains("hektar|\\bha\\b", na=False).to_numpy()
    factor = np.where(is_ha, 10_000.0, np.where(is_sot, 100.0, 1.0))
    return number * factor


def drop_repeated_listings(df):
    """
    The dump contains the SAME listing many times (re-scraped on different
    dates; only views / scrape time differ), so drop_duplicates() misses them.
    If such rows were split randomly, one listing would land in train AND test.
    Keep exactly one row per listing: the most recently scraped one.
    """
    if LISTING_ID_COLUMN not in df:
        return df
    ordered = df
    if SCRAPE_TIME_COLUMN in df:
        scraped = pd.to_datetime(
            df[SCRAPE_TIME_COLUMN], errors="coerce", utc=True, format="ISO8601"
        )
        ordered = df.assign(_scraped=scraped).sort_values(
            "_scraped", kind="mergesort", na_position="first"
        ).drop(columns="_scraped")
    kept = ordered.drop_duplicates(subset=LISTING_ID_COLUMN, keep="last")
    return kept.sort_index()


def load_raw(path: str = DATA_PATH):
    return pd.read_csv(path, encoding="utf-8")


def clean(df):
    """
    Raw dump -> cleaned frame (one row per listing, plausible values, leakage
    columns removed). Every step is documented in docs/data_decisions.md.
    """
    cleaned = df.copy()
    if "price" not in cleaned:
        raise ValueError("The dataset must contain a price column")

    # 1) numeric price, positive, AZN
    cleaned["price"] = _numeric_values(cleaned["price"])
    valid_price = cleaned["price"].notna() & np.isfinite(cleaned["price"])
    valid_price &= cleaned["price"] > 0
    for currency_column in ("currency_x", "currency_y", "currency"):
        if currency_column in cleaned:
            valid_price &= cleaned[currency_column].fillna("").str.upper().eq("AZN")
            break
    cleaned = cleaned.loc[valid_price].drop_duplicates()

    # 2) one row per listing (latest scrape) -- BEFORE the plausibility filters,
    #    so an old row of a listing is never resurrected by dropping its newest row
    cleaned = drop_repeated_listings(cleaned).copy()

    # 3) numeric columns; area units (sot / hektar) converted to m^2
    if "Sahə" in cleaned:
        cleaned["area"] = _area_in_m2(cleaned["Sahə"])
    if "Torpaq sahəsi" in cleaned:
        cleaned["land_area"] = _area_in_m2(cleaned["Torpaq sahəsi"])
    if "Otaq sayı" in cleaned:
        cleaned["rooms"] = _numeric_values(cleaned["Otaq sayı"])

    if "Mərtəbə" in cleaned:
        floor_text = cleaned["Mərtəbə"].astype("string")
        floor_parts = floor_text.str.extract(
            r"^\s*(\d+(?:[.,]\d+)?)\s*(?:/|\\|\s+of\s+)\s*(\d+(?:[.,]\d+)?)?",
            expand=True,
        )
        cleaned["floor_number"] = _numeric_values(floor_parts[0])
        cleaned["building_floors"] = _numeric_values(floor_parts[1])

    # 4) plausibility filters (fixed constants, see top of file)
    keep = cleaned["price"].between(MIN_PRICE, MAX_PRICE)
    if "area" in cleaned:
        keep &= cleaned["area"].isna() | cleaned["area"].between(MIN_AREA_M2, MAX_AREA_M2)
    cleaned = cleaned.loc[keep].copy()

    if "land_area" in cleaned:       # implausible land area -> missing (imputed later)
        land = cleaned["land_area"]
        cleaned["land_area"] = land.where(land.between(1.0, MAX_LAND_AREA_M2))
    if "lat" in cleaned and "lng" in cleaned:   # coordinates outside Azerbaijan -> missing
        lat = _numeric_values(cleaned["lat"])
        lng = _numeric_values(cleaned["lng"])
        inside = lat.between(*LAT_RANGE) & lng.between(*LNG_RANGE)
        cleaned["lat"] = lat.where(inside)
        cleaned["lng"] = lng.where(inside)

    cleaned = cleaned.drop(columns=LEAKAGE_COLUMNS, errors="ignore")
    return cleaned.reset_index(drop=True)


def make_features(df):
    if "price" not in df:
        raise ValueError("The cleaned dataset must contain a price column")

    feature_frame = pd.DataFrame(index=df.index)
    for column in _NUMERIC_FEATURES:
        if column in df:
            feature_frame[column] = _numeric_values(df[column])
    for column in _CATEGORICAL_FEATURES:
        if column in df:
            if df[column].isna().mean() > MAX_MISSING_SHARE:
                continue            # nearly empty column: only noise (e.g. Binanın növü)
            feature_frame[column] = df[column].astype("string").fillna("Unknown")

    if feature_frame.empty:
        raise ValueError("The dataset does not contain supported feature columns")

    X = pd.get_dummies(
        feature_frame,
        columns=list(feature_frame.select_dtypes(exclude=["number"]).columns),
        prefix_sep="=",
        dtype=float,
    )
    X = X.astype(float)
    y = _numeric_values(df["price"]).to_numpy(dtype=float)
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("Prices must be finite and positive before feature creation")
    return X, y, X.columns.tolist()


def make_tier_label(y_price, threshold=None):
    prices = np.asarray(y_price, dtype=float)
    if prices.ndim != 1 or prices.size == 0 or not np.all(np.isfinite(prices)):
        raise ValueError("y_price must be a non-empty finite one-dimensional array")
    tier_threshold = float(np.median(prices)) if threshold is None else float(threshold)
    if not np.isfinite(tier_threshold):
        raise ValueError("threshold must be finite")
    return (prices > tier_threshold).astype(int), tier_threshold


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    targets = np.asarray(y)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("y must be a non-empty one-dimensional array")
    if len(X) != targets.size:
        raise ValueError("X and y must contain the same number of samples")
    if (
        not 0 < val_size < 1
        or not 0 < test_size < 1
        or val_size + test_size >= 1
    ):
        raise ValueError("val_size and test_size must be positive and sum to less than 1")

    tiers, stratification_threshold = make_tier_label(targets)
    stratify = tiers if np.unique(tiers).size > 1 else None
    holdout_size = val_size + test_size
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        targets,
        stratify=stratify,
        test_size=holdout_size,
        random_state=seed,
    )
    holdout_test_fraction = test_size / holdout_size
    tiers_holdout = (y_holdout > stratification_threshold).astype(int)
    holdout_stratify = tiers_holdout if np.unique(tiers_holdout).size > 1 else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        stratify=holdout_stratify,
        test_size=holdout_test_fraction,
        random_state=seed,
    )
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_val = X_val.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        fill_values = X_train.median(numeric_only=True).fillna(0.0)
        X_train = X_train.fillna(fill_values)
        X_val = X_val.fillna(fill_values)
        X_test = X_test.fillna(fill_values)
    else:
        X_train = np.asarray(X_train, dtype=float)
        X_val = np.asarray(X_val, dtype=float)
        X_test = np.asarray(X_test, dtype=float)
        X_train[~np.isfinite(X_train)] = np.nan
        X_val[~np.isfinite(X_val)] = np.nan
        X_test[~np.isfinite(X_test)] = np.nan
        fill_values = np.zeros(X_train.shape[1], dtype=float)
        for column in range(X_train.shape[1]):
            finite_values = X_train[np.isfinite(X_train[:, column]), column]
            if finite_values.size:
                fill_values[column] = np.median(finite_values)
        for split in (X_train, X_val, X_test):
            missing_rows, missing_columns = np.where(~np.isfinite(split))
            split[missing_rows, missing_columns] = fill_values[missing_columns]
    return X_train, y_train, X_val, y_val, X_test, y_test


def standardize(X_tr, *others):
    training = np.asarray(X_tr, dtype=float)
    if training.ndim != 2 or training.shape[0] == 0:
        raise ValueError("X_tr must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(training)):
        raise ValueError("X_tr must contain only finite values")
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale[scale == 0] = 1.0
    scaled = [(training - mean) / scale]
    for values in others:
        array = np.asarray(values, dtype=float)
        if array.ndim != 2 or array.shape[1] != training.shape[1]:
            raise ValueError("Every split must have the same feature count as X_tr")
        if not np.all(np.isfinite(array)):
            raise ValueError("All feature splits must contain only finite values")
        scaled.append((array - mean) / scale)
    return tuple(scaled)