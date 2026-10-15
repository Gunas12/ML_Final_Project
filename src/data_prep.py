from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
DATA_PATH = "data/bina_az_sale.csv"
LEAKAGE_COLUMNS = ["unit_price", "total_price"]

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


def load_raw(path: str = DATA_PATH):
    return pd.read_csv(path, encoding="utf-8")


def clean(df):
    cleaned = df.copy()
    if "price" not in cleaned:
        raise ValueError("The dataset must contain a price column")

    cleaned["price"] = _numeric_values(cleaned["price"])
    valid_price = cleaned["price"].notna() & np.isfinite(cleaned["price"])
    valid_price &= cleaned["price"] > 0
    for currency_column in ("currency_x", "currency_y", "currency"):
        if currency_column in cleaned:
            valid_price &= cleaned[currency_column].fillna("").str.upper().eq("AZN")
            break
    cleaned = cleaned.loc[valid_price].drop_duplicates().copy()

    numeric_sources = {
        "Sahə": "area",
        "Torpaq sahəsi": "land_area",
        "Otaq sayı": "rooms",
    }
    for source, destination in numeric_sources.items():
        if source in cleaned:
            cleaned[destination] = _numeric_values(cleaned[source])

    if "Mərtəbə" in cleaned:
        floor_text = cleaned["Mərtəbə"].astype("string")
        floor_parts = floor_text.str.extract(
            r"^\s*(\d+(?:[.,]\d+)?)\s*(?:/|\\|\s+of\s+)(\d+(?:[.,]\d+)?)?",
            expand=True,
        )
        cleaned["floor_number"] = _numeric_values(floor_parts[0])
        cleaned["building_floors"] = _numeric_values(floor_parts[1])

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
