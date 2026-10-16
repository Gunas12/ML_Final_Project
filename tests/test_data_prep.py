import os

import numpy as np
import pandas as pd
import pytest

from src import data_prep


def make_raw_data(rows=40):
    prices = np.linspace(100_000, 900_000, rows)
    return pd.DataFrame(
        {
            "price": prices,
            "unit_price": ["2 500 AZN/m²"] * rows,
            "total_price": prices,
            "currency_x": ["AZN"] * rows,
            "location": ["Region A" if index % 2 else "Region B" for index in range(rows)],
            "city": ["Baku"] * rows,
            "lat": np.linspace(40.3, 40.5, rows),
            "lng": np.linspace(49.7, 50.0, rows),
            "Sahə": ["100 m²"] * rows,
            "Otaq sayı": [2 + index % 3 for index in range(rows)],
            "Mərtəbə": ["3/12"] * rows,
            "Kateqoriya": ["Yeni tikili"] * rows,
            "Təmir": ["var"] * rows,
            "Çıxarış": ["var"] * rows,
            "İpoteka": ["yoxdur"] * rows,
        }
    )


def test_price_derived_columns_are_removed_from_clean_data_and_features():
    cleaned = data_prep.clean(make_raw_data())
    X, y, feature_names = data_prep.make_features(cleaned)

    assert set(data_prep.LEAKAGE_COLUMNS).isdisjoint(cleaned.columns)
    assert set(data_prep.LEAKAGE_COLUMNS).isdisjoint(feature_names)
    assert X.shape[0] == y.size == len(cleaned)


def test_tier_threshold_is_calculated_from_training_prices_only():
    training_prices = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    validation_prices = np.array([1_000.0, 2_000.0])
    test_prices = np.array([3_000.0, 4_000.0])

    training_tiers, threshold = data_prep.make_tier_label(training_prices)
    validation_tiers, validation_threshold = data_prep.make_tier_label(
        validation_prices, threshold=threshold
    )
    test_tiers, test_threshold = data_prep.make_tier_label(
        test_prices, threshold=threshold
    )

    assert threshold == np.median(training_prices)
    assert validation_threshold == test_threshold == threshold
    np.testing.assert_array_equal(training_tiers, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(validation_tiers, [1, 1])
    np.testing.assert_array_equal(test_tiers, [1, 1])


def test_split_sizes_are_seventy_fifteen_fifteen():
    X = np.arange(2_000).reshape(1_000, 2)
    y = np.tile([100.0, 100.0, 100.0, 100.0, 1_000.0], 200)

    splits = data_prep.train_val_test_split(X, y, seed=42)

    assert tuple(len(splits[index]) for index in (0, 2, 4)) == (700, 150, 150)


def test_split_preserves_price_tier_proportions():
    X = np.arange(2_000).reshape(1_000, 2)
    y = np.tile([100.0, 100.0, 100.0, 100.0, 1_000.0], 200)
    global_tiers, _ = data_prep.make_tier_label(y)
    expected_rate = global_tiers.mean()

    splits = data_prep.train_val_test_split(X, y, seed=42)

    for y_split in (splits[1], splits[3], splits[5]):
        global_split_tiers = (y_split > np.median(y)).astype(int)
        assert abs(global_split_tiers.mean() - expected_rate) < 0.02


def test_repeated_listings_are_collapsed_to_the_latest_scrape():
    raw = make_raw_data(rows=6)
    raw["estate_rel_url"] = ["/a", "/a", "/b", "/b", "/b", "/c"]
    raw["datetime_scrape_x"] = [
        "2024-10-05 10:00:00.5+00", "2024-11-01 10:00:00.5+00",   # /a: 2nd is newer
        "2024-11-09 10:00:00.5+00", "2024-10-01 10:00:00+00",     # /b: 1st is newest
        "2024-10-02 10:00:00+00", "2024-10-03 10:00:00+00",
    ]
    raw["price"] = [100_000.0, 110_000.0, 200_000.0, 190_000.0, 180_000.0, 300_000.0]

    cleaned = data_prep.clean(raw)

    assert cleaned["estate_rel_url"].is_unique
    assert sorted(cleaned["price"]) == [110_000.0, 200_000.0, 300_000.0]


def test_old_scrape_is_not_resurrected_when_the_latest_row_is_an_outlier():
    raw = make_raw_data(rows=2)
    raw["estate_rel_url"] = ["/a", "/a"]
    raw["datetime_scrape_x"] = ["2024-10-01 10:00:00+00", "2024-11-01 10:00:00+00"]
    raw["price"] = [150_000.0, 600_000_000.0]      # newest row has a typo price

    cleaned = data_prep.clean(raw)

    assert len(cleaned) == 0


def test_building_floors_are_parsed_from_floor_text_with_spaces():
    raw = make_raw_data(rows=4)
    raw["Mərtəbə"] = ["7 / 9", "3/5", "16 / 17", "2"]

    cleaned = data_prep.clean(raw)

    assert cleaned["floor_number"].tolist()[:3] == [7, 3, 16]
    assert cleaned["building_floors"].tolist()[:3] == [9, 5, 17]
    assert "building_floors" in data_prep.make_features(cleaned)[2]


def test_area_units_are_converted_to_square_metres():
    raw = make_raw_data(rows=5)
    raw["Sahə"] = ["120 m²", "12 sot", "50.5 m²", "2 hektar", "0.5 sot"]
    raw["Torpaq sahəsi"] = [None, "5 sot", "1.3 sot", None, None]

    cleaned = data_prep.clean(raw)

    # 0.5 sot = 50 m2 is kept; every value is now in m2
    assert cleaned["area"].tolist() == [120.0, 1200.0, 50.5, 20_000.0, 50.0]
    assert cleaned["land_area"].iloc[1] == 500.0
    assert cleaned["land_area"].iloc[2] == 130.0
    assert np.isnan(cleaned["land_area"].iloc[0])


def test_price_outliers_are_removed_with_inclusive_limits():
    raw = make_raw_data(rows=6)
    raw["price"] = [
        11.0, 4_999.0, data_prep.MIN_PRICE, 250_000.0, data_prep.MAX_PRICE, 600_000_000.0,
    ]

    cleaned = data_prep.clean(raw)

    assert cleaned["price"].tolist() == [data_prep.MIN_PRICE, 250_000.0, data_prep.MAX_PRICE]


def test_area_outliers_are_removed():
    raw = make_raw_data(rows=4)
    raw["Sahə"] = ["4 m²", "8 m²", "90 m²", "20000000 m²"]

    cleaned = data_prep.clean(raw)

    assert cleaned["area"].tolist() == [8.0, 90.0]


def test_coordinates_outside_azerbaijan_become_missing():
    raw = make_raw_data(rows=3)
    raw["lat"] = [40.4, 32.7, 40.5]
    raw["lng"] = [49.8, 12.6, 49.9]

    cleaned = data_prep.clean(raw)

    assert cleaned["lat"].notna().tolist() == [True, False, True]
    assert cleaned["lng"].notna().tolist() == [True, False, True]


def test_nearly_empty_categorical_columns_are_not_used_as_features():
    raw = make_raw_data(rows=40)
    raw["Binanın növü"] = [None] * 39 + ["Villa"]       # 97.5% missing

    X, _, names = data_prep.make_features(data_prep.clean(raw))

    assert not any(name.startswith("Binanın növü") for name in names)
    assert any(name.startswith("Təmir") for name in names)     # normal columns stay


def test_cleaning_does_not_modify_its_input():
    raw = make_raw_data(rows=10)
    before = raw.copy(deep=True)

    data_prep.clean(raw)

    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.skipif(not os.path.exists(data_prep.DATA_PATH), reason="dataset not downloaded")
def test_real_data_is_clean_and_has_no_listing_shared_between_splits():
    cleaned = data_prep.clean(data_prep.load_raw())
    X, y, names = data_prep.make_features(cleaned)

    # one row per listing, plausible values, leakage columns gone
    assert cleaned["estate_rel_url"].is_unique
    assert cleaned["price"].between(data_prep.MIN_PRICE, data_prep.MAX_PRICE).all()
    assert cleaned["area"].between(data_prep.MIN_AREA_M2, data_prep.MAX_AREA_M2).all()
    assert cleaned["building_floors"].notna().sum() > 0
    assert set(data_prep.LEAKAGE_COLUMNS).isdisjoint(names)
    assert not np.isinf(X.to_numpy()).any()

    # land plots are in m2 now (sot x 100), not tiny numbers
    land_median = cleaned.loc[cleaned["Kateqoriya"] == "Torpaq", "area"].median()
    assert land_median >= 500.0

    # no listing appears in two splits
    ids = cleaned["estate_rel_url"].to_numpy()
    index = np.arange(len(y)).reshape(-1, 1)
    tr, _, va, _, te, _ = data_prep.train_val_test_split(index, y, seed=data_prep.SEED)
    sets = [set(ids[part.astype(int).ravel()]) for part in (tr, va, te)]
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])


def test_splits_contain_only_finite_values_and_are_reproducible():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    X[::17, 1] = np.nan
    y = rng.uniform(10_000, 500_000, size=300)

    first = data_prep.train_val_test_split(X, y, seed=42)
    second = data_prep.train_val_test_split(X, y, seed=42)

    for part in (first[0], first[2], first[4]):
        assert np.isfinite(part).all()
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_standardize_uses_only_training_statistics():
    X_tr = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]])
    X_te = np.array([[100.0, 10.0]])

    scaled_tr, scaled_te = data_prep.standardize(X_tr, X_te)

    np.testing.assert_allclose(scaled_tr.mean(axis=0), [0.0, 0.0], atol=1e-12)
    expected = (100.0 - X_tr[:, 0].mean()) / X_tr[:, 0].std()
    assert scaled_te[0, 0] == pytest.approx(expected)
    assert scaled_te[0, 1] == 0.0          # constant column: no division by zero