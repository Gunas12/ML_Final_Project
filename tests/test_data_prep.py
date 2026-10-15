import numpy as np
import pandas as pd

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
