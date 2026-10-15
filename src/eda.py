from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _split_arrays(splits):
    if len(splits) != 6:
        raise ValueError("splits must contain train, validation, and test X/y pairs")
    X_train = splits[0]
    y_train = np.asarray(splits[1], dtype=float)
    if y_train.ndim != 1 or y_train.size == 0 or not np.all(np.isfinite(y_train)):
        raise ValueError("training prices must be a non-empty finite vector")
    if len(X_train) != y_train.size:
        raise ValueError("training features and prices must have equal row counts")
    if isinstance(X_train, pd.DataFrame):
        features = X_train.reset_index(drop=True)
    else:
        values = np.asarray(X_train, dtype=float)
        if values.ndim != 2:
            raise ValueError("training features must be a two-dimensional matrix")
        features = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(values.shape[1])])
    return features, y_train


def _column(features, candidates):
    for name in candidates:
        if name in features.columns:
            return pd.to_numeric(features[name], errors="coerce")
    return None


def _save(figure, output_directory, name):
    path = output_directory / name
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def _location_values(features):
    if "location" in features:
        return features["location"].astype(str)
    location_columns = [
        column for column in features if str(column).startswith("location=")
    ]
    if location_columns:
        encoded = features[location_columns].to_numpy(dtype=float)
        labels = np.asarray(
            [str(column).split("=", 1)[1] for column in location_columns],
            dtype=object,
        )
        return pd.Series(labels[np.argmax(encoded, axis=1)])
    city_columns = [column for column in features if str(column).startswith("city=")]
    if city_columns:
        encoded = features[city_columns].to_numpy(dtype=float)
        labels = np.asarray(
            [str(column).split("=", 1)[1] for column in city_columns],
            dtype=object,
        )
        return pd.Series(labels[np.argmax(encoded, axis=1)])
    if "city" in features:
        return features["city"].astype(str)
    return None


def run(splits, out_dir="report/figures"):
    features, prices = _split_arrays(splits)
    output_directory = Path(out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_paths = {}

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(prices, bins=60, color="steelblue", edgecolor="white")
    axis.set(title="Sale price distribution", xlabel="Price (AZN)", ylabel="Listings")
    figure_paths["price_distribution"] = _save(
        figure, output_directory, "fig_eda_price_distribution.png"
    )

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(np.log(prices), bins=60, color="darkorange", edgecolor="white")
    axis.set(
        title="Log sale price distribution",
        xlabel="log(Price in AZN)",
        ylabel="Listings",
    )
    figure_paths["log_price_distribution"] = _save(
        figure, output_directory, "fig_eda_log_price_distribution.png"
    )

    area = _column(features, ("area", "Sahə", "sahə"))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    if area is not None:
        valid = area.notna() & (area > 0)
        selected = np.flatnonzero(valid.to_numpy())
        if selected.size > 10_000:
            selected = np.random.default_rng(42).choice(
                selected, size=10_000, replace=False
            )
        axis.scatter(
            area.iloc[selected],
            prices[selected],
            s=8,
            alpha=0.25,
            color="tab:green",
            edgecolors="none",
        )
    axis.set(
        title="Property area versus sale price",
        xlabel="Area (m²)",
        ylabel="Price (AZN)",
    )
    figure_paths["area_price"] = _save(
        figure, output_directory, "fig_eda_area_vs_price.png"
    )

    rooms = _column(features, ("rooms", "Otaq sayı", "otaq sayı"))
    figure, axis = plt.subplots(figsize=(7, 4.5))
    if rooms is not None:
        room_counts = rooms.dropna().round().astype(int).value_counts().sort_index()
        axis.bar(room_counts.index, room_counts.values, color="tab:purple")
    axis.set(title="Room-count distribution", xlabel="Rooms", ylabel="Listings")
    figure_paths["room_count"] = _save(
        figure, output_directory, "fig_eda_room_count.png"
    )

    locations = _location_values(features)
    figure, axis = plt.subplots(figsize=(8, 5))
    if locations is not None:
        location_prices = pd.DataFrame(
            {"location": locations.to_numpy(), "price": prices}
        )
        medians = (
            location_prices.groupby("location", observed=True)["price"]
            .agg(["median", "count"])
            .query("count >= 20")
            .sort_values("median")
            .tail(15)
        )
        axis.barh(medians.index.astype(str), medians["median"], color="tab:cyan")
    axis.set(
        title="Median sale price by region",
        xlabel="Median price (AZN)",
        ylabel="Region",
    )
    figure_paths["region_impact"] = _save(
        figure, output_directory, "fig_eda_region_impact.png"
    )

    latitude = _column(features, ("lat", "latitude"))
    longitude = _column(features, ("lng", "lon", "longitude"))
    figure, axis = plt.subplots(figsize=(7, 5))
    if latitude is not None and longitude is not None:
        valid = latitude.notna() & longitude.notna()
        selected = np.flatnonzero(valid.to_numpy())
        if selected.size > 10_000:
            selected = np.random.default_rng(42).choice(
                selected, size=10_000, replace=False
            )
        points = axis.scatter(
            longitude.iloc[selected],
            latitude.iloc[selected],
            c=np.log(prices[selected]),
            cmap="viridis",
            s=8,
            alpha=0.6,
            edgecolors="none",
        )
        figure.colorbar(points, ax=axis, label="log(Price in AZN)")
    axis.set(
        title="Geographic distribution and price",
        xlabel="Longitude",
        ylabel="Latitude",
    )
    figure_paths["geographic_impact"] = _save(
        figure, output_directory, "fig_eda_lat_lng_impact.png"
    )

    tier_labels = (prices > np.median(prices)).astype(int)
    tier_counts = np.bincount(tier_labels, minlength=2)
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.bar(
        ("Standard", "Premium"),
        tier_counts,
        color=("tab:blue", "tab:red"),
    )
    axis.set(title="Training price-tier balance", ylabel="Listings")
    figure_paths["tier_balance"] = _save(
        figure, output_directory, "fig_eda_tier_balance.png"
    )

    return figure_paths
