from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "bina_az_sale.csv"
FIGURES_PATH = PROJECT_ROOT / "report" / "figures"
SEED = 42
PCA_COMPONENT_LIMIT = 20
KMEANS_CLUSTER_LIMIT = 10
DBSCAN_MIN_SAMPLES = 5
DBSCAN_EPS_KM = 1.0
EARTH_RADIUS_KM = 6371.0088

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data_prep


def _save_figure(figure, output_path):
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _region_labels(cleaned):
    for column in ("location", "city"):
        if column in cleaned:
            return cleaned[column].astype("string").fillna("Unknown").to_numpy()
    raise ValueError("The cleaned dataset must contain a location or city region column")


def _plot_scree(features, output_path):
    component_count = min(
        PCA_COMPONENT_LIMIT,
        features.shape[0],
        features.shape[1],
    )
    pca = PCA(n_components=component_count, svd_solver="randomized", random_state=SEED)
    pca.fit(features)
    explained = pca.explained_variance_ratio_
    components = np.arange(1, component_count + 1)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(components, explained, color="steelblue", label="Individual variance")
    axis.plot(
        components,
        np.cumsum(explained),
        marker="o",
        color="darkorange",
        label="Cumulative variance",
    )
    axis.set(
        title="PCA scree plot",
        xlabel="Principal component",
        ylabel="Explained variance ratio",
        xticks=components,
    )
    axis.legend()
    _save_figure(figure, output_path)
    return explained


def _plot_elbow(features, output_path, selected_cluster_count):
    cluster_counts = range(
        2,
        min(max(KMEANS_CLUSTER_LIMIT, selected_cluster_count), features.shape[0]) + 1,
    )
    inertias = []
    for cluster_count in cluster_counts:
        model = KMeans(
            n_clusters=cluster_count,
            n_init=10,
            random_state=SEED,
        )
        model.fit(features)
        inertias.append(float(model.inertia_))

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(list(cluster_counts), inertias, marker="o", color="tab:blue")
    axis.set(
        title="K-means elbow plot",
        xlabel="Number of clusters",
        ylabel="Within-cluster sum of squares",
        xticks=list(cluster_counts),
    )
    _save_figure(figure, output_path)

    labels = KMeans(
        n_clusters=selected_cluster_count,
        n_init=10,
        random_state=SEED,
    ).fit_predict(features)
    return labels


def _plot_k_distance(coordinates, output_path):
    neighbors = NearestNeighbors(
        n_neighbors=DBSCAN_MIN_SAMPLES,
        algorithm="ball_tree",
        metric="haversine",
    )
    distances, _ = neighbors.fit(coordinates).kneighbors(coordinates)
    k_distances_km = np.sort(distances[:, -1] * EARTH_RADIUS_KM)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.arange(1, len(k_distances_km) + 1), k_distances_km, color="tab:green")
    axis.axhline(
        DBSCAN_EPS_KM,
        color="tab:red",
        linestyle="--",
        label=f"DBSCAN eps = {DBSCAN_EPS_KM:g} km",
    )
    axis.set(
        title=f"{DBSCAN_MIN_SAMPLES}-nearest-neighbor distance (latitude/longitude)",
        xlabel="Points sorted by distance",
        ylabel="Distance (km)",
    )
    axis.legend()
    _save_figure(figure, output_path)

    model = DBSCAN(
        eps=DBSCAN_EPS_KM / EARTH_RADIUS_KM,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
    )
    return model.fit_predict(coordinates)


def run(data_path=DATA_PATH, figures_path=FIGURES_PATH, n_clusters=4):
    raw = data_prep.load_raw(data_path)
    cleaned = data_prep.clean(raw)
    if len(cleaned) < 2:
        raise ValueError("At least two cleaned listings are required for clustering")

    feature_frame, prices, _ = data_prep.make_features(cleaned)
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True).fillna(0.0))
    features = StandardScaler().fit_transform(feature_frame.to_numpy(dtype=float))
    if n_clusters < 2 or n_clusters > len(cleaned):
        raise ValueError("n_clusters must be between 2 and the number of cleaned listings")

    regions = _region_labels(cleaned)
    price_tiers = (prices > np.median(prices)).astype(int)
    output_directory = Path(figures_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    explained = _plot_scree(features, output_directory / "fig_bonus_scree.png")
    kmeans_labels = _plot_elbow(
        features,
        output_directory / "fig_bonus_elbow.png",
        n_clusters,
    )

    latitude = pd.to_numeric(cleaned["lat"], errors="coerce").to_numpy(dtype=float)
    longitude = pd.to_numeric(cleaned["lng"], errors="coerce").to_numpy(dtype=float)
    valid_coordinates = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude >= -90)
        & (latitude <= 90)
        & (longitude >= -180)
        & (longitude <= 180)
    )
    if valid_coordinates.sum() < DBSCAN_MIN_SAMPLES:
        raise ValueError(
            f"At least {DBSCAN_MIN_SAMPLES} valid coordinate pairs are required for DBSCAN"
        )
    coordinates_radians = np.radians(
        np.column_stack((latitude[valid_coordinates], longitude[valid_coordinates]))
    )
    dbscan_labels = _plot_k_distance(
        coordinates_radians,
        output_directory / "fig_bonus_k_distance.png",
    )

    coordinate_prices = price_tiers[valid_coordinates]
    coordinate_regions = regions[valid_coordinates]
    cluster_count = len(set(dbscan_labels) - {-1})
    noise_count = int(np.count_nonzero(dbscan_labels == -1))
    print(f"Price-tier threshold (dataset median): {np.median(prices):,.2f} AZN")
    print(f"K-means comparison uses k={n_clusters}")
    print(f"DBSCAN uses eps={DBSCAN_EPS_KM:g} km, min_samples={DBSCAN_MIN_SAMPLES}")
    print(
        f"DBSCAN found {cluster_count} clusters and {noise_count:,} noise points "
        f"among {valid_coordinates.sum():,} listings with valid coordinates."
    )
    print("Adjusted Rand Index (ARI; DBSCAN noise is treated as a cluster label):")
    print(
        "  K-means vs price tier: "
        f"{adjusted_rand_score(price_tiers, kmeans_labels):.4f}"
    )
    print(
        "  K-means vs region: "
        f"{adjusted_rand_score(regions, kmeans_labels):.4f}"
    )
    print(
        "  DBSCAN vs price tier (valid coordinates): "
        f"{adjusted_rand_score(coordinate_prices, dbscan_labels):.4f}"
    )
    print(
        "  DBSCAN vs region (valid coordinates): "
        f"{adjusted_rand_score(coordinate_regions, dbscan_labels):.4f}"
    )
    print(
        f"Explained variance by first {len(explained)} principal components: "
        f"{np.sum(explained):.4f}"
    )
    for figure_name in (
        "fig_bonus_scree.png",
        "fig_bonus_elbow.png",
        "fig_bonus_k_distance.png",
    ):
        print(f"Saved plot: {output_directory / figure_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run PCA, K-means, and geographic DBSCAN clustering analyses."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--figures", type=Path, default=FIGURES_PATH)
    parser.add_argument("--clusters", type=int, default=4)
    arguments = parser.parse_args()
    run(arguments.data, arguments.figures, arguments.clusters)


if __name__ == "__main__":
    main()
