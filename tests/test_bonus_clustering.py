import re

import numpy as np
import pandas as pd
import pytest

from src import bonus_clustering


def make_clustering_data():
    rows = 30
    positions = np.repeat(np.arange(6), 5)
    centers = np.array(
        [
            [40.10, 49.80],
            [40.20, 49.90],
            [40.30, 50.00],
            [40.40, 50.10],
            [40.50, 50.20],
            [40.60, 50.30],
        ]
    )
    offsets = np.tile(np.linspace(-0.0001, 0.0001, 5), 6)
    cleaned = pd.DataFrame(
        {
            "lat": centers[positions, 0] + offsets,
            "lng": centers[positions, 1] + offsets,
            "location": [f"region_{position % 3}" for position in positions],
        }
    )
    prices = np.linspace(50_000.0, 200_000.0, rows)
    feature_frame = pd.DataFrame(
        {
            "area": np.linspace(30.0, 180.0, rows),
            "rooms": np.tile([1.0, 2.0, 3.0], rows // 3),
            "feature_a": np.sin(np.arange(rows)),
            "feature_b": np.cos(np.arange(rows) / 2),
        }
    )
    return cleaned, feature_frame, prices


def test_run_creates_figures_and_reports_finite_ari_values(tmp_path, monkeypatch, capsys):
    cleaned, feature_frame, prices = make_clustering_data()
    monkeypatch.setattr(bonus_clustering.data_prep, "load_raw", lambda path: object())
    monkeypatch.setattr(bonus_clustering.data_prep, "clean", lambda raw: cleaned)
    monkeypatch.setattr(
        bonus_clustering.data_prep,
        "make_features",
        lambda frame: (feature_frame, prices, None),
    )
    output_directory = tmp_path / "nested" / "figures"

    bonus_clustering.run(
        data_path=tmp_path / "unused.csv",
        figures_path=output_directory,
        n_clusters=3,
    )

    expected_names = {
        "fig_bonus_scree.png",
        "fig_bonus_elbow.png",
        "fig_bonus_k_distance.png",
    }
    figure_paths = {path.name: path for path in output_directory.iterdir()}
    assert set(figure_paths) == expected_names
    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in figure_paths.values()
    )

    output = capsys.readouterr().out
    ari_values = re.findall(
        r"(?:K-means|DBSCAN) vs [^\n]+: (-?\d+(?:\.\d+)?)",
        output,
    )
    assert len(ari_values) == 4
    assert all(np.isfinite(float(value)) for value in ari_values)


def test_run_rejects_invalid_cluster_count(tmp_path, monkeypatch):
    cleaned, feature_frame, prices = make_clustering_data()
    monkeypatch.setattr(bonus_clustering.data_prep, "load_raw", lambda path: object())
    monkeypatch.setattr(bonus_clustering.data_prep, "clean", lambda raw: cleaned)
    monkeypatch.setattr(
        bonus_clustering.data_prep,
        "make_features",
        lambda frame: (feature_frame, prices, None),
    )

    with pytest.raises(ValueError, match="n_clusters"):
        bonus_clustering.run(
            data_path=tmp_path / "unused.csv",
            figures_path=tmp_path / "figures",
            n_clusters=len(cleaned) + 1,
        )
