from pathlib import Path

import numpy as np
import pandas as pd

from src.eda import run


def test_run_writes_requested_eda_figures(tmp_path):
    rows = 80
    X = pd.DataFrame(
        {
            "area": np.linspace(35, 180, rows),
            "rooms": np.tile([1, 2, 3, 4], rows // 4),
            "lat": np.linspace(40.3, 40.5, rows),
            "lng": np.linspace(49.7, 50.0, rows),
            "location=Central": np.tile([1.0, 0.0], rows // 2),
            "location=Outer": np.tile([0.0, 1.0], rows // 2),
        }
    )
    prices = np.linspace(60_000, 700_000, rows)
    splits = (X, prices, X.iloc[:12], prices[:12], X.iloc[12:24], prices[12:24])

    figures = run(splits, out_dir=tmp_path)

    expected_names = {
        "fig_eda_price_distribution.png",
        "fig_eda_log_price_distribution.png",
        "fig_eda_area_vs_price.png",
        "fig_eda_room_count.png",
        "fig_eda_region_impact.png",
        "fig_eda_lat_lng_impact.png",
        "fig_eda_tier_balance.png",
    }
    paths = [Path(path) for path in figures.values()]
    assert {path.name for path in paths} == expected_names
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
