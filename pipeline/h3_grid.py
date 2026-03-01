"""H3 hexagonal grid generation — no WKT, multi-resolution support."""

from __future__ import annotations

import numpy as np
import pandas as pd
import h3
from h3 import LatLngPoly

from pipeline.config import BBOX, H3_RESOLUTIONS


def generate_h3_grid(
    bbox: dict = BBOX,
    resolutions: list[int] = H3_RESOLUTIONS,
) -> dict[int, pd.DataFrame]:
    """Generate H3 grids for one or more resolutions.

    Returns a dict mapping resolution → DataFrame with columns:
      h3_index, lat, lon, area_km2, resolution
    """
    polygon = LatLngPoly(
        [
            (bbox["min_lat"], bbox["min_lon"]),
            (bbox["min_lat"], bbox["max_lon"]),
            (bbox["max_lat"], bbox["max_lon"]),
            (bbox["max_lat"], bbox["min_lon"]),
            (bbox["min_lat"], bbox["min_lon"]),
        ]
    )

    result: dict[int, pd.DataFrame] = {}

    for res in resolutions:
        cells = list(h3.h3shape_to_cells(polygon, res))
        print(f"   🔢 H3 res {res}: {len(cells):,} cells")

        # Vectorised lat/lon extraction
        latlng = np.array([h3.cell_to_latlng(c) for c in cells])
        lats = latlng[:, 0]
        lons = latlng[:, 1]

        areas = np.full(len(cells), h3.cell_area(cells[0], "km^2"))

        result[res] = pd.DataFrame(
            {
                "h3_index": cells,
                "lat": lats,
                "lon": lons,
                "area_km2": areas,
                "resolution": np.full(len(cells), res, dtype=np.uint8),
            }
        )

    return result
