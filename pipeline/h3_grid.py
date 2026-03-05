"""H3 hexagonal grid generation — no WKT, multi-resolution support."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import h3
from h3 import LatLngPoly

from pipeline.config import BBOX, H3_RESOLUTIONS

log = logging.getLogger(__name__)


def generate_h3_grid(
    bbox: dict = BBOX,
    resolutions: list[int] = H3_RESOLUTIONS,
) -> dict[int, pd.DataFrame]:
    """Generate H3 grids for one or more resolutions.

    Returns a dict mapping resolution → DataFrame with columns:
      h3_index (int64), lat, lon, resolution
    """
    is_global = (bbox["max_lat"] - bbox["min_lat"]) >= 170 and (
        bbox["max_lon"] - bbox["min_lon"]
    ) >= 350

    result: dict[int, pd.DataFrame] = {}

    for res in resolutions:
        if is_global:
            # LatLngPoly cannot represent the full globe — use all base cells
            # expanded to the target resolution instead.
            cells = list(h3.uncompact_cells(h3.get_res0_cells(), res))
        else:
            polygon = LatLngPoly(
                [
                    (bbox["min_lat"], bbox["min_lon"]),
                    (bbox["min_lat"], bbox["max_lon"]),
                    (bbox["max_lat"], bbox["max_lon"]),
                    (bbox["max_lat"], bbox["min_lon"]),
                    (bbox["min_lat"], bbox["min_lon"]),
                ]
            )
            cells = list(h3.h3shape_to_cells(polygon, res))

        log.info("[H3] res %d: %s cells", res, f"{len(cells):,}")

        if not cells:
            result[res] = pd.DataFrame(columns=["h3_index", "lat", "lon", "resolution"])
            continue

        # Vectorised lat/lon extraction
        latlng = np.array([h3.cell_to_latlng(c) for c in cells])
        lats = latlng[:, 0]
        lons = latlng[:, 1]

        # Convert H3 hex strings to int64 for efficient Parquet encoding
        h3_ints = np.array([int(c, 16) for c in cells], dtype=np.int64)

        result[res] = pd.DataFrame(
            {
                "h3_index": h3_ints,
                "lat": lats,
                "lon": lons,
                "resolution": np.full(len(cells), res, dtype=np.uint8),
            }
        )

    return result
