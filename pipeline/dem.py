"""DEM loading — pre-computed H3 Parquet (primary) or STAC raster (fallback).

Primary source  : H3-indexed Parquet on Source Cooperative (public, no auth)
Fallback source : Copernicus GLO-30 via Microsoft Planetary Computer STAC
Fallback #2     : openlandmap merged 30 m DEM (stac.openlandmap.org)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pipeline.config import (
    BBOX,
    DEM_PARQUET_BASE,
    DEM_PARQUET_MAX_RES,
    DEM_PARQUET_REGION,
)
from pipeline.gpu import to_device, to_numpy, xp, xp_ndimage

log = logging.getLogger(__name__)


def _aggregate_dem_to_parent(dem_df: pd.DataFrame, target_res: int) -> pd.DataFrame:
    """Aggregate DEM data from a finer resolution to a coarser one via H3 parents."""
    import h3

    dem_df = dem_df.copy()
    current_res = h3.get_resolution(dem_df["h3_index"].iloc[0])
    dem_df["parent"] = dem_df["h3_index"].apply(
        lambda c: h3.cell_to_parent(c, target_res)
    )

    agg = (
        dem_df.groupby("parent")
        .agg(
            elev=("elev", "mean"),
            slope=("slope", "mean"),
            aspect=("aspect", "mean"),
            tri=("tri", "mean"),
            tpi=("tpi", "mean"),
        )
        .reset_index()
        .rename(columns={"parent": "h3_index"})
    )

    log.info(
        "[DEM] Aggregated res %d -> res %d (%d -> %d cells)",
        current_res,
        target_res,
        len(dem_df),
        len(agg),
    )
    return agg


# Minimum resolution available as pre-computed Parquet.
# For resolutions below this, we load DEM_PARQUET_MIN_RES and aggregate to parents.
DEM_PARQUET_MIN_RES = 1


def load_dem_parquet(h3_res: int, h3_df: pd.DataFrame) -> dict | None:
    """Load pre-computed terrain from H3-indexed Parquet on S3.

    Returns a dict with 1D arrays matching the order of *h3_df*, plus
    ``h3_native=True`` so downstream code skips RegularGridInterpolator.

    Returns None if the resolution is not available as Parquet.

    For resolutions below DEM_PARQUET_MIN_RES, loads the min-res Parquet
    and aggregates to parent cells.
    """
    if h3_res > DEM_PARQUET_MAX_RES:
        return None

    # Determine which resolution to actually load from S3
    load_res = max(h3_res, DEM_PARQUET_MIN_RES)
    url = f"{DEM_PARQUET_BASE}/h3/h3_res={load_res}/data.parquet"
    log.info("[DEM] Loading from Parquet (H3 res %d)", load_res)

    try:
        import pyarrow.parquet as pq
        from pyarrow.fs import S3FileSystem

        fs = S3FileSystem(region=DEM_PARQUET_REGION, anonymous=True)
        # Strip the s3:// scheme for PyArrow
        s3_path = url.removeprefix("s3://")
        dem_table = pq.read_table(
            s3_path,
            filesystem=fs,
            columns=["h3_index", "elev", "slope", "aspect", "tri", "tpi"],
        )
        dem_df = dem_table.to_pandas()
    except Exception as e:
        log.warning("Parquet load failed: %s", e)
        return None

    # Aggregate to coarser resolution if needed
    if h3_res < load_res:
        dem_df = _aggregate_dem_to_parent(dem_df, h3_res)

    # Join on h3_index to align DEM values with the pipeline's H3 grid
    merged = h3_df[["h3_index"]].merge(dem_df, on="h3_index", how="left")

    n_total = len(merged)
    n_matched = merged["elev"].notna().sum()
    n_missing = n_total - n_matched

    if n_matched == 0:
        log.warning("No matching H3 cells in Parquet -- falling back to STAC")
        return None

    # Fill missing terrain values (ocean cells not in the DEM dataset)
    for col in ("elev", "slope", "aspect", "tri", "tpi"):
        merged[col] = merged[col].fillna(0.0)

    log.info(
        "[DEM] Parquet: %s/%s cells matched  elev %s-%s m  slope max %.1f deg",
        f"{n_matched:,}",
        f"{n_total:,}",
        f"{merged['elev'].min():.0f}",
        f"{merged['elev'].max():.0f}",
        merged["slope"].max(),
    )
    if n_missing > 0:
        log.info("[DEM] %s ocean/missing cells filled with 0", f"{n_missing:,}")

    return {
        "h3_native": True,
        "elev": merged["elev"].values.astype(np.float32),
        "slope": merged["slope"].values.astype(np.float32),
        "aspect": merged["aspect"].values.astype(np.float32),
        "tri": merged["tri"].values.astype(np.float32),
        "tpi": merged["tpi"].values.astype(np.float32),
        "lat": h3_df["lat"].values,
        "lon": h3_df["lon"].values,
    }


def load_dem(bbox: dict = BBOX, resolution: float | None = None) -> dict:
    """Load DEM raster and compute terrain derivatives on GPU (or CPU).

    When *resolution* is None it is auto-computed to target ~2000x2000 pixels.

    Returns a dict with keys: elev, lat, lon, slope, aspect, tri, tpi, dx, dy.
    All array values are numpy (CPU) arrays — GPU tensors are only used
    transiently during derivative computation.
    """

    # Auto-scale resolution to target ~2000×2000 px
    if resolution is None:
        lat_span = bbox["max_lat"] - bbox["min_lat"]
        lon_span = bbox["max_lon"] - bbox["min_lon"]
        resolution = max(lat_span, lon_span) / 2000
        resolution = max(resolution, 0.0003)  # floor at ~30 m

    # DEM bbox: add a small buffer for edge interpolation,
    # but skip for global bbox (edges wrap at ±180/±90).
    is_global = (bbox["max_lat"] - bbox["min_lat"]) >= 170
    buffer = 0.0 if is_global else 0.05
    dem_bbox = {
        "min_lat": max(bbox["min_lat"] - buffer, -90.0),
        "max_lat": min(bbox["max_lat"] + buffer, 90.0),
        "min_lon": max(bbox["min_lon"] - buffer, -180.0),
        "max_lon": min(bbox["max_lon"] + buffer, 180.0),
    }

    log.info("[DEM] Loading from STAC (raster fallback)")

    dem_ds = _load_from_planetary_computer(dem_bbox, resolution)

    if dem_ds is None:
        log.warning("Planetary Computer unavailable, trying openlandmap")
        dem_ds = _load_from_openlandmap(dem_bbox, resolution)

    dem = dem_ds["data"].squeeze()

    lons = dem.longitude.values if "longitude" in dem.coords else dem.x.values
    lats = dem.latitude.values if "latitude" in dem.coords else dem.y.values

    # Ensure latitude is ascending (south→north) so np.gradient dz/dy
    # points northward, which the aspect formula requires.
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        elev_np = dem.values[::-1]
    else:
        elev_np = dem.values

    elev_gpu = to_device(elev_np)

    _xp = xp()
    _ndi = xp_ndimage()

    lat0 = (lats.max() + lats.min()) / 2
    dx = abs(lons[1] - lons[0]) * 111_320 * np.cos(np.radians(lat0))
    dy = abs(lats[1] - lats[0]) * 111_320

    dz_dy, dz_dx = _xp.gradient(elev_gpu, dy, dx)
    slope = _xp.arctan(_xp.sqrt(dz_dx**2 + dz_dy**2)) * 180 / _xp.pi
    aspect = (_xp.arctan2(-dz_dx, -dz_dy) * 180 / _xp.pi + 360) % 360

    kernel_ones = _xp.ones((3, 3), dtype=_xp.float32)
    sum_z2 = _ndi.convolve(elev_gpu**2, kernel_ones, mode="nearest")
    sum_z = _ndi.convolve(elev_gpu, kernel_ones, mode="nearest")
    tri = _xp.sqrt(_xp.maximum(sum_z2 - 2 * elev_gpu * sum_z + 9 * elev_gpu**2, 0) / 9)

    tpi = elev_gpu - _ndi.uniform_filter(elev_gpu, size=5, mode="nearest")

    log.info(
        "[DEM] %s  elev %.0f-%.0f m  slope max %.1f deg",
        elev_gpu.shape,
        float(_xp.min(elev_gpu)),
        float(_xp.max(elev_gpu)),
        float(_xp.max(slope)),
    )

    return {
        "elev": to_numpy(elev_gpu),
        "lat": lats,
        "lon": lons,
        "slope": to_numpy(slope),
        "aspect": to_numpy(aspect),
        "tri": to_numpy(tri),
        "tpi": to_numpy(tpi),
        "dx": dx,
        "dy": dy,
    }


# ── private helpers ───────────────────────────────────────────────────────────


def _load_from_planetary_computer(bbox: dict, resolution: float):
    try:
        import odc.stac
        import pystac_client
        import planetary_computer

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        items = list(
            catalog.search(
                collections=["cop-dem-glo-30"],
                bbox=[
                    bbox["min_lon"],
                    bbox["min_lat"],
                    bbox["max_lon"],
                    bbox["max_lat"],
                ],
            ).items()
        )

        return odc.stac.load(
            items,
            bands=["data"],
            crs="EPSG:4326",
            resolution=resolution,
            resampling="bilinear",
            bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
        )
    except Exception as e:
        log.warning("Planetary Computer error: %s", e)
        return None


def _load_from_openlandmap(bbox: dict, resolution: float):
    """Fallback: OpenLandMap merged 30 m DEM via STAC."""
    import odc.stac
    import pystac_client

    catalog = pystac_client.Client.open("https://stac.openlandmap.org")
    items = list(
        catalog.search(
            collections=["gedtm-30m"],
            bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
        ).items()
    )

    if not items:
        raise RuntimeError("No DEM tiles found in openlandmap either.")

    return odc.stac.load(
        items,
        bands=["data"],
        crs="EPSG:4326",
        resolution=resolution,
        resampling="bilinear",
        bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
    )
