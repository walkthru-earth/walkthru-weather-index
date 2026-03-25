"""NOAA AI-NWP S3 data loading.

Public bucket — no credentials required (--no-sign-request / UNSIGNED).
"""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr
from obstore.store import S3Store

from pipeline.config import (
    AI_MODELS,
    BBOX,
    CACHE_DIR,
    FORECAST_DAYS,
    S3_BUCKET,
    WEATHER_PADDING,
)

log = logging.getLogger(__name__)


def _store() -> S3Store:
    """Public NOAA bucket — no credentials needed."""
    return S3Store(S3_BUCKET, region="us-east-1", skip_signature=True)


def latest_s3_key(model_name: str = "GraphCast_GFS") -> str:
    """Return the S3 key of the most recently modified .nc file for *model_name*."""
    code = AI_MODELS[model_name]["code"]
    store = _store()

    all_keys: list[tuple] = []  # (last_modified, path)
    for chunk in store.list(prefix=f"{code}/"):
        for meta in chunk:
            if meta["path"].endswith(".nc"):
                all_keys.append((meta["last_modified"], meta["path"]))

    if not all_keys:
        raise FileNotFoundError(f"No .nc files found under {code}/ in {S3_BUCKET}")

    all_keys.sort(reverse=True)
    return all_keys[0][1]


def load_weather(
    model_name: str = "GraphCast_GFS",
    s3_key: str | None = None,
    cache_dir: Path = CACHE_DIR,
    bbox: dict | None = None,
) -> tuple[xr.Dataset, Path]:
    """Download (or use cache) the latest NetCDF for *model_name*.

    If *s3_key* is given it is used directly (for event-driven runs where
    the detector already resolved the key).
    """
    if bbox is None:
        bbox = BBOX

    code = AI_MODELS[model_name]["code"]

    if s3_key is None:
        s3_key = latest_s3_key(model_name)

    filename = Path(s3_key).name
    nc_path = cache_dir / filename

    if not nc_path.exists():
        log.info("[LOAD] Downloading %s", s3_key)
        _evict_old_cache(cache_dir, code)
        result = _store().get(s3_key)
        with open(nc_path, "wb") as f:
            for chunk in result.stream(min_chunk_size=8 * 1024 * 1024):
                f.write(chunk)
        log.info(
            "[LOAD] Saved %s (%s bytes)", nc_path.name, f"{nc_path.stat().st_size:,}"
        )
    else:
        log.info("[LOAD] Using cache: %s", nc_path.name)

    ds = xr.open_dataset(nc_path, engine="h5netcdf")

    # Ensure latitude is ascending
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.sortby("latitude")

    # Ensure pressure levels are in ascending order (1000→50 hPa)
    # so that fixed level indices match the expected 13-level ordering.
    if "level" in ds.dims:
        levels = ds.level.values
        if levels[0] < levels[-1]:
            # Levels are ascending (50→1000 hPa from top-of-atmosphere) — reverse
            ds = ds.sortby("level", ascending=False)
        log.info("Pressure levels (%d): %s", len(ds.level), ds.level.values.tolist())

    # For global bbox, skip clipping (use full dataset).
    # Otherwise clip to region of interest + padding.
    is_global = (bbox["max_lat"] - bbox["min_lat"]) >= 170
    if is_global:
        region = ds
    else:
        padded = {
            "min_lat": bbox["min_lat"] - WEATHER_PADDING,
            "max_lat": bbox["max_lat"] + WEATHER_PADDING,
            "min_lon": bbox["min_lon"] - WEATHER_PADDING,
            "max_lon": bbox["max_lon"] + WEATHER_PADDING,
        }
        region = ds.sel(
            latitude=slice(padded["min_lat"], padded["max_lat"]),
            longitude=slice(padded["min_lon"], padded["max_lon"]),
        )
    log.info(
        "[LOAD] Source region: lat %.2f-%.2f (%d pts)  lon %.2f-%.2f (%d pts)",
        region.latitude.values[0],
        region.latitude.values[-1],
        len(region.latitude),
        region.longitude.values[0],
        region.longitude.values[-1],
        len(region.longitude),
    )

    # Limit to forecast window
    max_ts = FORECAST_DAYS * 4 + 1
    if "time" in region.dims and len(region.time) > max_ts:
        region = region.isel(time=slice(0, max_ts))

    return region, nc_path


def _evict_old_cache(cache_dir: Path, code: str) -> None:
    for old in cache_dir.glob(f"*{code}*.nc"):
        old.unlink()
