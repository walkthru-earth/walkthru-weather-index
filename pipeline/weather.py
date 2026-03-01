"""NOAA AI-NWP S3 data loading.

Public bucket — no credentials required (--no-sign-request / UNSIGNED).
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
import boto3

from pipeline.config import (
    AI_MODELS,
    BBOX,
    CACHE_DIR,
    FORECAST_DAYS,
    S3_BUCKET,
    WEATHER_PADDING,
)

_UNSIGNED_CFG = Config(signature_version=UNSIGNED)


def _s3() -> boto3.client:
    return boto3.client("s3", config=_UNSIGNED_CFG)


def latest_s3_key(model_name: str = "GraphCast_GFS") -> str:
    """Return the S3 key of the most recently modified .nc file for *model_name*."""
    code = AI_MODELS[model_name]["code"]
    client = _s3()

    all_keys: list[tuple] = []  # (LastModified, Key)
    kwargs: dict = {"Bucket": S3_BUCKET, "Prefix": f"{code}/", "MaxKeys": 1000}

    while True:
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            if obj["Key"].endswith(".nc"):
                all_keys.append((obj["LastModified"], obj["Key"]))
        if not resp.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]

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
        print(f"   📥 Downloading {s3_key} …")
        _evict_old_cache(cache_dir, code)
        _s3().download_file(S3_BUCKET, s3_key, str(nc_path))
        print(f"   ✅ Saved → {nc_path.name}  ({nc_path.stat().st_size:,} bytes)")
    else:
        print(f"   📦 Using cache: {nc_path.name}")

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
        print(f"   ℹ️  Pressure levels ({len(ds.level)}): {ds.level.values.tolist()}")

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
    print(
        f"   🗺️  Source region: "
        f"lat {region.latitude.values[0]:.2f}→{region.latitude.values[-1]:.2f} "
        f"({len(region.latitude)} pts)  "
        f"lon {region.longitude.values[0]:.2f}→{region.longitude.values[-1]:.2f} "
        f"({len(region.longitude)} pts)"
    )

    # Limit to forecast window
    max_ts = FORECAST_DAYS * 4 + 1
    if "time" in region.dims and len(region.time) > max_ts:
        region = region.isel(time=slice(0, max_ts))

    return region, nc_path


def _evict_old_cache(cache_dir: Path, code: str) -> None:
    for old in cache_dir.glob(f"*{code}*.nc"):
        old.unlink()
