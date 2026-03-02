"""Pipeline entrypoint -- used by HuggingFace Jobs, Kaggle kernels, and local runs.

Environment variables (all optional -- fall back to config.py defaults):
  NOAA_FILE        S3 key of the new .nc file (set by the detector workflow)
  MODEL_NAME       e.g. GraphCast_GFS  (default: GraphCast_GFS)
  H3_RESOLUTIONS   comma-separated, e.g. 5,7  (default: from config.py)
  BBOX             min_lat,max_lat,min_lon,max_lon  (default: global)
  S3_BUCKET        output bucket name (no s3:// prefix, no trailing slash)
  S3_PREFIX        key prefix inside the bucket (e.g. "indices/v1")
  AWS_DEFAULT_REGION
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from datetime import datetime, timezone

import numpy as np

from pipeline import gpu as _gpu_mod

log = logging.getLogger(__name__)


def _parse_init_time(noaa_file: str | None) -> datetime | None:
    """Extract the forecast initialization time from a NOAA filename.

    Example: 'GRAP_v100_GFS_2026030200_f000_f240_06.nc' -> 2026-03-02 00:00 UTC
    """
    if not noaa_file:
        return None
    m = re.search(r"(\d{10})", noaa_file)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H").replace(tzinfo=timezone.utc)


def _parse_bbox(raw: str) -> dict:
    """Parse 'min_lat,max_lat,min_lon,max_lon' into a bbox dict."""
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"BBOX must be 4 comma-separated floats, got {len(parts)}"
        )
    return {
        "min_lat": parts[0],
        "max_lat": parts[1],
        "min_lon": parts[2],
        "max_lon": parts[3],
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="NOAA MLWP -> H3 Parquet pipeline")
    parser.add_argument("--noaa-file", default=os.environ.get("NOAA_FILE"))
    parser.add_argument(
        "--model", default=os.environ.get("MODEL_NAME", "GraphCast_GFS")
    )
    parser.add_argument(
        "--h3-resolutions", default=os.environ.get("H3_RESOLUTIONS", "")
    )
    parser.add_argument(
        "--bbox",
        default=os.environ.get("BBOX", ""),
        help="Bounding box as min_lat,max_lat,min_lon,max_lon (default: global)",
    )
    parser.add_argument(
        "--s3-bucket",
        default=os.environ.get("S3_BUCKET", os.environ.get("AWS_S3_OUTPUT_BUCKET", "")),
    )
    parser.add_argument(
        "--s3-prefix",
        default=os.environ.get("S3_PREFIX", ""),
        help="Key prefix inside the S3 bucket (e.g. 'indices/v1')",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU mode")
    parser.add_argument(
        "--no-parquet-dem",
        action="store_true",
        help="Force STAC raster DEM even when Parquet is available",
    )
    args = parser.parse_args()

    # Create local dirs
    from pipeline.config import ensure_dirs

    ensure_dirs()

    # GPU status
    if not args.no_gpu:
        log.info("[GPU] GPU available: %s", _gpu_mod.gpu_available())
    else:
        import pipeline.gpu as gm

        gm._GPU = False
        log.info("[CPU] Running in CPU mode (--no-gpu flag set)")

    # Resolve BBOX
    from pipeline.config import BBOX as DEFAULT_BBOX

    if args.bbox:
        bbox = _parse_bbox(args.bbox)
    else:
        bbox = DEFAULT_BBOX
    log.info(
        "BBOX: lat %.1f-%.1f, lon %.1f-%.1f",
        bbox["min_lat"],
        bbox["max_lat"],
        bbox["min_lon"],
        bbox["max_lon"],
    )

    # Resolve H3 resolutions
    from pipeline.config import H3_RESOLUTIONS as DEFAULT_RESOLUTIONS

    if args.h3_resolutions:
        resolutions = [int(r.strip()) for r in args.h3_resolutions.split(",")]
    else:
        resolutions = DEFAULT_RESOLUTIONS
    log.info("H3 resolutions: %s", resolutions)

    # -- Step 1: load weather data ---------------------------------------------
    from pipeline.weather import load_weather

    log.info("[LOAD] Loading weather data [%s]", args.model)
    weather_ds, nc_path = load_weather(
        model_name=args.model,
        s3_key=args.noaa_file or None,
        bbox=bbox,
    )
    log.info("Region dims: %s", dict(weather_ds.sizes))

    # -- Step 2: generate H3 grids ---------------------------------------------
    from pipeline.h3_grid import generate_h3_grid

    log.info("[H3] Generating grids for resolutions %s", resolutions)
    h3_grids = generate_h3_grid(bbox=bbox, resolutions=resolutions)

    # -- Step 3: init export filesystem ----------------------------------------
    from pipeline.export import init_filesystem, write_resolution_to_s3

    s3_bucket = args.s3_bucket
    if not s3_bucket:
        log.warning("No S3 bucket set -- writing locally to ./output/")

    filesystem, base_dir, use_s3 = init_filesystem(
        s3_bucket=s3_bucket or "output",
        s3_prefix=args.s3_prefix,
    )

    # Parse forecast init time from the NOAA filename (e.g. "...2026030200_f000...")
    # so partitions use the model init hour, not wall-clock time.
    run_time = _parse_init_time(args.noaa_file) or datetime.now(tz=timezone.utc)

    # -- Step 4: per-resolution loop: DEM -> interpolate -> write -> free ------
    from pipeline.dem import load_dem, load_dem_parquet
    from pipeline.variables import extract_all

    raster_dem = None  # lazy-loaded only if needed
    reference_elevation: float | None = None

    for res in resolutions:
        h3_df = h3_grids[res]

        # Load DEM for this resolution
        dem = None
        if not args.no_parquet_dem:
            dem = load_dem_parquet(h3_res=res, h3_df=h3_df)
        if dem is None:
            if raster_dem is None:
                log.info("[DEM] Loading raster DEM (STAC fallback)")
                raster_dem = load_dem(bbox=bbox)
            dem = raster_dem

        # Compute reference elevation once from the first resolution's DEM
        if reference_elevation is None:
            reference_elevation = float(np.nanmean(dem["elev"]))
            log.info("Reference elevation: %.0f m (DEM mean)", reference_elevation)

        # Interpolate
        log.info("[INTERP] H3 res %d (%s cells)", res, f"{len(h3_df):,}")
        interpolated = extract_all(
            ds=weather_ds,
            tgt_lats=h3_df["lat"].values,
            tgt_lons=h3_df["lon"].values,
            dem=dem,
            model_name=args.model,
            reference_elevation=reference_elevation,
        )

        # Free weather data before write (no longer needed after extraction)
        if res == resolutions[-1]:
            del weather_ds
            import gc

            gc.collect()
            log.info("[LOAD] Weather data freed")

        # Write immediately
        write_resolution_to_s3(
            res=res,
            h3_df=h3_df,
            interpolated=interpolated,
            model_name=args.model,
            filesystem=filesystem,
            base_dir=base_dir,
            use_s3=use_s3,
            run_time=run_time,
        )

        # Free memory before next resolution
        del dem, interpolated

    uri = f"s3://{base_dir}" if use_s3 else base_dir
    log.info("[DONE] Output: %s", uri)


if __name__ == "__main__":
    main()
