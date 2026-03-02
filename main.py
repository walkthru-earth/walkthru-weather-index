"""Pipeline entrypoint — used by HuggingFace Jobs, Kaggle kernels, and local runs.

Environment variables (all optional — fall back to config.py defaults):
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

import os
import argparse
from datetime import datetime, timezone

import numpy as np

from pipeline import gpu as _gpu_mod


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
    parser = argparse.ArgumentParser(description="NOAA MLWP → H3 Parquet pipeline")
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

    # GPU status
    if not args.no_gpu:
        print(f"🎮 GPU available : {_gpu_mod.gpu_available()}")
    else:
        import pipeline.gpu as gm

        gm._GPU = False
        print("🖥️  Running in CPU mode (--no-gpu flag set)")

    # Resolve BBOX
    from pipeline.config import BBOX as DEFAULT_BBOX

    if args.bbox:
        bbox = _parse_bbox(args.bbox)
    else:
        bbox = DEFAULT_BBOX
    print(
        f"🗺️  BBOX: lat {bbox['min_lat']:.1f}→{bbox['max_lat']:.1f}, "
        f"lon {bbox['min_lon']:.1f}→{bbox['max_lon']:.1f}"
    )

    # Resolve H3 resolutions
    from pipeline.config import H3_RESOLUTIONS as DEFAULT_RESOLUTIONS

    if args.h3_resolutions:
        resolutions = [int(r.strip()) for r in args.h3_resolutions.split(",")]
    else:
        resolutions = DEFAULT_RESOLUTIONS
    print(f"🔢 H3 resolutions : {resolutions}")

    # ── Step 1: load weather data ─────────────────────────────────────────────
    from pipeline.weather import load_weather

    print(f"\n📦 Loading weather data  [{args.model}] …")
    weather_ds, nc_path = load_weather(
        model_name=args.model,
        s3_key=args.noaa_file or None,
        bbox=bbox,
    )
    print(f"   Region dims: {dict(weather_ds.sizes)}")

    # ── Step 2: generate H3 grids ─────────────────────────────────────────────
    from pipeline.h3_grid import generate_h3_grid

    print(f"\n🔢 Generating H3 grids for resolutions {resolutions} …")
    h3_grids = generate_h3_grid(bbox=bbox, resolutions=resolutions)

    # ── Step 3: load DEM (per resolution from Parquet, or single raster) ─────
    from pipeline.dem import load_dem, load_dem_parquet

    dem_by_res: dict[int, dict] = {}
    raster_dem = None  # lazy-loaded only if needed

    for res in resolutions:
        dem = None
        if not args.no_parquet_dem:
            dem = load_dem_parquet(h3_res=res, h3_df=h3_grids[res])
        if dem is None:
            # Fallback: load raster DEM once, reuse for all resolutions
            if raster_dem is None:
                print("\n🏔️  Loading raster DEM (STAC fallback) …")
                raster_dem = load_dem(bbox=bbox)
            dem = raster_dem
        dem_by_res[res] = dem

    # Compute reference elevation (mean across the first resolution's DEM)
    first_dem = dem_by_res[resolutions[0]]
    reference_elevation = float(np.nanmean(first_dem["elev"]))
    print(f"   📏 Reference elevation: {reference_elevation:.0f} m (DEM mean)")

    # ── Step 4: interpolate variables for each resolution ────────────────────
    from pipeline.variables import extract_all

    interpolated_by_res: dict[int, dict] = {}

    for res, h3_df in h3_grids.items():
        print(f"\n🔄 Interpolating → H3 res {res}  ({len(h3_df):,} cells) …")
        interpolated_by_res[res] = extract_all(
            ds=weather_ds,
            tgt_lats=h3_df["lat"].values,
            tgt_lons=h3_df["lon"].values,
            dem=dem_by_res[res],
            model_name=args.model,
            reference_elevation=reference_elevation,
        )

    # ── Step 5: export to S3 ──────────────────────────────────────────────────
    from pipeline.export import write_to_s3

    s3_bucket = args.s3_bucket
    if not s3_bucket:
        print("\n⚠️  No S3 bucket set — writing locally to ./output/")

    print(f"\n💾 Writing partitioned Parquet {'to S3' if s3_bucket else 'locally'} …")
    uri = write_to_s3(
        h3_grids=h3_grids,
        interpolated=interpolated_by_res,
        model_name=args.model,
        s3_bucket=s3_bucket or "output",
        s3_prefix=args.s3_prefix,
        run_time=datetime.now(tz=timezone.utc),
    )

    print(f"\n🎉 Done!  Output → {uri}")


if __name__ == "__main__":
    main()
