# CLAUDE.md

## Project

**walkthru-weather-index** -- Event-driven weather downscaling pipeline: NOAA AI-NWP > H3 hexagonal grid > partitioned Parquet on S3.

## Commands

```bash
# Setup
uv sync                               # Install deps (CPU)
uv sync --extra gpu                   # Install deps (GPU)

# Lint & format (ALWAYS run before committing)
uv run ruff check .                   # Lint all files
uv run ruff format .                  # Format all files

# Local test
uv run python main.py --no-gpu --h3-resolutions 5

# GitHub CLI -- trigger and monitor workflows
gh workflow run trigger-hf-job.yml                    # Trigger pipeline
gh run list --workflow=trigger-hf-job.yml             # List recent runs
gh run watch <run-id>                                 # Watch live logs
gh run view <run-id> --log                            # View completed logs

# HuggingFace CLI -- monitor jobs
hf jobs ps                                            # List all jobs
hf jobs inspect <job-id>                              # Job details
hf jobs logs <job-id>                                 # View logs
hf jobs logs -f <job-id>                              # Stream logs live
```

## Architecture

```
NOAA S3 (public) > GitHub Actions (polls 2x/day) > HuggingFace Jobs (A10G GPU)
                                                         |
                    +------------------------------------+
                    v
    weather.py > h3_grid.py > dem.py > variables.py > export.py > S3 Parquet
```

## Key design decisions

- **GPU bilinear interpolation**: `pipeline/interpolation.py` uses CuPy `map_coordinates` (order=1 bilinear, O(N) per timestep) on GPU, with scipy `RegularGridInterpolator` CPU fallback. Handles global longitude wrap-around via circular padding.
- **DEM from Parquet (not STAC)**: `pipeline/dem.py` loads pre-computed H3-indexed terrain from Source Cooperative (`s3://us-west-2.opendata.source.coop/walkthru-earth/dem-terrain/`). Source: [walkthru-earth/dem-terrain](https://github.com/walkthru-earth/dem-terrain). Resolutions 1-7 available. Falls back to STAC raster for res > 7 or with `--no-parquet-dem`.
- **H3-native DEM**: When `dem["h3_native"]` is True, `corrections.py` skips `RegularGridInterpolator` entirely -- values are already at cell centers.
- **Global H3 grids**: `h3_grid.py` uses `h3.uncompact_cells(get_res0_cells(), res)` for global bbox (LatLngPoly can't represent the full globe).
- **Default resolution**: `[5]` -- max 7 until res 8-10 Parquet files land, then bump `DEM_PARQUET_MAX_RES` in `config.py`.
- **Progressive writes**: Each resolution is written to S3 immediately after interpolation, so partial results survive failures.
- **Structured logging**: Uses Python `logging` module throughout (not print). Flushes per record for real-time HF Jobs log streaming.
- **Native Parquet GEOMETRY**: DuckDB post-processes each Parquet partition to add `ST_Point(lon, lat)::GEOMETRY('EPSG:4326')` with per-row-group `BoundingBox` stats for spatial predicate pushdown. Rows sorted by `h3_index` for spatial locality (tight bounding boxes). Same pattern as [dem-terrain](https://github.com/walkthru-earth/dem-terrain).

## DEM terrain dataset

Separate project at [walkthru-earth/dem-terrain](https://github.com/walkthru-earth/dem-terrain). Generates GEDTM-30m > H3 Parquet via DuckDB 1.5 with native Parquet 2.11+ GEOMETRY. Hosted on Source Cooperative (public, no auth).

- Res 1-7: uploaded and live
- Res 8-10: processing, coming soon
- When ready: bump `DEM_PARQUET_MAX_RES` in `pipeline/config.py` and add res 8+ to defaults

## Deployment

- **GitHub > HuggingFace**: code is pushed to both remotes. HF Space builds a Docker image.
- **Trigger**: `gh workflow run trigger-hf-job.yml` or automatic via detect-new-data.yml schedule.
- **Hardware**: a10g-large (A10G 24 GB, 12 vCPU, 46 GB RAM), 2h timeout.
- **Monitor HF jobs**: `hf jobs ps`, `hf jobs inspect <id>`, `hf jobs logs <id>`
- **AWS profile for Source Coop S3**: `sc-iam`

## File layout

```
main.py                          Pipeline entrypoint
pipeline/
  config.py                      All constants, BBOX, H3_RESOLUTIONS, DEM_PARQUET_*
  gpu.py                         CuPy/NumPy abstraction
  weather.py                     NOAA S3 NetCDF download
  h3_grid.py                     H3 cell generation (global + regional)
  dem.py                         DEM loading (Parquet primary, STAC fallback)
  interpolation.py               GPU bilinear (CuPy map_coordinates) / CPU bilinear (scipy)
  corrections.py                 Topographic corrections (lapse rate, wind, precip, humidity)
  variables.py                   Variable extraction, unit conversion, derived quantities
  export.py                      Hive-partitioned Parquet > S3
scripts/
  submit_hf_job.py               Submit one-shot HF Job
  create_scheduled_job.py        Register recurring HF schedule
.github/workflows/
  detect-new-data.yml            Poll NOAA S3 every 12h (GraphCast_GFS only)
  trigger-hf-job.yml             Submit HF Job from GitHub Actions
```
