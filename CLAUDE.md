# CLAUDE.md

## Project

**walkthru-weather-index** — Event-driven weather downscaling pipeline: NOAA AI-NWP → H3 hexagonal grid → partitioned Parquet on S3.

## Commands

```bash
uv sync                               # Install deps (CPU)
uv sync --extra gpu                   # Install deps (GPU)
uv run ruff check .                   # Lint
uv run ruff format .                  # Format
uv run python main.py --no-gpu --h3-resolutions 5  # Local test
```

## Architecture

```
NOAA S3 (public) → GitHub Actions (polls 2x/day) → HuggingFace Jobs (A10G GPU)
                                                         │
                    ┌────────────────────────────────────┘
                    ▼
    weather.py → h3_grid.py → dem.py → variables.py → export.py → S3 Parquet
```

## Key design decisions

- **DEM from Parquet (not STAC)**: `pipeline/dem.py` loads pre-computed H3-indexed terrain from Source Cooperative (`s3://us-west-2.opendata.source.coop/walkthru-earth/dem-terrain/`). Resolutions 1–7 available. Falls back to STAC raster for res > 7 or with `--no-parquet-dem`.
- **H3-native DEM**: When `dem["h3_native"]` is True, `corrections.py` skips `RegularGridInterpolator` entirely — values are already at cell centers.
- **Global H3 grids**: `h3_grid.py` uses `h3.uncompact_cells(get_res0_cells(), res)` for global bbox (LatLngPoly can't represent the full globe).
- **Default resolutions**: `[5, 7]` — max 7 until res 8–10 Parquet files land, then bump `DEM_PARQUET_MAX_RES` in `config.py`.

## DEM terrain dataset

Separate project at `../dem/` (or `walkthru-earth/dem-terrain` on GitHub). Generates GEDTM-30m → H3 Parquet via DuckDB 1.5 with native Parquet 2.11+ GEOMETRY. Hosted on Source Cooperative (public, no auth).

- Res 1–7: uploaded and live
- Res 8–10: processing on Verda CPU node (360 vCPU, 1440 GB RAM), coming soon
- When ready: bump `DEM_PARQUET_MAX_RES` in `pipeline/config.py` and add res 8+ to defaults

## Deployment

- **GitHub → HuggingFace**: code is pushed to both remotes. HF Space builds a Docker image.
- **Trigger**: `gh workflow run trigger-hf-job.yml` or automatic via detect-new-data.yml schedule.
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
  interpolation.py               GPU Gaussian kernel / CPU bilinear interpolation
  corrections.py                 Topographic corrections (lapse rate, wind, precip, humidity)
  variables.py                   Variable extraction, unit conversion, derived quantities
  export.py                      Hive-partitioned Parquet → S3
scripts/
  submit_hf_job.py               Submit one-shot HF Job
  create_scheduled_job.py        Register recurring HF schedule
.github/workflows/
  detect-new-data.yml            Poll NOAA S3 every 12h
  trigger-hf-job.yml             Submit HF Job from GitHub Actions
```
