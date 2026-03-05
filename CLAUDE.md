# CLAUDE.md

## Project

**walkthru-weather-index** -- Event-driven weather downscaling pipeline: NOAA AI-NWP > H3 hexagonal grid > partitioned Parquet on S3.

Part of the [walkthru-earth](https://github.com/walkthru-earth) index family alongside `dem-terrain`, `walkthru-building-index`, and `walkthru-pop-index`.

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

## Source data

- **NOAA AI-NWP**: AWS bucket `noaa-oar-mlwp-data` (public). GraphCast_GFS 0.25° global, 6-hourly, NetCDF.
  - Citation: Lam, R. et al. (2023). Learning skillful medium-range global weather forecasting. *Science*, 382(6677), 1416–1421. [doi:10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336)
- **DEM terrain** (for topo corrections): `s3://us-west-2.opendata.source.coop/walkthru-earth/dem-terrain/v2/h3/h3_res={res}/data.parquet`
  - Source: [walkthru-earth/dem-terrain](https://github.com/walkthru-earth/dem-terrain). GEDTM-30m, res 1–10.
  - Citation: Ho, Y. et al. (2025). GEDTM30. *PeerJ*, 13, e19673. [doi:10.7717/peerj.19673](https://doi.org/10.7717/peerj.19673)
- **Fallback DEM** (res > `DEM_PARQUET_MAX_RES` or `--no-parquet-dem`): Copernicus GLO-30 via Planetary Computer STAC (`https://planetarycomputer.microsoft.com/api/stac/v1`)

## Architecture

```
NOAA S3 (public) > GitHub Actions (polls 2x/day) > HuggingFace Jobs (A10G GPU)
                                                         |
                    +------------------------------------+
                    v
    weather.py > h3_grid.py > dem.py > variables.py > export.py > S3 Parquet
```

## S3 output layout

```
s3://us-west-2.opendata.source.coop/walkthru-earth/indices/weather/
  model=GraphCast_GFS/
    date=YYYY-MM-DD/
      hour={0,12}/
        h3_res=0/data.parquet      ~103 KB
        h3_res=1/data.parquet      ~525 KB
        h3_res=2/data.parquet      ~3.3 MB
        h3_res=3/data.parquet     ~21.2 MB
        h3_res=4/data.parquet    ~141.2 MB
        h3_res=5/data.parquet    ~931.4 MB  (~42M rows per forecast run)
```

## Key design decisions

- **GPU bilinear interpolation**: `pipeline/interpolation.py` uses CuPy `map_coordinates` (order=1 bilinear, O(N) per timestep) on GPU, with scipy `RegularGridInterpolator` CPU fallback. Handles global longitude wrap-around via circular padding.
- **DEM from Parquet (not STAC)**: `pipeline/dem.py` loads pre-computed H3-indexed terrain from Source Cooperative. Path: `{DEM_PARQUET_BASE}/v2/h3/h3_res={res}/data.parquet`. Falls back to STAC raster for res > `DEM_PARQUET_MAX_RES` or with `--no-parquet-dem`.
- **H3-native DEM**: When `dem["h3_native"]` is True, `corrections.py` skips `RegularGridInterpolator` entirely -- values are already at cell centers.
- **Global H3 grids**: `h3_grid.py` uses `h3.uncompact_cells(get_res0_cells(), res)` for global bbox (LatLngPoly can't represent the full globe).
- **Default resolution**: `[0,1,2,3,4,5]` -- `DEM_PARQUET_MAX_RES=10` (all resolutions live). Res 0 uses res 1 DEM aggregated to parent cells.
- **Progressive writes**: Each resolution is written to S3 immediately after interpolation, so partial results survive failures.
- **Structured logging**: Uses Python `logging` module throughout (not print). Flushes per record for real-time HF Jobs log streaming.
- **Lean Parquet output**: DuckDB merges part files into a single sorted `data.parquet` per partition. `h3_index` is BIGINT (int64), no geometry/lat/lon/area_km2 columns. Weather values rounded to meteorologically appropriate precision for ~63% better ZSTD compression. Sorted by `h3_index` for spatial locality and row group pushdown.

## DEM terrain dataset

Separate project at [walkthru-earth/dem-terrain](https://github.com/walkthru-earth/dem-terrain). GEDTM-30m → H3 Parquet via DuckDB 1.5 with `GEOPARQUET_VERSION 'BOTH'`. Hosted on Source Cooperative (public, no auth).

- Res 1–10: all uploaded and live
- DEM Parquet base path in `pipeline/config.py`: `s3://us-west-2.opendata.source.coop/walkthru-earth/dem-terrain`
- DEM Parquet URL pattern in `pipeline/dem.py`: `{base}/v2/h3/h3_res={res}/data.parquet`

## Deployment

- **GitHub > HuggingFace**: code is pushed to both remotes. HF Space builds the Docker image only (idle CMD); Jobs override CMD to run the pipeline.
- **Trigger**: `gh workflow run trigger-hf-job.yml` or automatic via detect-new-data.yml schedule.
- **Partitioning**: `model={name}/date={YYYY-MM-DD}/hour={HH}/h3_res={res}` -- hour is parsed from the NOAA filename init time (e.g. `2026030200` → `hour=0`), not wall-clock.
- **Hardware**: a10g-large (A10G 24 GB, 12 vCPU, 46 GB RAM), 2h timeout.
- **Monitor HF jobs**: `hf jobs ps`, `hf jobs inspect <id>`, `hf jobs logs <id>`
- **AWS profile for Source Coop S3**: `sc-iam`

## Documentation files

- `README.md` — GitHub repo README (code usage)
- `SC_README.md` — Source Cooperative dataset README (uploaded to S3 as `indices/weather/README.md`)
- `docs/` — detailed pipeline, math, variables, infrastructure docs

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

## License

CC BY 4.0 by walkthru-earth. NOAA AI-NWP data is public domain (US Government work).
