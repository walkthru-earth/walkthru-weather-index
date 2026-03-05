---
title: walkthru-weather-index
emoji: 🌤️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# walkthru-weather-index

**Event-driven weather downscaling pipeline**

Downloads NOAA AI-NWP model output (GraphCast, FourCastNet, Pangu-Weather), interpolates all weather variables onto an H3 hexagonal grid using GPU-accelerated bilinear interpolation, applies physically-based topographic corrections using pre-computed H3-indexed terrain data, and writes partitioned Parquet directly to S3.

## Architecture

```
NOAA S3 (public) --> GitHub Actions detector --> HuggingFace Jobs (A10G GPU)
                     (polls every 12h, free)     (triggered on new .nc file)
                                                       |
                                                       v
                                            S3: {prefix}/weather/model=X/date=Y/...
```

**Code** lives on GitHub (`walkthru-earth/walkthru-weather-index`).
**Compute** runs on HuggingFace Jobs under a personal HF Pro account.
The GitHub Actions workflow triggers HF Jobs via the Hub API -- no GPU needed on GitHub's side.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/walkthru-earth/walkthru-weather-index
cd walkthru-weather-index
uv sync                     # CPU only
uv sync --extra gpu         # + CuPy (requires CUDA 12)

# 2. Copy and fill env vars
cp .env.example .env

# 3. Run locally (CPU, no S3)
uv run python main.py --no-gpu --h3-resolutions 5

# 4. Run with GPU + S3 output
uv run python main.py --h3-resolutions 5 \
  --s3-bucket your-bucket --s3-prefix your/prefix

# 5. Submit a one-off HF Job
HF_TOKEN=hf_xxx uv run python scripts/submit_hf_job.py

# 6. Register recurring HF scheduled job (run once)
HF_TOKEN=hf_xxx uv run python scripts/create_scheduled_job.py
```

## Setup

### 1. GitHub repo secrets

Go to **Settings > Secrets and variables > Actions** on the GitHub repo and add:

| Secret | Description |
|---|---|
| `HF_TOKEN` | HuggingFace write token (from your personal Pro account) |
| `AWS_ACCESS_KEY_ID` | Output S3 bucket credentials |
| `AWS_SECRET_ACCESS_KEY` | Output S3 bucket credentials |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `S3_BUCKET` | Bare bucket name (no `s3://`, no trailing `/`) |
| `S3_PREFIX` | Key prefix inside the bucket (e.g. `walkthru-earth/indices`); leave empty for bucket root |

### 2. HuggingFace repo

The pipeline runs on [HuggingFace Jobs](https://huggingface.co/docs/hub/jobs), which pulls code from an HF repo. Push the code there:

```bash
# Create the HF repo (one-time)
pip install huggingface_hub
huggingface-cli repo create walkthru-weather-index --type space --space-sdk docker

# Add HF as a second remote and push
git remote add hf https://huggingface.co/spaces/yharby/walkthru-weather-index
git push hf main
```

> **Note:** HF Jobs requires a Pro subscription ($9/month). The `HF_REPO_ID` in the scripts defaults to `yharby/walkthru-weather-index` -- your personal account, since that's where the Pro subscription lives.

### 3. Local `.env` (for local runs only)

```bash
cp .env.example .env
# Fill in your values -- this file is gitignored
```

## S3 output schema

```
s3://{S3_BUCKET}/{S3_PREFIX}/weather/
  model=GraphCast_GFS/
    date=2026-03-05/
      hour=0/
        h3_res=1/data.parquet
        h3_res=2/data.parquet
        h3_res=3/data.parquet
        h3_res=4/data.parquet
        h3_res=5/data.parquet    ~1 GB (42M rows)
```

Single sorted `data.parquet` per partition. Compression: ZSTD level 3. Row groups: 1M rows. Weather values rounded to meteorologically appropriate precision (~63% smaller than raw float32).

**Schema versions:**
- **Current (March 2026+):** `h3_index` is BIGINT (int64), no `geometry`/`lat`/`lon`/`area_km2` columns. Coordinates derivable via DuckDB h3 extension (`h3_cell_to_lat()`, `h3_cell_to_lng()`). 23 columns total.
- **Legacy (pre-March 2026):** `h3_index` was VARCHAR (hex string) with `geometry`, `lat`, `lon`, `area_km2` columns and multiple `part-*.parquet` files per partition. Old files may still exist on S3 for dates before the schema change.

When `S3_PREFIX` is empty, files land directly at `s3://{bucket}/weather/...`.

## H3 resolutions

| res | cell area | typical use |
|---|---|---|
| 3 | ~12 392 km2 | Global overview |
| 5 | ~253 km2 | Continental / regional (default) |
| 7 | ~5 km2 | City-level |
| 9 | ~0.1 km2 | Street-level |

Configure in `pipeline/config.py` > `H3_RESOLUTIONS` or via `--h3-resolutions 5`.

## CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--model` | `GraphCast_GFS` | Model name (must match a key in `AI_MODELS`) |
| `--h3-resolutions` | from `config.py` | Comma-separated, e.g. `5` or `5,7` |
| `--bbox` | global | `min_lat,max_lat,min_lon,max_lon` |
| `--s3-bucket` | `$S3_BUCKET` env | Output bucket; omit to write locally |
| `--s3-prefix` | `$S3_PREFIX` env | Key prefix inside the bucket |
| `--noaa-file` | latest on S3 | Specific S3 key to process |
| `--no-gpu` | off | Force CPU mode (numpy + scipy) |
| `--no-parquet-dem` | off | Force STAC raster DEM (skip Parquet) |

## DEM terrain data

Pre-computed H3-indexed terrain from [walkthru-earth/dem-terrain](https://github.com/walkthru-earth/dem-terrain), hosted on [Source Cooperative](https://source.coop/walkthru-earth/dem-terrain) (public, no auth):

```
s3://us-west-2.opendata.source.coop/walkthru-earth/dem-terrain/v2/h3/h3_res={res}/data.parquet
```

v2 schema: `h3_index` (BIGINT), `elev`, `slope`, `aspect`, `tri`, `tpi` (all FLOAT). Resolutions 1--10 available.

Fallback sources (for res > 7 or with `--no-parquet-dem`):

1. **Copernicus GLO-30** via Microsoft Planetary Computer
2. **OpenLandMap merged 30 m DEM** via `stac.openlandmap.org`

## Documentation

| Document | Description |
|---|---|
| [Getting Started](docs/getting-started.md) | Installation, configuration, running locally and on HuggingFace |
| [Pipeline Architecture](docs/pipeline.md) | Step-by-step data flow, module responsibilities |
| [Mathematics](docs/mathematics.md) | All equations: interpolation, topographic corrections, terrain derivatives |
| [Weather Variables](docs/variables.md) | Full reference of input variables, output columns, units |
| [Infrastructure](docs/infrastructure.md) | HuggingFace Jobs, GitHub Actions, S3 output schema |
| [Scientific Review](docs/scientific-review.md) | Scientific audit of all calculations against recent literature |
| [Global DEM Strategy](docs/global-dem-strategy.md) | Design rationale for H3 GeoParquet terrain approach |

## Sources

**Weather**: [NOAA AI-NWP](https://registry.opendata.aws/noaa-oar-mlwp/) — AI Neural Weather Prediction models hosted on AWS Open Data.

> Lam, R., Sanchez-Gonzalez, A., Willson, M., et al. (2023). Learning skillful medium-range global weather forecasting. *Science*, 382(6677), 1416–1421. [doi:10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336)

**Terrain**: [GEDTM-30m](https://doi.org/10.5281/zenodo.14900181) via [walkthru-earth/dem-terrain](https://github.com/walkthru-earth/dem-terrain).

> Ho, Y., Grohmann, C. H., Lindsay, J., Reuter, H. I., Parente, L., Witjes, M., & Hengl, T. (2025). GEDTM30: global ensemble digital terrain model at 30 m and derived multiscale terrain variables. *PeerJ*, 13, e19673. [doi:10.7717/peerj.19673](https://doi.org/10.7717/peerj.19673)

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by [walkthru.earth](https://github.com/walkthru-earth). See [LICENSE](LICENSE) for details. The source [NOAA AI-NWP](https://registry.opendata.aws/noaa-oar-mlwp/) data is public domain (US Government work).

Contact: [hi@walkthru.earth](mailto:hi@walkthru.earth)
