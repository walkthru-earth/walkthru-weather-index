---
title: walkthru-weather-index
emoji: "\U0001F326"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: cc-by-4.0
---

# walkthru-weather-index

**Event-driven weather downscaling pipeline**

Downloads NOAA AI-NWP model output (GraphCast, FourCastNet, Pangu-Weather), interpolates all weather variables onto an H3 hexagonal grid at multiple resolutions using GPU-accelerated Gaussian kernel smoothing, applies physically-based topographic corrections from Copernicus 30 m DEM, and writes partitioned Parquet directly to S3.

## Architecture

```
NOAA S3 (public) ──► GitHub Actions detector ──► HuggingFace Jobs (A10G GPU)
                      (polls every 12h, free)      (triggered on new .nc file)
                                                         │
                                                         ▼
                                              S3: {prefix}/weather/model=X/date=Y/…
```

**Code** lives on GitHub (`walkthru-earth/walkthru-weather-index`).
**Compute** runs on HuggingFace Jobs under a personal HF Pro account.
The GitHub Actions workflow triggers HF Jobs via the Hub API — no GPU needed on GitHub's side.

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/walkthru-earth/walkthru-weather-index
cd walkthru-weather-index
uv sync                     # CPU only
uv sync --extra gpu         # + CuPy (requires CUDA 12)

# 2. Copy and fill env vars
cp .env.example .env

# 3. Run locally (CPU, small resolution, no S3)
uv run python main.py --no-gpu --h3-resolutions 5

# 4. Run with GPU + S3 output
uv run python main.py --h3-resolutions 5,7 \
  --s3-bucket your-bucket --s3-prefix your/prefix

# 5. Submit a one-off HF Job
HF_TOKEN=hf_xxx uv run python scripts/submit_hf_job.py

# 6. Register recurring HF scheduled job (run once)
HF_TOKEN=hf_xxx uv run python scripts/create_scheduled_job.py
```

## Setup

### 1. GitHub repo secrets

Go to **Settings → Secrets and variables → Actions** on the GitHub repo and add:

| Secret | Description |
|---|---|
| `HF_TOKEN` | HuggingFace write token (from your personal Pro account) |
| `AWS_ACCESS_KEY_ID` | Output S3 bucket credentials |
| `AWS_SECRET_ACCESS_KEY` | Output S3 bucket credentials |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `S3_BUCKET` | Bare bucket name (no `s3://`, no trailing `/`) |
| `S3_PREFIX` | Key prefix inside the bucket (e.g. `indices/v1`); leave empty for bucket root |

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

> **Note:** HF Jobs requires a Pro subscription ($9/month). The `HF_REPO_ID` in the scripts defaults to `yharby/walkthru-weather-index` — your personal account, since that's where the Pro subscription lives.

### 3. Local `.env` (for local runs only)

```bash
cp .env.example .env
# Fill in your values — this file is gitignored
```

## S3 output schema

```
s3://{S3_BUCKET}/{S3_PREFIX}/weather/
  model=GraphCast_GFS/
    date=2026-01-01/
      hour=0/
        h3_res=7/
          part-00000.parquet
        h3_res=9/
          part-00000.parquet
```

Compression: ZSTD level 3 · row groups: 100k · statistics enabled for predicate pushdown.

When `S3_PREFIX` is empty, files land directly at `s3://{bucket}/weather/…`.

## H3 resolutions

| res | cell area | typical use |
|---|---|---|
| 3 | ~12 392 km² | Global overview |
| 5 | ~253 km² | Continental / regional |
| 7 | ~5 km² | City-level |
| 9 | ~0.1 km² | Street-level |

Configure in `pipeline/config.py` → `H3_RESOLUTIONS` or via `--h3-resolutions 7,9`.

## CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--model` | `GraphCast_GFS` | Model name (must match a key in `AI_MODELS`) |
| `--h3-resolutions` | from `config.py` | Comma-separated, e.g. `7,9` |
| `--bbox` | global | `min_lat,max_lat,min_lon,max_lon` |
| `--s3-bucket` | `$S3_BUCKET` env | Output bucket; omit to write locally |
| `--s3-prefix` | `$S3_PREFIX` env | Key prefix inside the bucket |
| `--noaa-file` | latest on S3 | Specific S3 key to process |
| `--no-gpu` | off | Force CPU mode (numpy + scipy) |
| `--no-parquet-dem` | off | Force STAC raster DEM (skip Parquet) |

## DEM sources

1. **Pre-computed H3 Parquet** on [Source Cooperative](https://source.coop/walkthru-earth/dem-terrain) (primary) — terrain derivatives already at H3 cell centres, res 1–7
2. **Copernicus GLO-30** via Microsoft Planetary Computer (fallback for res > 7)
3. **OpenLandMap merged 30 m DEM** via `stac.openlandmap.org` (fallback #2)

## Documentation

| Document | Description |
|---|---|
| [Getting Started](docs/getting-started.md) | Installation, configuration, running locally and on HuggingFace |
| [Pipeline Architecture](docs/pipeline.md) | Step-by-step data flow, module responsibilities |
| [Mathematics](docs/mathematics.md) | All equations: interpolation, topographic corrections, terrain derivatives |
| [Weather Variables](docs/variables.md) | Full reference of input variables, output columns, units |
| [Infrastructure](docs/infrastructure.md) | HuggingFace Jobs, GitHub Actions, S3 output schema |
| [Scientific Review](docs/scientific-review.md) | Audit of all calculations against recent literature |
| [Global DEM Strategy](docs/global-dem-strategy.md) | Approaches for scaling terrain data globally |

## License

[CC BY 4.0](LICENSE)
