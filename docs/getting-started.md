# Getting Started

## Prerequisites

| Tool | Version | Purpose |
|---|---|---|
| Python | ≥ 3.12 | Runtime |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.10 | Package management |
| Git | any | Version control |
| CUDA 12.x *(optional)* | 12.0+ | GPU acceleration via CuPy |

---

## Installation

```bash
git clone https://github.com/walkthru-earth/walkthru-weather-index
cd walkthru-weather-index

# CPU only (local testing, orchestrator nodes)
uv sync

# GPU (HuggingFace Jobs / RunPod / Vast.ai)
uv sync --extra gpu     # installs cupy-cuda12x
```

---

## Configuration

All pipeline parameters live in **`pipeline/config.py`**. The bounding box defaults to **global** but can be overridden via `--bbox` CLI arg or `BBOX` env var.

```python
# Default: global
BBOX = {
    "min_lat": -90.0, "max_lat": 90.0,
    "min_lon": -180.0, "max_lon": 180.0,
}

# H3 resolutions to process (can be one or many)
H3_RESOLUTIONS = [5, 7]

# Extra degrees added to BBOX when loading source weather data.
# Ensures all H3 cell centres are surrounded by source grid points → no NaN.
# Rule: ≥ (H3_cell_diameter / 2) + model_grid_resolution
# Default 2.0° is safe for all resolutions ≤ 9 with 0.25° model grids.
WEATHER_PADDING = 2.0
```

### H3 resolution reference

| Resolution | Cell area (km²) | Typical use |
|---|---|---|
| 3 | ~12 392 | Global overview |
| 5 | ~253 | Continental / regional |
| 7 | ~5 | City-level |
| 9 | ~0.1 | Street-level |

---

## Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required for | Description |
|---|---|---|
| `HF_TOKEN` | HF Jobs | HuggingFace write-access token |
| `AWS_ACCESS_KEY_ID` | S3 output | Credentials for *your* output bucket |
| `AWS_SECRET_ACCESS_KEY` | S3 output | — |
| `AWS_DEFAULT_REGION` | S3 output | e.g. `us-east-1` |
| `S3_BUCKET` | S3 output | Bare bucket name (no `s3://`, no trailing `/`) |
| `S3_PREFIX` | S3 output | Key prefix inside the bucket (e.g. `indices/v1`); leave empty for root |
| `NOAA_FILE` | Event-driven | Set automatically by the detector workflow |
| `MODEL_NAME` | Optional | Override model (default: `GraphCast_GFS`) |
| `H3_RESOLUTIONS` | Optional | Override resolutions (default: from config.py) |
| `BBOX` | Optional | `min_lat,max_lat,min_lon,max_lon` (default: global) |

> **The NOAA input bucket (`noaa-oar-mlwp-data`) is public — no AWS credentials needed to read it.**

---

## Running locally

### Minimal test (no GPU, no S3, 9 cells)

```bash
uv run python main.py --no-gpu --h3-resolutions 5
```

Output is written to `./output/weather/model=.../`.

### Full local run with GPU

```bash
uv run python main.py --h3-resolutions 7,9
```

### Write to S3

```bash
AWS_ACCESS_KEY_ID=xxx \
AWS_SECRET_ACCESS_KEY=yyy \
uv run python main.py --h3-resolutions 7,9 --s3-bucket my-bucket --s3-prefix my/prefix
```

### CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--model` | `GraphCast_GFS` | Model name (must match a key in `AI_MODELS`) |
| `--h3-resolutions` | from `config.py` | Comma-separated list, e.g. `7,9` |
| `--bbox` | global | `min_lat,max_lat,min_lon,max_lon` |
| `--s3-bucket` | `$S3_BUCKET` env | Output bucket; omit to write locally |
| `--s3-prefix` | `$S3_PREFIX` env | Key prefix inside the bucket |
| `--noaa-file` | latest on S3 | Specific S3 key to process |
| `--no-gpu` | off | Force CPU mode (numpy + scipy) |
| `--no-parquet-dem` | off | Force STAC raster DEM even when Parquet is available |

---

## Running on HuggingFace Jobs

### One-off job

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export S3_BUCKET=my-bucket

uv run python scripts/submit_hf_job.py
```

### Recurring scheduled job (run once to register)

```bash
uv run python scripts/create_scheduled_job.py
# Registers a cron job on HuggingFace: runs at 01:00 and 13:00 UTC daily
```

### Monitor

Visit [huggingface.co/jobs](https://huggingface.co/jobs) to see run status and logs.

---

## GitHub Actions setup

Add these secrets to your GitHub repository (`Settings → Secrets → Actions`):

| Secret | Description |
|---|---|
| `HF_TOKEN` | HuggingFace write token |
| `AWS_ACCESS_KEY_ID` | S3 output credentials |
| `AWS_SECRET_ACCESS_KEY` | S3 output credentials |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `S3_BUCKET` | Bare output bucket name |
| `S3_PREFIX` | Key prefix inside bucket (leave empty for root) |

The detector workflow (`.github/workflows/detect-new-data.yml`) runs twice daily, checks the public NOAA bucket for new files using `aws s3 ls --no-sign-request`, and triggers the pipeline only when a new file appears. No NOAA credentials are required.
