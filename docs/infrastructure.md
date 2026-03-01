# Infrastructure

## Architecture overview

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  GitHub (free)                                                  │
 │                                                                 │
 │  detect-new-data.yml          trigger-hf-job.yml               │
 │  ┌─────────────────┐          ┌──────────────────────────────┐  │
 │  │ cron: 01:15 UTC │          │ workflow_dispatch             │  │
 │  │ cron: 13:15 UTC │ ──────►  │ inputs: noaa_file, model     │  │
 │  │                 │          │                              │  │
 │  │ aws s3 ls       │          │ uv run submit_hf_job.py      │  │
 │  │ --no-sign-req   │          └──────────────┬───────────────┘  │
 │  └─────────────────┘                         │                  │
 └─────────────────────────────────────────────┼──────────────────┘
                                               │ HuggingFace Hub API
                                               ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  HuggingFace Jobs (GPU, pay-per-second)                         │
 │                                                                 │
 │  Image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel             │
 │  Flavor: a10g-small (A10G 24 GB, ~$1.00/hr)                     │
 │                                                                 │
 │  uv run python main.py                                          │
 │    → download NOAA NetCDF (public S3, no auth)                  │
 │    → load Copernicus DEM via STAC                               │
 │    → GPU RBF interpolation (CuPy)                               │
 │    → write partitioned Parquet → S3                             │
 └─────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  S3 output bucket (your own)                                    │
 │                                                                 │
 │  s3://bucket/weather/                                           │
 │    model=GraphCast_GFS/date=2026-01-01/hour=0/h3_res=7/         │
 │      part-00000.parquet                                         │
 └─────────────────────────────────────────────────────────────────┘
```

---

## Event-driven trigger

### Why not a fixed schedule?

NOAA publishes new AI-NWP model output approximately at 00Z and 12Z UTC, but the exact timing varies. A fixed cron risks either running before the file is available (and processing stale data) or waiting too long. Polling for file changes is more reliable.

### Detector workflow

The detector (`detect-new-data.yml`) runs twice daily on a free GitHub-hosted runner:

```
01:15 UTC  ← ~75 min after NOAA's 00Z target
13:15 UTC  ← ~75 min after NOAA's 12Z target
```

It lists all model prefixes in the public NOAA bucket (no credentials):

```bash
aws s3 ls s3://noaa-oar-mlwp-data/GRAP_v100_GFS/ \
  --recursive --no-sign-request
```

The latest key is compared to `state/noaa-last-seen.txt` committed in the repo. If a new file is found:
1. The state file is updated and committed (`[skip ci]` prevents recursion)
2. `trigger-hf-job.yml` is dispatched with the new file key

The detector runs in ~10 seconds and costs nothing (free GitHub-hosted runners).

---

## HuggingFace Jobs

### Requirements

- HuggingFace **PRO** account ($9/month) — required to use Jobs
- A write-access HF token (`hf_...`) added to GitHub Secrets as `HF_TOKEN`

### Hardware flavors

| Flavor | GPU | VRAM | $/hr | Recommended for |
|---|---|---|---|---|
| `t4-small` | T4 | 16 GB | $0.40 | Testing, res ≤ 7 |
| `l4x1` | L4 | 24 GB | $0.80 | res ≤ 8 |
| `a10g-small` | A10G | 24 GB | **$1.00** | **Default — res 7+9** |
| `a100-large` | A100 | 80 GB | $2.50 | Global / res ≥ 9 |

### One-off job

```python
from huggingface_hub import run_job

run_job(
    repo_id = "yharby/walkthru-weather-index",
    command = ["uv", "run", "python", "main.py",
               "--model", "GraphCast_GFS",
               "--h3-resolutions", "7,9"],
    flavor  = "a10g-small",
    secrets = {
        "AWS_ACCESS_KEY_ID":     "...",
        "AWS_SECRET_ACCESS_KEY": "...",
        "S3_BUCKET":             "my-bucket",
    },
    timeout = "3h",
)
```

### Scheduled job (registered once)

```python
from huggingface_hub import create_scheduled_job

create_scheduled_job(
    repo_id  = "yharby/walkthru-weather-index",
    command  = ["uv", "run", "python", "main.py",
                "--model", "GraphCast_GFS",
                "--h3-resolutions", "7,9"],
    schedule = "0 1,13 * * *",   # 01:00 and 13:00 UTC daily
    flavor   = "a10g-small",
    secrets  = { ... },
    timeout  = "3h",
)
```

The schedule is stored entirely on HuggingFace's side — no GitHub Actions cron needed for periodic runs. The GitHub Actions detector is still used for event-driven (new file) triggering.

---

## S3 output schema

### Partition layout

```
s3://{S3_BUCKET}/{S3_PREFIX}/weather/
  model={model_name}/
    date={YYYY-MM-DD}/
      hour={H}/
        h3_res={resolution}/
          part-00000.parquet
```

### File settings

| Setting | Value |
|---|---|
| Compression | ZSTD level 3 |
| Row groups | 100 000 rows |
| Max file size | 500 000 rows (~128 MB) |
| Statistics | Enabled (min/max on all columns) |

### Querying with DuckDB

```sql
-- Install and load httpfs extension for S3 access
INSTALL httpfs; LOAD httpfs;
SET s3_region = 'us-east-1';

-- Read a specific partition
SELECT
    h3_index,
    timestamp,
    temperature_2m_C,
    wind_speed_10m_ms,
    precipitation_mm_6hr
FROM read_parquet(
    's3://my-bucket/weather/model=GraphCast_GFS/date=2026-01-01/hour=0/h3_res=7/*.parquet'
)
ORDER BY timestamp, h3_index;

-- Read across all dates with partition pruning
SELECT date, AVG(temperature_2m_C) AS mean_temp
FROM read_parquet('s3://my-bucket/weather/**/*.parquet', hive_partitioning=true)
WHERE model = 'GraphCast_GFS'
  AND h3_res = 7
GROUP BY date
ORDER BY date;
```

### Querying with PyArrow

```python
import pyarrow.dataset as ds
from pyarrow.fs import S3FileSystem

s3 = S3FileSystem(region="us-east-1")
dataset = ds.dataset(
    "my-bucket/weather",
    filesystem=s3,
    format="parquet",
    partitioning="hive",
)

# Filter using partition pruning + column predicate
table = dataset.to_table(
    filter=(
        (ds.field("model")  == "GraphCast_GFS") &
        (ds.field("h3_res") == 7) &
        (ds.field("date")   >= "2026-01-01")
    ),
    columns=["h3_index", "timestamp", "temperature_2m_C",
             "wind_speed_10m_ms", "precipitation_mm_6hr"],
)
df = table.to_pandas()
```

---

## GitHub Actions secrets

| Secret | Where used | Description |
|---|---|---|
| `HF_TOKEN` | `trigger-hf-job.yml` | HuggingFace write-access token |
| `AWS_ACCESS_KEY_ID` | `trigger-hf-job.yml` | Credentials for output S3 bucket |
| `AWS_SECRET_ACCESS_KEY` | `trigger-hf-job.yml` | — |
| `AWS_DEFAULT_REGION` | `trigger-hf-job.yml` | e.g. `us-east-1` |
| `S3_BUCKET` | `trigger-hf-job.yml` | Bare output bucket name |
| `S3_PREFIX` | `trigger-hf-job.yml` | Key prefix inside the bucket (optional) |

> The NOAA input bucket is public. The detector workflow uses zero credentials.

---

## Alternative compute backends

See [PIPELINE_ARCHITECTURE.md](../PIPELINE_ARCHITECTURE.md) for full cost comparison. Quick reference:

| Backend | GPU | $/run (30 min) | Automation |
|---|---|---|---|
| **HF Jobs a10g-small** | A10G 24 GB | ~$0.50 | Native API + cron |
| RunPod RTX 4090 spot | RTX 4090 24 GB | ~$0.17 | REST API / OpenTofu |
| Vast.ai A100 bid | A100 40 GB | ~$0.15 | REST API |
| Kaggle (free) | T4 16 GB | $0.00 | `kaggle kernels push` |
| Hetzner CCX53 (CPU) | None | ~$0.17 | hcloud API / OpenTofu |
