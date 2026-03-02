# Documentation Index

**walkthru-weather-index** — Event-driven weather processing pipeline: NOAA AI-NWP → H3 hexagonal grid → partitioned Parquet on S3.

---

## Contents

| Document | Description |
|---|---|
| [Getting Started](getting-started.md) | Installation, configuration, running locally and on HuggingFace |
| [Pipeline Architecture](pipeline.md) | Step-by-step data flow, module responsibilities |
| [Mathematics](mathematics.md) | All equations: RBF interpolation, topographic corrections, terrain derivatives, derived variables |
| [Weather Variables](variables.md) | Full reference of input variables, output columns, units |
| [Infrastructure](infrastructure.md) | HuggingFace Jobs, GitHub Actions, S3 output schema, OpenTofu |
| [Scientific Review](scientific-review.md) | Audit of all calculations against recent literature, corrections made, references |
| [Global DEM Strategy](global-dem-strategy.md) | Comparison of approaches for global terrain data: H3 GeoParquet vs big COG vs distributed tiles |

---

## What this pipeline does in one paragraph

Every time a new NOAA AI-NWP model file (GraphCast, FourCastNet, or Pangu-Weather) appears in the public NOAA S3 bucket, a GitHub Actions detector triggers a GPU job on HuggingFace. That job downloads the NetCDF forecast, loads a 30 m Copernicus DEM for the target region, generates an H3 hexagonal grid at one or more resolutions, GPU-interpolates all weather variables from the coarse model grid onto the H3 cell centres using Gaussian RBF, applies elevation-aware topographic corrections, computes derived variables (wind speed, shear, moisture flux, geopotential anomaly), and writes the result as Hive-partitioned Parquet files directly to S3 — zero local temp files, zero NaN values.

---

## Quick reference

```bash
# Local test (CPU, 9 cells, no S3 needed)
uv run python main.py --no-gpu --h3-resolutions 5

# GPU run with S3 output
uv run python main.py --h3-resolutions 7,9 --s3-bucket my-bucket

# Submit one-off HuggingFace Job
HF_TOKEN=hf_xxx uv run python scripts/submit_hf_job.py

# Register recurring HF schedule (run once)
HF_TOKEN=hf_xxx uv run python scripts/create_scheduled_job.py
```
