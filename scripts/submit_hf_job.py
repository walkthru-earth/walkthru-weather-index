"""Submit a one-shot HuggingFace Job from GitHub Actions (or locally).

Required env vars:
  HF_TOKEN          -- HuggingFace access token (needs write access)

Optional env vars (forwarded as job secrets/env):
  HF_SPACE_ID       -- e.g. walkthru-earth/walkthru-weather-index (default)
  HF_JOB_FLAVOR     -- hardware flavor (default: a10g-large)
  NOAA_FILE         -- S3 key of the new .nc file
  MODEL_NAME        -- e.g. GraphCast_GFS
  H3_RESOLUTIONS    -- e.g. 5,7
  BBOX              -- min_lat,max_lat,min_lon,max_lon (default: global)
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_DEFAULT_REGION
  S3_BUCKET
  S3_PREFIX
"""

from __future__ import annotations

import logging
import os

from huggingface_hub import run_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

HF_TOKEN = os.environ["HF_TOKEN"]
SPACE_ID = os.environ.get("HF_SPACE_ID", "walkthru-earth/walkthru-weather-index")
FLAVOR = os.environ.get("HF_JOB_FLAVOR", "a10g-large")

# Secrets forwarded into the HF container (encrypted at rest by HF)
secrets = {
    k: os.environ[k]
    for k in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "S3_BUCKET",
        "S3_PREFIX",
    )
    if os.environ.get(k)
}

# Plain env vars (non-sensitive pipeline config)
env = {
    k: os.environ[k]
    for k in (
        "NOAA_FILE",
        "MODEL_NAME",
        "H3_RESOLUTIONS",
        "BBOX",
    )
    if os.environ.get(k)
}
env["PYTHONUNBUFFERED"] = "1"

# Build the command -- secrets are passed as env vars inside the container,
# but CLI args are needed for main.py to pick them up.
cmd = [
    "uv",
    "run",
    "python",
    "main.py",
    "--model",
    env.get("MODEL_NAME", "GraphCast_GFS"),
    "--h3-resolutions",
    env.get("H3_RESOLUTIONS", "0,1,2,3,4,5"),
]
if "BBOX" in env:
    cmd += ["--bbox", env["BBOX"]]
if "NOAA_FILE" in env:
    cmd += ["--noaa-file", env["NOAA_FILE"]]
if "S3_BUCKET" in secrets:
    cmd += ["--s3-bucket", secrets["S3_BUCKET"]]
if "S3_PREFIX" in secrets:
    cmd += ["--s3-prefix", secrets["S3_PREFIX"]]

log.info("Submitting HF Job")
log.info("  Space   : %s", SPACE_ID)
log.info("  Flavor  : %s", FLAVOR)
log.info("  Command : %s", " ".join(cmd))
log.info("  Env     : %s", env)
log.info("  Secrets : %s", list(secrets.keys()))

job = run_job(
    # Use the Docker image built from the HF Space
    image=f"hf.co/spaces/{SPACE_ID}",
    command=cmd,
    flavor=FLAVOR,
    secrets=secrets,
    env=env,
    timeout="2h",
    token=HF_TOKEN,
)

log.info("  Job ID  : %s", job.id)
log.info("  URL     : %s", job.url)
log.info("Job submitted -- monitor at https://huggingface.co/jobs")
