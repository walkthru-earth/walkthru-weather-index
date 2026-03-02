"""One-time script: register a recurring HuggingFace scheduled job.

Run this once locally after setting your HF_TOKEN:

  uv run python scripts/create_scheduled_job.py

The schedule is stored on HuggingFace's side -- no cron or GitHub Actions
needed for the timing. The NOAA detector (detect-new-data.yml) is still
used for event-driven runs; this script is for the periodic fallback.
"""

from __future__ import annotations

import logging
import os

from huggingface_hub import create_scheduled_job

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

HF_TOKEN = os.environ["HF_TOKEN"]
SPACE_ID = os.environ.get("HF_SPACE_ID", "yharby/walkthru-weather-index")

# Build command with S3 args baked in
cmd = [
    "uv",
    "run",
    "python",
    "main.py",
    "--model",
    "GraphCast_GFS",
    "--h3-resolutions",
    "5",
]
if os.environ.get("BBOX"):
    cmd += ["--bbox", os.environ["BBOX"]]
if os.environ.get("S3_BUCKET"):
    cmd += ["--s3-bucket", os.environ["S3_BUCKET"]]
if os.environ.get("S3_PREFIX"):
    cmd += ["--s3-prefix", os.environ["S3_PREFIX"]]

secrets = {
    k: v
    for k, v in {
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "S3_BUCKET": os.environ["S3_BUCKET"],
        "S3_PREFIX": os.environ.get("S3_PREFIX", ""),
    }.items()
    if v
}

job = create_scheduled_job(
    image=f"hf.co/spaces/{SPACE_ID}",
    command=cmd,
    schedule="0 1,13 * * *",  # 01:00 and 13:00 UTC (1h after NOAA 00Z/12Z updates)
    flavor="a10g-small",  # A10G 24 GB, 4 vCPU, 15 GB RAM
    secrets=secrets,
    env={"PYTHONUNBUFFERED": "1"},
    timeout="2h",
    token=HF_TOKEN,
)

log.info("Scheduled job created")
log.info("  ID      : %s", job.id)
log.info("  Schedule: 0 1,13 * * * UTC")
log.info("  Flavor  : a10g-small")
log.info("  URL     : https://huggingface.co/jobs")
