"""Compare NOAA source files against processed output on S3 and print missing NOAA keys.

Usage:
  uv run python scripts/detect_gaps.py --days 7
  uv run python scripts/detect_gaps.py --days 30 --model GraphCast_GFS

Output: one NOAA S3 key per line for each unprocessed forecast run.
Exit code 0 if gaps found, 1 if fully caught up.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from obstore.store import S3Store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# NOAA source bucket (public, no auth)
NOAA_BUCKET = "noaa-oar-mlwp-data"
NOAA_PREFIX_MAP = {
    "GraphCast_GFS": "GRAP_v100_GFS",
}

# Output S3 (needs credentials)
OUTPUT_BUCKET = os.environ.get("S3_BUCKET", "")
OUTPUT_PREFIX = os.environ.get("S3_PREFIX", "")


def _parse_noaa_key(key: str) -> tuple[str, int] | None:
    """Extract (date_str, hour) from a NOAA filename.

    Example: 'GRAP_v100_GFS/2026/0325/GRAP_v100_GFS_2026032500_f000_f240_06.nc'
             -> ('2026-03-25', 0)
    """
    m = re.search(r"(\d{10})_f\d+_f\d+_\d+\.nc$", key)
    if not m:
        return None
    ts = m.group(1)
    dt = datetime.strptime(ts, "%Y%m%d%H")
    return dt.strftime("%Y-%m-%d"), dt.hour


def list_noaa_files(model: str, days: int) -> dict[tuple[str, int], str]:
    """List NOAA files for the last N days. Returns {(date, hour): s3_key}."""
    prefix = NOAA_PREFIX_MAP.get(model)
    if not prefix:
        log.error("Unknown model: %s", model)
        return {}

    store = S3Store(NOAA_BUCKET, region="us-east-1", skip_signature=True)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)

    result: dict[tuple[str, int], str] = {}
    d = cutoff.date()
    today = datetime.now(tz=timezone.utc).date()
    while d <= today:
        day_prefix = f"{prefix}/{d.year}/{d.strftime('%m%d')}/"
        for chunk in store.list(prefix=day_prefix):
            for meta in chunk:
                path = meta["path"]
                if not path.endswith(".nc"):
                    continue
                parsed = _parse_noaa_key(path)
                if parsed:
                    result[parsed] = path
        d += timedelta(days=1)

    return result


def list_output_partitions(model: str, days: int) -> set[tuple[str, int]]:
    """List existing output partitions on S3. Returns {(date, hour)}."""
    if not OUTPUT_BUCKET:
        log.warning("No S3_BUCKET set -- cannot check output, assuming all missing")
        return set()

    # Build store with credentials from env or AWS profile
    profile = os.environ.get("AWS_PROFILE")
    if profile:
        from boto3 import Session as Boto3Session

        from obstore.auth.boto3 import Boto3CredentialProvider

        session = Boto3Session(profile_name=profile)
        cred = Boto3CredentialProvider(session)
        store = S3Store(
            OUTPUT_BUCKET,
            region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
            credential_provider=cred,
        )
    else:
        # Use env vars (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) picked up natively
        store = S3Store(
            OUTPUT_BUCKET,
            region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
        )

    base_parts = [p for p in [OUTPUT_PREFIX, "weather", f"model={model}"] if p]
    base = "/".join(base_parts)

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    result: set[tuple[str, int]] = set()

    d = cutoff.date()
    today = datetime.now(tz=timezone.utc).date()
    while d <= today:
        date_str = d.strftime("%Y-%m-%d")
        for hour in (0, 12):
            key = f"{base}/date={date_str}/hour={hour}/h3_res=5/data.parquet"
            try:
                store.head(key)
                result.add((date_str, hour))
            except FileNotFoundError:
                pass
        d += timedelta(days=1)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect unprocessed NOAA forecast runs"
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Look back N days (default: 7)"
    )
    parser.add_argument("--model", default="GraphCast_GFS")
    args = parser.parse_args()

    log.info("Scanning last %d days for model=%s", args.days, args.model)

    noaa_files = list_noaa_files(args.model, args.days)
    log.info("NOAA source files found: %d", len(noaa_files))

    output_done = list_output_partitions(args.model, args.days)
    log.info("Output partitions found: %d", len(output_done))

    missing = {k: v for k, v in sorted(noaa_files.items()) if k not in output_done}
    log.info("Missing (to process): %d", len(missing))

    if not missing:
        log.info("Fully caught up -- nothing to do")
        sys.exit(1)

    for (date_str, hour), noaa_key in missing.items():
        log.info("  GAP: date=%s hour=%d -> %s", date_str, hour, noaa_key)
        print(noaa_key)


if __name__ == "__main__":
    main()
