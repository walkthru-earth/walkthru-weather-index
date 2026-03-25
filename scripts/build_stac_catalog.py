"""Build a STAC GeoParquet catalog from existing weather index partitions on S3.

Discovers processed forecast runs by checking which NOAA source files have
corresponding output on S3 (via DuckDB parquet_metadata). Writes a single
catalog.parquet as a STAC GeoParquet with geo + stac-geoparquet file metadata.

Usage:
  uv run python scripts/build_stac_catalog.py
  uv run python scripts/build_stac_catalog.py --days 30
  uv run python scripts/build_stac_catalog.py --local /tmp/catalog.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import duckdb
from obstore.store import S3Store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

OUTPUT_BUCKET = os.environ.get("S3_BUCKET", "")
OUTPUT_PREFIX = os.environ.get("S3_PREFIX", "")
MODEL = "GraphCast_GFS"
COLLECTION_ID = "walkthru-weather-index"

# NOAA source
NOAA_BUCKET = "noaa-oar-mlwp-data"
NOAA_PREFIX_MAP = {"GraphCast_GFS": "GRAP_v100_GFS"}

# STAC forecast extension
FORECAST_EXT = "https://stac-extensions.github.io/forecast/v0.2.0/schema.json"


def _s3_base() -> str:
    if OUTPUT_PREFIX:
        return f"s3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}/weather"
    return f"s3://{OUTPUT_BUCKET}/weather"


def _https_base() -> str:
    """Public HTTPS URL for Source Cooperative."""
    if OUTPUT_PREFIX:
        return f"https://data.source.coop/{OUTPUT_PREFIX}/weather"
    return ""


def _geo_metadata() -> dict:
    return {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon"],
                "crs": {
                    "$schema": "https://proj.org/schemas/v0.7/projjson.schema.json",
                    "type": "GeographicCRS",
                    "name": "WGS 84",
                    "datum": {
                        "type": "GeodeticReferenceFrame",
                        "name": "World Geodetic System 1984",
                        "ellipsoid": {
                            "name": "WGS 84",
                            "semi_major_axis": 6378137,
                            "inverse_flattening": 298.257223563,
                        },
                    },
                    "coordinate_system": {
                        "subtype": "ellipsoidal",
                        "axis": [
                            {
                                "name": "Geodetic latitude",
                                "abbreviation": "Lat",
                                "direction": "north",
                                "unit": "degree",
                            },
                            {
                                "name": "Geodetic longitude",
                                "abbreviation": "Lon",
                                "direction": "east",
                                "unit": "degree",
                            },
                        ],
                    },
                    "id": {"authority": "EPSG", "code": 4326},
                },
                "bbox": [-180, -90, 180, 90],
                "covering": {
                    "bbox": {
                        "xmin": ["bbox", "xmin"],
                        "ymin": ["bbox", "ymin"],
                        "xmax": ["bbox", "xmax"],
                        "ymax": ["bbox", "ymax"],
                    }
                },
            }
        },
    }


def _stac_gp_metadata() -> dict:
    return {
        "version": "1.0.0",
        "collections": {
            COLLECTION_ID: {
                "type": "Collection",
                "id": COLLECTION_ID,
                "title": "Walkthru Weather Index",
                "description": (
                    "NOAA AI-NWP (GraphCast_GFS) weather forecasts downscaled to "
                    "H3 hexagonal grid, served as Hive-partitioned Parquet on S3."
                ),
                "license": "CC-BY-4.0",
                "extent": {
                    "spatial": {"bbox": [[-180, -90, 180, 90]]},
                    "temporal": {"interval": [["2026-03-01T00:00:00Z", None]]},
                },
                "links": [],
            }
        },
    }


def _parse_noaa_key(key: str) -> tuple[str, int] | None:
    m = re.search(r"(\d{10})_f\d+_f\d+_\d+\.nc$", key)
    if not m:
        return None
    dt = datetime.strptime(m.group(1), "%Y%m%d%H")
    return dt.strftime("%Y-%m-%d"), dt.hour


def discover_from_noaa(days: int) -> list[tuple[str, int]]:
    """List available NOAA forecast runs for the last N days."""
    prefix = NOAA_PREFIX_MAP[MODEL]
    store = S3Store(NOAA_BUCKET, region="us-east-1", skip_signature=True)
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)

    result = []
    d = cutoff.date()
    today = datetime.now(tz=timezone.utc).date()
    while d <= today:
        day_prefix = f"{prefix}/{d.year}/{d.strftime('%m%d')}/"
        for chunk in store.list(prefix=day_prefix):
            for meta in chunk:
                if not meta["path"].endswith(".nc"):
                    continue
                parsed = _parse_noaa_key(meta["path"])
                if parsed:
                    result.append(parsed)
        d += timedelta(days=1)

    result.sort()
    return result


def check_output_exists(
    con: duckdb.DuckDBPyConnection, date_str: str, hour: int
) -> int | None:
    """Check if h3_res=5 output exists, return compressed size or None."""
    s3_path = (
        f"{_s3_base()}/model={MODEL}/date={date_str}/hour={hour}/h3_res=5/data.parquet"
    )
    try:
        row = con.sql(
            f"SELECT sum(row_group_compressed_bytes)::BIGINT AS size "
            f"FROM parquet_metadata('{s3_path}')"
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def build_catalog(
    partitions: list[dict], output_path: str, con: duckdb.DuckDBPyConnection
) -> None:
    """Build STAC GeoParquet catalog using pure DuckDB + KV_METADATA."""
    if not partitions:
        log.warning("No partitions found — nothing to catalog")
        return

    con.sql("INSTALL spatial; LOAD spatial")
    con.sql("SET geometry_always_xy = true")

    # Build a VALUES clause for all items
    values_rows = []
    for p in partitions:
        item_id = f"{MODEL}-{p['date']}-{p['hour']:02d}z"
        dt_iso = f"{p['date']}T{p['hour']:02d}:00:00+00:00"
        base = _s3_base()
        https = _https_base()

        # Asset hrefs for each resolution
        assets = {}
        for res in range(1, 6):
            s3_href = f"{base}/model={MODEL}/date={p['date']}/hour={p['hour']}/h3_res={res}/data.parquet"
            assets[f"h3_res{res}"] = {
                "href": s3_href,
                "type": "application/vnd.apache.parquet",
                "title": f"H3 resolution {res}",
            }
            if https:
                https_href = f"{https}/model={MODEL}/date={p['date']}/hour={p['hour']}/h3_res={res}/data.parquet"
                assets[f"h3_res{res}_https"] = {
                    "href": https_href,
                    "type": "application/vnd.apache.parquet",
                    "title": f"H3 resolution {res} (HTTPS)",
                }

        esc_id = item_id.replace("'", "''")
        esc_dt = dt_iso.replace("'", "''")
        esc_assets = json.dumps(assets).replace("'", "''")
        values_rows.append(
            f"('{esc_id}', '{esc_dt}'::TIMESTAMPTZ, '{esc_assets}', {p['size_bytes']})"
        )

    values_sql = ",\n        ".join(values_rows)

    geo_escaped = json.dumps(_geo_metadata()).replace("'", "''")
    stac_escaped = json.dumps(_stac_gp_metadata()).replace("'", "''")
    forecast_ext = FORECAST_EXT
    collection = COLLECTION_ID

    sql = f"""
        COPY (
            WITH items(id, datetime, assets_json, size_bytes) AS (
                VALUES {values_sql}
            )
            SELECT
                'Feature' AS type,
                '1.1.0' AS stac_version,
                ['{forecast_ext}']::VARCHAR[] AS stac_extensions,
                id,
                '{collection}' AS collection,

                -- geometry: global bbox as WKB polygon
                ST_GeomFromText(
                    'POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))'
                ) AS geometry,

                -- bbox struct
                {{xmin: -180.0, ymin: -90.0, xmax: 180.0, ymax: 90.0}} AS bbox,

                -- timestamps
                datetime,
                datetime AS "forecast:reference_datetime",
                'PT0S' AS "forecast:duration",

                -- extra properties
                '{MODEL}' AS model,
                size_bytes,

                -- links
                []::STRUCT(href VARCHAR, rel VARCHAR, type VARCHAR, title VARCHAR)[] AS links,

                -- assets as JSON string
                assets_json AS assets

            FROM items
            ORDER BY datetime
        ) TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD,
         ROW_GROUP_SIZE 10000,
         KV_METADATA {{
            geo: '{geo_escaped}',
            'stac-geoparquet': '{stac_escaped}'
         }})
    """

    con.sql(sql)
    log.info("Catalog written: %s (%d items)", output_path, len(partitions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build STAC GeoParquet catalog")
    parser.add_argument(
        "--days", type=int, default=90, help="Look back N days (default: 90)"
    )
    parser.add_argument("--local", default="", help="Write to local path instead of S3")
    args = parser.parse_args()

    # Set up DuckDB with S3 credentials
    con = duckdb.connect()
    con.install_extension("httpfs")
    con.load_extension("httpfs")

    aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    if aws_key and aws_secret:
        con.sql(f"SET s3_region='{aws_region}'")
        con.sql(f"SET s3_access_key_id='{aws_key}'")
        con.sql(f"SET s3_secret_access_key='{aws_secret}'")
        con.sql("SET s3_url_style='path'")

    # Step 1: Discover available NOAA runs
    log.info("Discovering NOAA runs for last %d days...", args.days)
    noaa_runs = discover_from_noaa(args.days)
    log.info("NOAA runs found: %d", len(noaa_runs))

    # Step 2: Check which have been processed
    log.info("Checking output S3 for processed runs...")
    partitions = []
    for date_str, hour in noaa_runs:
        size = check_output_exists(con, date_str, hour)
        if size is not None:
            partitions.append({"date": date_str, "hour": hour, "size_bytes": size})
            log.info("  OK: date=%s hour=%d (%s bytes)", date_str, hour, f"{size:,}")
        else:
            log.info("  MISSING: date=%s hour=%d", date_str, hour)

    log.info("Processed runs: %d / %d", len(partitions), len(noaa_runs))

    if not partitions:
        log.warning("No processed runs found")
        sys.exit(1)

    # Step 3: Build catalog
    output_path = args.local if args.local else f"{_s3_base()}/catalog.parquet"
    build_catalog(partitions, output_path, con)


if __name__ == "__main__":
    main()
