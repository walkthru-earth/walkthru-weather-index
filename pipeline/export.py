"""Write interpolated results as Hive-partitioned Parquet to S3.

Partition scheme:
  s3://{bucket}/{prefix}/weather/
    model={model}/
      date={YYYY-MM-DD}/
        hour={HH}/
          h3_res={res}/
            data.parquet          (single file, sorted by h3_index)

DuckDB 1.5 writes a single sorted file per partition with native Parquet 2.11+
GEOMETRY and per-row-group bounding box stats.  This is optimal for DuckDB-WASM
consumers: one metadata fetch instead of 85 part-file round-trips.

Weather values are rounded to meteorologically appropriate precision before
writing, which dramatically improves ZSTD compression (~63% smaller) while
losing no scientifically meaningful information.  The source data (0.25° NOAA
AI-NWP) already limits effective precision.

The S3_PREFIX env var (or --s3-prefix CLI arg) controls the key prefix inside the
bucket.  Set it when IAM credentials are scoped to a specific prefix.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

log = logging.getLogger(__name__)

# Module-level lazy DuckDB connection (one per process).
_duckdb_con: duckdb.DuckDBPyConnection | None = None


_COLUMN_MAP = {
    "temperature_2m": "temperature_2m_C",
    "temperature_850hPa": "temperature_850hPa_C",
    "temp_diff_850hPa_2m": "temp_diff_850hPa_2m_C",
    "wind_speed": "wind_speed_10m_ms",
    "wind_direction": "wind_direction_10m_deg",
    "wind_speed_850hPa": "wind_speed_850hPa_ms",
    "wind_direction_850hPa": "wind_direction_850hPa_deg",
    "wind_shear_magnitude": "wind_shear_magnitude_ms",
    "wind_shear_direction": "wind_shear_direction_deg",
    "wind_u_10m": "wind_u_10m_ms",
    "wind_v_10m": "wind_v_10m_ms",
    "wind_u_850hPa": "wind_u_850hPa_ms",
    "wind_v_850hPa": "wind_v_850hPa_ms",
    "specific_humidity_surface": "specific_humidity_gkg",
    "moisture_flux_magnitude": "moisture_flux_magnitude",
    "moisture_flux_u": "moisture_flux_u",
    "moisture_flux_v": "moisture_flux_v",
    "pressure_msl": "pressure_msl_hPa",
    "precipitation": "precipitation_mm_6hr",
    "vertical_velocity_500hPa": "vertical_velocity_500hPa_Pas",
    "geopotential_500hPa": "geopotential_500hPa_m",
    "geopotential_anomaly_500hPa": "geopotential_anomaly_500hPa_m",
}

# Rounding precision per output column.  Source data is 0.25° (~28 km) NOAA
# AI-NWP — sub-decimal precision is interpolation noise, not real signal.
# Rounding lets ZSTD compress ~63% better without any scientific information loss.
_ROUND_DECIMALS: dict[str, int] = {
    "temperature_2m_C": 1,  # 0.1 °C
    "temperature_850hPa_C": 1,  # 0.1 °C
    "temp_diff_850hPa_2m_C": 1,  # 0.1 °C
    "wind_speed_10m_ms": 1,  # 0.1 m/s
    "wind_direction_10m_deg": 0,  # 1°
    "wind_speed_850hPa_ms": 1,  # 0.1 m/s
    "wind_direction_850hPa_deg": 0,  # 1°
    "wind_shear_magnitude_ms": 1,  # 0.1 m/s
    "wind_shear_direction_deg": 0,  # 1°
    "wind_u_10m_ms": 1,  # 0.1 m/s
    "wind_v_10m_ms": 1,  # 0.1 m/s
    "wind_u_850hPa_ms": 1,  # 0.1 m/s
    "wind_v_850hPa_ms": 1,  # 0.1 m/s
    "specific_humidity_gkg": 3,  # 0.001 g/kg
    "moisture_flux_magnitude": 4,  # 0.0001
    "moisture_flux_u": 4,  # 0.0001
    "moisture_flux_v": 4,  # 0.0001
    "pressure_msl_hPa": 1,  # 0.1 hPa
    "precipitation_mm_6hr": 2,  # 0.01 mm
    "vertical_velocity_500hPa_Pas": 3,  # 0.001 Pa/s
    "geopotential_500hPa_m": 1,  # 0.1 m
    "geopotential_anomaly_500hPa_m": 1,  # 0.1 m
}


def _get_duckdb(use_s3: bool) -> duckdb.DuckDBPyConnection:
    """Lazy-init a DuckDB connection with spatial + httpfs extensions."""
    global _duckdb_con  # noqa: PLW0603
    if _duckdb_con is not None:
        return _duckdb_con

    log.info("[EXPORT] Initializing DuckDB %s", duckdb.__version__)
    con = duckdb.connect()

    for ext in ("spatial", "httpfs"):
        try:
            con.load_extension(ext)
        except Exception:
            con.install_extension(ext)
            con.load_extension(ext)
        log.info("[EXPORT]   Extension '%s' loaded", ext)

    if use_s3:
        aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        if aws_key and aws_secret:
            con.sql(f"SET s3_region='{aws_region}'")
            con.sql(f"SET s3_access_key_id='{aws_key}'")
            con.sql(f"SET s3_secret_access_key='{aws_secret}'")
            con.sql("SET s3_url_style='path'")
            log.info("[EXPORT]   S3 configured: region=%s, url_style=path", aws_region)

    _duckdb_con = con
    return con


def _merge_to_single_file(
    base_dir: str,
    model_name: str,
    date_str: str,
    hour_int: int,
    res: int,
    filesystem: object,
    use_s3: bool,
) -> None:
    """Merge part files into one sorted file with GEOMETRY and rounded values.

    DuckDB reads all part files, rounds weather values to meteorologically
    appropriate precision (improves ZSTD compression ~63%), adds
    ST_Point(lon, lat)::GEOMETRY('EPSG:4326'), sorts by h3_index for spatial
    locality, and writes a single ``data.parquet``.

    Single-file output is optimal for DuckDB-WASM: one HTTP range request for
    Parquet footer metadata instead of 85 separate fetches.
    """
    partition_dir = (
        f"{base_dir}/model={model_name}/date={date_str}/hour={hour_int}/h3_res={res}"
    )

    from pyarrow.fs import FileSelector

    try:
        file_infos = filesystem.get_file_info(FileSelector(partition_dir))
    except Exception as e:
        log.warning("[EXPORT] Could not list partition dir %s: %s", partition_dir, e)
        return

    parquet_files = [
        fi.path for fi in file_infos if fi.path.endswith(".parquet") and fi.size > 0
    ]
    if not parquet_files:
        log.warning("[EXPORT] No parquet files found in %s", partition_dir)
        return

    con = _get_duckdb(use_s3)
    t0 = time.time()

    # Build the glob pattern for all part files
    if use_s3:
        glob_pattern = f"s3://{partition_dir}/part-*.parquet"
        final_path = f"s3://{partition_dir}/data.parquet"
    else:
        glob_pattern = f"{partition_dir}/part-*.parquet"
        final_path = f"{partition_dir}/data.parquet"

    # Build SELECT with rounding for weather columns
    round_exprs = []
    for col_name, decimals in _ROUND_DECIMALS.items():
        round_exprs.append(f"round({col_name}, {decimals})::FLOAT AS {col_name}")
    round_sql = ",\n                   ".join(round_exprs)

    # Build list of non-weather columns to pass through
    pass_through = [
        "h3_index",
        "round(lat, 2)::FLOAT AS lat",
        "round(lon, 2)::FLOAT AS lon",
        "area_km2",
        "timestamp",
    ]

    select_sql = (
        ",\n                   ".join(pass_through)
        + ",\n                   "
        + round_sql
    )

    con.sql(f"""
        COPY (
            SELECT {select_sql},
                   ST_Point(round(lon, 2), round(lat, 2))::GEOMETRY('EPSG:4326') AS geometry
            FROM read_parquet('{glob_pattern}', hive_partitioning=false)
            ORDER BY h3_index
        ) TO '{final_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3,
         ROW_GROUP_SIZE 1000000, GEOPARQUET_VERSION 'BOTH')
    """)

    elapsed = time.time() - t0
    log.info(
        "[EXPORT]   merged + rounded → %s (%.1fs)",
        final_path.rsplit("/", 1)[-1],
        elapsed,
    )

    # Remove original part files
    for src_path in parquet_files:
        try:
            if use_s3:
                import boto3

                bucket, key = src_path.split("/", 1)
                boto3.client("s3").delete_object(Bucket=bucket, Key=key)
            else:
                Path(src_path).unlink(missing_ok=True)
        except Exception as e:
            log.warning("[EXPORT]   failed to remove %s: %s", src_path, e)

    log.info("[EXPORT]   removed %d part files", len(parquet_files))


def init_filesystem(
    s3_bucket: str,
    s3_prefix: str = "",
) -> tuple[object, str, bool]:
    """Set up S3 or local filesystem for writing. Called once per run.

    Returns (filesystem, base_dir, use_s3).
    """
    use_s3 = bool(s3_bucket and s3_bucket != "output")

    if use_s3:
        try:
            from pyarrow.fs import S3FileSystem

            filesystem = S3FileSystem(
                region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
        except Exception as e:
            log.warning("S3FileSystem unavailable (%s), writing locally instead.", e)
            use_s3 = False
            filesystem = None

    if not use_s3:
        from pyarrow.fs import LocalFileSystem

        filesystem = LocalFileSystem()

    # Build base_dir: bucket[/prefix]/weather
    # PyArrow S3 paths use the form "bucket/key" (no s3:// scheme).
    if use_s3:
        parts = [s3_bucket]
        if s3_prefix:
            parts.append(s3_prefix.strip("/"))
        parts.append("weather")
        base_dir = "/".join(parts)
    else:
        import pathlib

        base_dir = "output/weather"
        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

    log.info(
        "[EXPORT] Filesystem ready: %s", f"s3://{base_dir}" if use_s3 else base_dir
    )
    return filesystem, base_dir, use_s3


def write_resolution_to_s3(
    res: int,
    h3_df: pd.DataFrame,
    interpolated: dict[str, np.ndarray],
    model_name: str,
    filesystem: object,
    base_dir: str,
    use_s3: bool,
    run_time: datetime | None = None,
) -> str | None:
    """Write a single resolution immediately. Returns the URI written to, or None."""
    if not interpolated:
        log.warning("[EXPORT] No data to write for res %d", res)
        return None

    if run_time is None:
        run_time = datetime.now(tz=timezone.utc)

    date_str = run_time.strftime("%Y-%m-%d")
    hour_int = int(run_time.strftime("%H"))

    T = next(iter(interpolated.values())).shape[0]
    timestamps = pd.date_range(
        start=run_time.replace(minute=0, second=0, microsecond=0),
        periods=T,
        freq="6h",
    )
    N = len(h3_df)
    total = T * N

    log.info(
        "[EXPORT] Building table: %d ts x %s cells = %s rows",
        T,
        f"{N:,}",
        f"{total:,}",
    )

    # Build PyArrow arrays directly from numpy to avoid Python list overhead.
    # Python float objects are 28 bytes each vs 4 bytes in numpy float32.
    # For 42M rows x 27 float columns, lists would use ~27 GB; numpy uses ~4.5 GB.
    arrays: dict[str, pa.Array] = {}

    # Grid metadata: tile N values T times (numpy arrays)
    arrays["h3_index"] = pa.array(np.tile(h3_df["h3_index"].values, T))
    arrays["lat"] = pa.array(
        np.tile(h3_df["lat"].values.astype(np.float32), T), type=pa.float32()
    )
    arrays["lon"] = pa.array(
        np.tile(h3_df["lon"].values.astype(np.float32), T), type=pa.float32()
    )
    arrays["area_km2"] = pa.array(
        np.tile(h3_df["area_km2"].values.astype(np.float32), T), type=pa.float32()
    )

    # Timestamps: repeat each timestamp N times
    arrays["timestamp"] = pa.array(
        np.repeat(timestamps.values, N), type=pa.timestamp("s", tz="UTC")
    )

    # Partition columns
    arrays["model"] = pa.array([model_name] * total, type=pa.string())
    arrays["date"] = pa.array([date_str] * total, type=pa.string())
    arrays["hour"] = pa.array(np.full(total, hour_int, dtype=np.uint8), type=pa.uint8())
    arrays["h3_res"] = pa.array(np.full(total, res, dtype=np.uint8), type=pa.uint8())

    # Weather variables: flatten numpy (T, N) -> (T*N,) with zero-copy to PyArrow
    for src_name, out_name in _COLUMN_MAP.items():
        arr = interpolated.get(src_name)
        if arr is None:
            continue
        arrays[out_name] = pa.array(
            arr.flatten(order="C").astype(np.float32), type=pa.float32()
        )

    table = pa.table(arrays)
    del arrays  # free the dict reference

    # Write raw part files via PyArrow (fast multipart upload to S3)
    partition_dir = (
        f"{base_dir}/model={model_name}/date={date_str}/hour={hour_int}/h3_res={res}"
    )
    if not use_s3:
        Path(partition_dir).mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq

    # Write as temporary part files (will be merged by DuckDB)
    rows_per_part = 500_000
    for i in range(0, len(table), rows_per_part):
        part = table.slice(i, min(rows_per_part, len(table) - i))
        part_path = f"{partition_dir}/part-{i // rows_per_part}.parquet"
        if use_s3:
            pq.write_table(
                part,
                part_path,
                filesystem=filesystem,
                compression="zstd",
                compression_level=1,
            )
        else:
            pq.write_table(part, part_path, compression="zstd", compression_level=1)

    # DuckDB post-processing: merge all parts into a single sorted file with
    # rounded values, native GEOMETRY, and per-row-group bounding box stats.
    _merge_to_single_file(
        base_dir, model_name, date_str, hour_int, res, filesystem, use_s3
    )

    uri = f"s3://{base_dir}" if use_s3 else base_dir
    log.info("[EXPORT] res %d (%s rows) written to %s", res, f"{len(table):,}", uri)
    return uri
