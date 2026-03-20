"""Write interpolated results as Hive-partitioned Parquet to S3.

Partition scheme:
  s3://{bucket}/{prefix}/weather/
    model={model}/
      date={YYYY-MM-DD}/
        hour={HH}/
          h3_res={res}/
            data.parquet          (single file, sorted by h3_index)

OPTIMIZATION STRATEGY (DuckDB 1.5):
  1. Finest resolution (res 8): Python interpolation → DuckDB direct write
     (register PyArrow table → round + sort → COPY TO parquet in one step,
     no intermediate part files)
  2. Coarser resolutions (7→1): DuckDB h3 aggregate-of-aggregates rollup
     using h3_cell_to_parent + AVG per (h3_parent, timestamp) group
  3. Vector quantities (wind, moisture flux): avg u/v components, then
     recompute speed/direction/magnitude for physical correctness
  4. preserve_insertion_order=false for memory-efficient GROUP BY
  5. temp_directory set for out-of-core spilling on large datasets
  6. Weather values rounded to meteorologically appropriate precision
     (~63% better ZSTD compression, no scientific information loss)
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
    "temperature_2m_C": 1,
    "temperature_850hPa_C": 1,
    "temp_diff_850hPa_2m_C": 1,
    "wind_speed_10m_ms": 1,
    "wind_direction_10m_deg": 0,
    "wind_speed_850hPa_ms": 1,
    "wind_direction_850hPa_deg": 0,
    "wind_shear_magnitude_ms": 1,
    "wind_shear_direction_deg": 0,
    "wind_u_10m_ms": 1,
    "wind_v_10m_ms": 1,
    "wind_u_850hPa_ms": 1,
    "wind_v_850hPa_ms": 1,
    "specific_humidity_gkg": 3,
    "moisture_flux_magnitude": 4,
    "moisture_flux_u": 4,
    "moisture_flux_v": 4,
    "pressure_msl_hPa": 1,
    "precipitation_mm_6hr": 2,
    "vertical_velocity_500hPa_Pas": 3,
    "geopotential_500hPa_m": 1,
    "geopotential_anomaly_500hPa_m": 1,
}

# ── H3 aggregation column specs ─────────────────────────────────────────────
# Weather uses AVG (intensive properties) unlike places which uses SUM (counts).
# Vector components are averaged; derived quantities (speed, direction) are
# recomputed from the averaged components for physical correctness.

# Scalar columns: simple AVG
_AGG_SCALAR: dict[str, int] = {
    "temperature_2m_C": 1,
    "temperature_850hPa_C": 1,
    "specific_humidity_gkg": 3,
    "pressure_msl_hPa": 1,
    "precipitation_mm_6hr": 2,
    "vertical_velocity_500hPa_Pas": 3,
    "geopotential_500hPa_m": 1,
    "geopotential_anomaly_500hPa_m": 1,
}

# Vector component columns: AVG then recompute derived speed/direction
_AGG_VECTORS: dict[str, int] = {
    "wind_u_10m_ms": 1,
    "wind_v_10m_ms": 1,
    "wind_u_850hPa_ms": 1,
    "wind_v_850hPa_ms": 1,
    "moisture_flux_u": 4,
    "moisture_flux_v": 4,
}


# ── DuckDB connection ───────────────────────────────────────────────────────


def _get_duckdb(use_s3: bool) -> duckdb.DuckDBPyConnection:
    """Lazy-init a DuckDB 1.5 connection with performance settings."""
    global _duckdb_con  # noqa: PLW0603
    if _duckdb_con is not None:
        return _duckdb_con

    log.info("[EXPORT] Initializing DuckDB %s", duckdb.__version__)
    con = duckdb.connect()

    # DuckDB 1.5 performance settings
    con.sql("SET preserve_insertion_order = false")  # memory-efficient GROUP BY
    con.sql("SET temp_directory = 'duckdb_temp.tmp'")  # out-of-core spilling

    for ext in ("httpfs",):
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


# ── Filesystem setup ────────────────────────────────────────────────────────


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

    if use_s3:
        parts = [s3_bucket]
        if s3_prefix:
            parts.append(s3_prefix.strip("/"))
        parts.append("weather")
        base_dir = "/".join(parts)
    else:
        base_dir = "output/weather"
        Path(base_dir).mkdir(parents=True, exist_ok=True)

    log.info(
        "[EXPORT] Filesystem ready: %s",
        f"s3://{base_dir}" if use_s3 else base_dir,
    )
    return filesystem, base_dir, use_s3


# ── Finest resolution: DuckDB direct write ──────────────────────────────────


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
    """Write a single resolution via DuckDB direct write (no part files).

    DuckDB 1.5 registers the PyArrow table as a zero-copy view, then writes
    a single sorted + rounded Parquet file in one COPY TO query.  This
    replaces the legacy two-step pattern (PyArrow part files → DuckDB merge
    → cleanup) with a single-pass approach.
    """
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

    # Build PyArrow arrays (h3_index + timestamp + weather only, no partition cols)
    arrays: dict[str, pa.Array] = {}
    arrays["h3_index"] = pa.array(np.tile(h3_df["h3_index"].values, T), type=pa.int64())
    arrays["timestamp"] = pa.array(
        np.repeat(timestamps.values, N), type=pa.timestamp("s", tz="UTC")
    )

    for src_name, out_name in _COLUMN_MAP.items():
        arr = interpolated.get(src_name)
        if arr is None:
            continue
        arrays[out_name] = pa.array(
            arr.flatten(order="C").astype(np.float32), type=pa.float32()
        )

    table = pa.table(arrays)
    del arrays

    # DuckDB direct write: register → round + sort → COPY TO parquet
    con = _get_duckdb(use_s3)
    con.register("_raw_weather", table)

    partition_dir = (
        f"{base_dir}/model={model_name}/date={date_str}/hour={hour_int}/h3_res={res}"
    )
    s3_pfx = "s3://" if use_s3 else ""
    final_path = f"{s3_pfx}{partition_dir}/data.parquet"

    if not use_s3:
        Path(partition_dir).mkdir(parents=True, exist_ok=True)

    # Dynamic SELECT with rounding (handles models with missing columns)
    select_parts = ["h3_index", "timestamp"]
    for col_name in table.column_names:
        if col_name in ("h3_index", "timestamp"):
            continue
        if col_name in _ROUND_DECIMALS:
            d = _ROUND_DECIMALS[col_name]
            select_parts.append(f"round({col_name}, {d})::FLOAT AS {col_name}")
        else:
            select_parts.append(col_name)
    select_sql = ", ".join(select_parts)

    t0 = time.time()
    con.sql(f"""
        COPY (
            SELECT {select_sql}
            FROM _raw_weather
            ORDER BY h3_index
        ) TO '{final_path}'
        (FORMAT PARQUET, PARQUET_VERSION v2, COMPRESSION ZSTD,
         COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 1000000)
    """)
    con.unregister("_raw_weather")

    elapsed = time.time() - t0
    uri = f"s3://{base_dir}" if use_s3 else base_dir
    log.info(
        "[EXPORT] res %d (%s rows) → data.parquet (%.1fs) %s",
        res,
        f"{total:,}",
        elapsed,
        uri,
    )
    return uri


# ── H3 rollup: aggregate-of-aggregates ──────────────────────────────────────


def _build_weather_agg_sql(
    source_table: str,
    target_table: str,
    target_res: int,
    available_cols: set[str],
) -> str:
    """Build CREATE TABLE SQL for weather H3 rollup to coarser resolution.

    Vector quantities (wind, moisture flux) are handled correctly:
    u/v components are averaged, then speed/direction/magnitude are
    recomputed from the averaged components (not averaged directly).

    For the aggregate-of-aggregates cascade (res 8→7→6→...→1), each level
    averages its parent's already-averaged values.  This is exact for H3
    hexagon parents (7 children each) and < 0.5% off for the 12 pentagon
    cells — well within the rounding precision of the output.
    """
    parts = [
        f"h3_cell_to_parent(h3_index::UBIGINT, {target_res})::BIGINT AS h3_index",
        "timestamp",
    ]

    # Scalar AVGs
    for col, d in _AGG_SCALAR.items():
        if col in available_cols:
            parts.append(f"round(avg({col}), {d})::FLOAT AS {col}")

    # Vector component AVGs
    for col, d in _AGG_VECTORS.items():
        if col in available_cols:
            parts.append(f"round(avg({col}), {d})::FLOAT AS {col}")

    # Derived: temp diff (from averaged base temps)
    if {"temperature_850hPa_C", "temperature_2m_C"} <= available_cols:
        parts.append(
            "round(avg(temperature_850hPa_C) - avg(temperature_2m_C),"
            " 1)::FLOAT AS temp_diff_850hPa_2m_C"
        )

    # Derived: wind 10m speed + direction (from averaged u, v)
    if {"wind_u_10m_ms", "wind_v_10m_ms"} <= available_cols:
        parts.append(
            "round(sqrt(avg(wind_u_10m_ms) * avg(wind_u_10m_ms)"
            " + avg(wind_v_10m_ms) * avg(wind_v_10m_ms)), 1)::FLOAT"
            " AS wind_speed_10m_ms"
        )
        parts.append(
            "round(((degrees(atan2(-avg(wind_u_10m_ms),"
            " -avg(wind_v_10m_ms))) + 360) % 360), 0)::FLOAT"
            " AS wind_direction_10m_deg"
        )

    # Derived: wind 850hPa speed + direction
    if {"wind_u_850hPa_ms", "wind_v_850hPa_ms"} <= available_cols:
        parts.append(
            "round(sqrt(avg(wind_u_850hPa_ms) * avg(wind_u_850hPa_ms)"
            " + avg(wind_v_850hPa_ms) * avg(wind_v_850hPa_ms)), 1)::FLOAT"
            " AS wind_speed_850hPa_ms"
        )
        parts.append(
            "round(((degrees(atan2(-avg(wind_u_850hPa_ms),"
            " -avg(wind_v_850hPa_ms))) + 360) % 360), 0)::FLOAT"
            " AS wind_direction_850hPa_deg"
        )

    # Derived: wind shear (850hPa minus 10m)
    shear_deps = {
        "wind_u_850hPa_ms",
        "wind_v_850hPa_ms",
        "wind_u_10m_ms",
        "wind_v_10m_ms",
    }
    if shear_deps <= available_cols:
        parts.append(
            "round(sqrt("
            "(avg(wind_u_850hPa_ms) - avg(wind_u_10m_ms))"
            " * (avg(wind_u_850hPa_ms) - avg(wind_u_10m_ms))"
            " + (avg(wind_v_850hPa_ms) - avg(wind_v_10m_ms))"
            " * (avg(wind_v_850hPa_ms) - avg(wind_v_10m_ms))"
            "), 1)::FLOAT AS wind_shear_magnitude_ms"
        )
        parts.append(
            "round(((degrees(atan2("
            "-(avg(wind_u_850hPa_ms) - avg(wind_u_10m_ms)),"
            " -(avg(wind_v_850hPa_ms) - avg(wind_v_10m_ms))"
            ")) + 360) % 360), 0)::FLOAT AS wind_shear_direction_deg"
        )

    # Derived: moisture flux magnitude
    if {"moisture_flux_u", "moisture_flux_v"} <= available_cols:
        parts.append(
            "round(sqrt(avg(moisture_flux_u) * avg(moisture_flux_u)"
            " + avg(moisture_flux_v) * avg(moisture_flux_v)), 4)::FLOAT"
            " AS moisture_flux_magnitude"
        )

    select_clause = ",\n        ".join(parts)

    return f"""CREATE OR REPLACE TABLE {target_table} AS
SELECT
    {select_clause}
FROM {source_table}
GROUP BY 1, 2"""


def aggregate_resolutions(
    base_dir: str,
    model_name: str,
    date_str: str,
    hour_int: int,
    finest_res: int,
    coarsest_res: int,
    use_s3: bool,
) -> None:
    """Roll up weather from finest H3 resolution to all coarser resolutions.

    Uses DuckDB h3 extension with aggregate-of-aggregates pattern:
    compute finest resolution once via Python interpolation, then derive
    all coarser resolutions by grouping with h3_cell_to_parent.

    OPTIMIZATION (DuckDB 1.5):
      - preserve_insertion_order=false reduces GROUP BY memory
      - temp_directory enables out-of-core spilling for large datasets
      - Each level: CREATE TABLE (no ORDER BY) → COPY TO (ORDER BY for
        spatial locality + compression) → DROP (free memory)
      - AVG/sqrt/atan2 all support disk spilling in DuckDB 1.5
      - Vector components averaged; derived quantities recomputed
    """
    con = _get_duckdb(use_s3)

    # Load h3 extension
    try:
        con.load_extension("h3")
    except Exception:
        con.install_extension("h3", repository="community")
        con.load_extension("h3")
    log.info("[AGG] DuckDB h3 extension loaded")

    partition_base = f"{base_dir}/model={model_name}/date={date_str}/hour={hour_int}"
    s3_pfx = "s3://" if use_s3 else ""

    # Read finest resolution parquet
    src_path = f"{s3_pfx}{partition_base}/h3_res={finest_res}/data.parquet"
    log.info("[AGG] Reading res %d: %s", finest_res, src_path)

    # Detect available weather columns from parquet schema
    schema_rows = con.sql(
        f"DESCRIBE SELECT * FROM read_parquet('{src_path}')"
    ).fetchall()
    available_cols = {row[0] for row in schema_rows}
    log.info("[AGG] Available columns: %d", len(available_cols))

    # Load finest resolution into DuckDB table
    con.sql(f"""
        CREATE OR REPLACE TABLE h3_res{finest_res} AS
        SELECT * FROM read_parquet('{src_path}')
    """)
    row_count = con.sql(f"SELECT count(*) FROM h3_res{finest_res}").fetchone()[0]
    log.info("[AGG] Loaded res %d: %s rows", finest_res, f"{row_count:,}")

    # Roll up: 8 → 7 → 6 → ... → coarsest_res
    source_res = finest_res
    for target_res in range(finest_res - 1, coarsest_res - 1, -1):
        t0 = time.time()

        # CREATE TABLE with GROUP BY (no ORDER BY — avoid blocking sort)
        agg_sql = _build_weather_agg_sql(
            source_table=f"h3_res{source_res}",
            target_table=f"h3_res{target_res}",
            target_res=target_res,
            available_cols=available_cols,
        )
        con.sql(agg_sql)

        target_rows = con.sql(f"SELECT count(*) FROM h3_res{target_res}").fetchone()[0]

        # COPY TO parquet (ORDER BY here for spatial locality → better ZSTD)
        out_path = f"{s3_pfx}{partition_base}/h3_res={target_res}/data.parquet"
        if not use_s3:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        con.sql(f"""
            COPY (
                SELECT * FROM h3_res{target_res}
                ORDER BY h3_index, timestamp
            ) TO '{out_path}'
            (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3,
             PARQUET_VERSION v2, ROW_GROUP_SIZE 1000000)
        """)

        elapsed = time.time() - t0
        log.info(
            "[AGG] res %d → %d: %s rows (%.1fs)",
            source_res,
            target_res,
            f"{target_rows:,}",
            elapsed,
        )

        # Free memory: drop source table before next iteration
        con.sql(f"DROP TABLE h3_res{source_res}")
        source_res = target_res

    # Drop last table
    con.sql(f"DROP TABLE h3_res{source_res}")
    log.info("[AGG] All resolutions %d → %d complete", finest_res, coarsest_res)
