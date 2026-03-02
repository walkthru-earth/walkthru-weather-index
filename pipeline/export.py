"""Write interpolated results as Hive-partitioned Parquet directly to S3.

Partition scheme:
  s3://{bucket}/{prefix}/weather/
    model={model}/
      date={YYYY-MM-DD}/
        hour={HH}/
          h3_res={res}/
            part-XXXXX.parquet

The S3_PREFIX env var (or --s3-prefix CLI arg) controls the key prefix inside the
bucket.  Set it when IAM credentials are scoped to a specific prefix.

No local temp files are written -- PyArrow streams directly via multipart upload.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

log = logging.getLogger(__name__)


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

    partition_schema = pa.schema(
        [
            ("model", pa.string()),
            ("date", pa.string()),
            ("hour", pa.uint8()),
            ("h3_res", pa.uint8()),
        ]
    )

    write_opts = ds.ParquetFileFormat().make_write_options(
        compression="zstd",
        compression_level=3,
        write_statistics=True,
    )

    ds.write_dataset(
        table,
        base_dir=base_dir,
        filesystem=filesystem,
        format=ds.ParquetFileFormat(),
        partitioning=ds.partitioning(partition_schema, flavor="hive"),
        file_options=write_opts,
        max_rows_per_file=500_000,
        min_rows_per_group=50_000,
        max_rows_per_group=100_000,
        existing_data_behavior="overwrite_or_ignore",
    )

    uri = f"s3://{base_dir}" if use_s3 else base_dir
    log.info("[EXPORT] res %d (%s rows) written to %s", res, f"{len(table):,}", uri)
    return uri
