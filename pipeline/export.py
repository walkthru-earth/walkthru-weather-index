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

No local temp files are written — PyArrow streams directly via multipart upload.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


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


def write_to_s3(
    h3_grids: dict[int, pd.DataFrame],
    interpolated: dict[int, dict[str, np.ndarray]],
    model_name: str,
    s3_bucket: str,
    s3_prefix: str = "",
    run_time: datetime | None = None,
) -> str:
    """Write all resolutions to S3.  Returns the S3 base URI written to.

    Parameters
    ----------
    s3_bucket : str
        Bare bucket name (no ``s3://`` scheme, no trailing slash).
    s3_prefix : str
        Key prefix inside the bucket (e.g. ``"indices/v1"``).  Useful when
        IAM credentials are scoped to a specific prefix.
    """
    if run_time is None:
        run_time = datetime.now(tz=timezone.utc)

    date_str = run_time.strftime("%Y-%m-%d")
    hour_int = int(run_time.strftime("%H"))

    use_s3 = bool(s3_bucket and s3_bucket != "output")

    if use_s3:
        try:
            from pyarrow.fs import S3FileSystem

            filesystem = S3FileSystem(
                region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            )
        except Exception as e:
            print(f"   ⚠️  S3FileSystem unavailable ({e}), writing locally instead.")
            use_s3 = False
            filesystem = None
    else:
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
        base_dir = "output/weather"

    if not use_s3:
        import pathlib

        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

    all_tables = []
    for res, h3_df in h3_grids.items():
        vars_for_res = interpolated.get(res, {})
        if not vars_for_res:
            continue

        T = next(iter(vars_for_res.values())).shape[0]
        timestamps = pd.date_range(
            start=run_time.replace(minute=0, second=0, microsecond=0),
            periods=T,
            freq="6h",
        )
        N = len(h3_df)

        rows: dict[str, list] = {
            "h3_index": h3_df["h3_index"].tolist() * T,
            "lat": h3_df["lat"].tolist() * T,
            "lon": h3_df["lon"].tolist() * T,
            "area_km2": h3_df["area_km2"].tolist() * T,
            "timestamp": [ts for ts in timestamps for _ in range(N)],
            "model": [model_name] * (T * N),
            "date": [date_str] * (T * N),
            "hour": [hour_int] * (T * N),
            "h3_res": [res] * (T * N),
        }

        for src_name, out_name in _COLUMN_MAP.items():
            arr = vars_for_res.get(src_name)
            if arr is None:
                continue
            # arr shape: (T, N) — flatten to (T*N,) in row-major order
            flat = arr.flatten(order="C").astype(np.float32)
            rows[out_name] = flat.tolist()

        all_tables.append(
            pa.table({k: pa.array(v, type=_pa_type(k, v)) for k, v in rows.items()})
        )

    if not all_tables:
        print("   ⚠️  No data to write.")
        return base_dir

    combined = pa.concat_tables(all_tables)

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
        combined,
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
    print(f"   ✅ Written → {uri}")
    return uri


# ── helpers ───────────────────────────────────────────────────────────────────


def _pa_type(col: str, values: list):
    if col in ("h3_index", "model", "date"):
        return pa.string()
    if col in ("h3_res", "hour"):
        return pa.uint8()
    if col == "timestamp":
        return pa.timestamp("s", tz="UTC")
    return pa.float32()
