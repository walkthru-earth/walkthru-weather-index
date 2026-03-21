# Global Weather Forecasts (H3-indexed)

AI-powered weather forecasts from [NOAA AI-NWP](https://registry.opendata.aws/noaa-oar-mlwp/) (GraphCast) downscaled to [H3](https://h3geo.org/) hexagonal cells with topographic corrections. Updated automatically every 12 hours, 5-day forecasts at 6-hour intervals, ~2M cells per timestep at H3 resolution 5. Coordinates are derivable from `h3_index` via the DuckDB h3 extension.

| | |
|---|---|
| **Source** | [NOAA AI-NWP](https://registry.opendata.aws/noaa-oar-mlwp/) GraphCast_GFS, 0.25° global grid, 6-hourly timesteps |
| **Terrain** | Topographic corrections via [walkthru-earth/dem-terrain](https://source.coop/walkthru-earth/dem-terrain) ([code](https://github.com/walkthru-earth/dem-terrain)) — H3-indexed [GEDTM-30m](https://doi.org/10.5281/zenodo.14900181) elevation, slope, aspect |
| **Format** | Apache Parquet, Hive-partitioned, single `data.parquet` per partition |
| **Size** | ~1 GB per forecast run (~42M rows), updated every 12 hours |
| **Update** | Automated every 12 hours via GitHub Actions + HuggingFace Jobs (A10G GPU) |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by [walkthru.earth](https://walkthru.earth/links) |
| **Code** | [walkthru-earth/walkthru-weather-index](https://github.com/walkthru-earth/walkthru-weather-index) |

## Quick Start

```sql
-- DuckDB
INSTALL h3 FROM community; LOAD h3;
INSTALL httpfs; LOAD httpfs;
SET s3_region = 'us-west-2';

SELECT h3_index,
       h3_cell_to_lat(h3_index) AS lat,
       h3_cell_to_lng(h3_index) AS lng,
       timestamp, temperature_2m_C, wind_speed_10m_ms,
       precipitation_mm_6hr, pressure_msl_hPa
FROM read_parquet(
    's3://us-west-2.opendata.source.coop/walkthru-earth/indices/weather/model=GraphCast_GFS/date=2026-03-05/hour=12/h3_res=5/data.parquet'
)
ORDER BY timestamp, temperature_2m_C DESC
LIMIT 20;
```

```python
# Python
import duckdb

con = duckdb.connect()
con.install_extension("h3", repository="community"); con.load_extension("h3")
con.install_extension("httpfs"); con.load_extension("httpfs")
con.sql("SET s3_region = 'us-west-2'")

df = con.sql("""
    SELECT h3_index,
           h3_cell_to_lat(h3_index) AS lat,
           h3_cell_to_lng(h3_index) AS lng,
           timestamp, temperature_2m_C, wind_speed_10m_ms, pressure_msl_hPa
    FROM read_parquet(
        's3://us-west-2.opendata.source.coop/walkthru-earth/indices/weather/model=GraphCast_GFS/date=2026-03-05/hour=12/h3_res=5/data.parquet'
    )
    WHERE h3_cell_to_lat(h3_index) BETWEEN 20 AND 35
      AND h3_cell_to_lng(h3_index) BETWEEN 68 AND 90
""").fetchdf()
```

## Files

```
walkthru-earth/indices/weather/
  model=GraphCast_GFS/
    date=YYYY-MM-DD/
      hour={0,12}/
        h3_res=1/data.parquet      ~525 KB    17,955 rows
        h3_res=2/data.parquet      ~3.3 MB   112,518 rows
        h3_res=3/data.parquet     ~21.2 MB   698,901 rows
        h3_res=4/data.parquet    ~141.2 MB  4,407,816 rows
        h3_res=5/data.parquet    ~931.4 MB 42,353,682 rows (2M cells × 21 timesteps)
```

Five H3 resolutions (1–5) per forecast run. Each is a **single `data.parquet` file**, sorted by `h3_index`. New forecasts are uploaded every 12 hours. Compression: ZSTD level 3, 1M-row row groups for efficient range pushdown.

## Schema

### Grid metadata

| Column | Type | Description |
|--------|------|-------------|
| `h3_index` | BIGINT | H3 cell ID (int64). Use `h3_cell_to_lat(h3_index)` / `h3_cell_to_lng(h3_index)` from the DuckDB h3 extension to derive coordinates |
| `timestamp` | TIMESTAMPTZ | Forecast valid time (UTC) |

Partition path components (`model`, `date`, `hour`, `h3_res`) are encoded in the Hive directory path only and are not stored as columns inside the Parquet file.

### Weather variables — primary (topographically corrected)

| Column | Units | Precision | Topographic correction |
|--------|-------|-----------|------------------------|
| `temperature_2m_C` | °C | 0.1 | Variable lapse rate (derived from T850 − T2m per timestep) |
| `wind_u_10m_ms` | m/s | 0.1 | Elevation + slope channelling |
| `wind_v_10m_ms` | m/s | 0.1 | Elevation + slope channelling |
| `precipitation_mm_6hr` | mm | 0.01 | Dynamic orographic enhancement (wind-direction-aware) |
| `specific_humidity_gkg` | g/kg | 0.001 | Exponential elevation adjustment (scale height 2 km) |
| `pressure_msl_hPa` | hPa | 0.1 | None (sea-level reference) |
| `temperature_850hPa_C` | °C | 0.1 | None (free-atmosphere) |
| `wind_u_850hPa_ms` | m/s | 0.1 | None (free-atmosphere) |
| `wind_v_850hPa_ms` | m/s | 0.1 | None (free-atmosphere) |
| `vertical_velocity_500hPa_Pas` | Pa/s | 0.001 | None (negative = upward motion) |
| `geopotential_500hPa_m` | m | 0.1 | None (geopotential height) |

### Weather variables — derived

| Column | Units | Precision | Formula |
|--------|-------|-----------|---------|
| `wind_speed_10m_ms` | m/s | 0.1 | sqrt(u10² + v10²) |
| `wind_direction_10m_deg` | ° (meteorological) | 1 | Direction wind blows *from*, 0° = N |
| `wind_speed_850hPa_ms` | m/s | 0.1 | sqrt(u850² + v850²) |
| `wind_direction_850hPa_deg` | ° (meteorological) | 1 | Same convention at 850 hPa |
| `wind_shear_magnitude_ms` | m/s | 0.1 | sqrt((u850−u10)² + (v850−v10)²) |
| `wind_shear_direction_deg` | ° (meteorological) | 1 | Shear vector direction |
| `temp_diff_850hPa_2m_C` | °C | 0.1 | T850 − T2m (stability indicator) |
| `moisture_flux_u` | g/kg·m/s | 0.0001 | q × u10 |
| `moisture_flux_v` | g/kg·m/s | 0.0001 | q × v10 |
| `moisture_flux_magnitude` | g/kg·m/s | 0.0001 | sqrt((qu)² + (qv)²) |
| `geopotential_anomaly_500hPa_m` | m | 0.1 | H500 − 5574 (ICAO standard reference) |

### Precision policy

All weather values are rounded to meteorologically appropriate precision before ZSTD compression. The source data is 0.25° (~28 km) NOAA AI-NWP — sub-decimal digits in the raw interpolated output are numerical noise, not real atmospheric signal. Rounding improves ZSTD compression by ~63% while preserving all scientifically meaningful information.

## How It Works

1. GitHub Actions polls NOAA S3 every 12 hours for new AI-NWP forecasts (GraphCast_GFS NetCDF)
2. When a new forecast is detected, a HuggingFace Job is triggered on an A10G GPU
3. The 0.25° global NetCDF is interpolated onto ~2M H3 cells (resolution 5) using GPU-accelerated bilinear interpolation
4. Topographic corrections are applied using the [walkthru-earth/dem-terrain](https://source.coop/walkthru-earth/dem-terrain) dataset — H3-indexed terrain derivatives (elevation, slope, aspect, TRI, TPI) computed from the [GEDTM-30m](https://doi.org/10.5281/zenodo.14900181) global DEM. The terrain data is joined by `h3_index` to apply physically-based corrections: variable lapse rates for temperature, slope channelling for wind, and orographic enhancement for precipitation
5. Weather values are rounded to meteorologically appropriate precision (e.g. temperature ±0.1 °C, wind ±0.1 m/s, direction ±1°) — this improves ZSTD compression ~63% with zero scientific information loss
6. All timesteps are merged into a single `data.parquet` per partition, sorted by `h3_index`, with 1M-row row groups. `h3_index` is written as BIGINT (int64); coordinates are derivable via the DuckDB h3 extension

## More Examples

```sql
-- Latest 5-day temperature forecast for a city (e.g. London)
SELECT h3_index,
       h3_cell_to_lat(h3_index) AS lat,
       h3_cell_to_lng(h3_index) AS lng,
       timestamp, temperature_2m_C, wind_speed_10m_ms,
       precipitation_mm_6hr, pressure_msl_hPa
FROM read_parquet(
    's3://us-west-2.opendata.source.coop/walkthru-earth/indices/weather/model=GraphCast_GFS/date=2026-03-05/hour=12/h3_res=5/data.parquet'
)
WHERE h3_cell_to_lat(h3_index) BETWEEN 51.0 AND 52.0
  AND h3_cell_to_lng(h3_index) BETWEEN -0.5 AND 0.5
ORDER BY timestamp
LIMIT 100;

-- Wind shear analysis (severe weather indicator)
SELECT h3_index,
       h3_cell_to_lat(h3_index) AS lat,
       h3_cell_to_lng(h3_index) AS lng,
       timestamp, wind_shear_magnitude_ms, temp_diff_850hPa_2m_C
FROM read_parquet(
    's3://us-west-2.opendata.source.coop/walkthru-earth/indices/weather/model=GraphCast_GFS/date=2026-03-05/hour=12/h3_res=5/data.parquet'
)
WHERE wind_shear_magnitude_ms > 20
ORDER BY wind_shear_magnitude_ms DESC
LIMIT 20;

-- DuckDB-WASM (browser) — use HTTPS URL (single file, no glob needed)
SELECT h3_index,
       h3_cell_to_lat(h3_index) AS lat,
       h3_cell_to_lng(h3_index) AS lng,
       timestamp, temperature_2m_C, wind_speed_10m_ms
FROM read_parquet(
    'https://data.source.coop/walkthru-earth/indices/weather/model=GraphCast_GFS/date=2026-03-05/hour=12/h3_res=5/data.parquet'
)
WHERE h3_cell_to_lat(h3_index) BETWEEN 35 AND 45
  AND h3_cell_to_lng(h3_index) BETWEEN -10 AND 5
LIMIT 100;
```

## Note on Schema Changes

Forecasts produced from March 2026 onward use the lean schema described above: `h3_index` is BIGINT (int64) and geometry/lat/lon/area_km2 columns are omitted. Older forecasts may still contain the previous schema with VARCHAR hex h3_index, geometry, lat, lon, and area_km2 columns.

## Sources

**Weather**: [NOAA AI-NWP](https://registry.opendata.aws/noaa-oar-mlwp/) — AI Neural Weather Prediction models hosted on AWS Open Data. GraphCast_GFS produces 0.25° global forecasts on 13 pressure levels + surface variables, 6-hourly timesteps out to 10 days, initialized from GFS analysis. NetCDF format, updated twice daily.

> Lam, R., Sanchez-Gonzalez, A., Willson, M., et al. (2023). Learning skillful medium-range global weather forecasting. *Science*, 382(6677), 1416–1421. [doi:10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336)

**Terrain**: [GEDTM-30m](https://doi.org/10.5281/zenodo.14900181) via [walkthru-earth/dem-terrain](https://source.coop/walkthru-earth/dem-terrain) ([code](https://github.com/walkthru-earth/dem-terrain)) — Global Ensemble Digital Terrain Model at 30m resolution, converted to H3-indexed Parquet with elevation, slope, aspect, TRI, and TPI derivatives. Used for all topographic corrections in this dataset.

> Ho, Y., Grohmann, C. H., Lindsay, J., Reuter, H. I., Parente, L., Witjes, M., & Hengl, T. (2025). GEDTM30: global ensemble digital terrain model at 30 m and derived multiscale terrain variables. *PeerJ*, 13, e19673. [doi:10.7717/peerj.19673](https://doi.org/10.7717/peerj.19673)

## License

This dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by [walkthru.earth](https://walkthru.earth/links). The source [NOAA AI-NWP](https://registry.opendata.aws/noaa-oar-mlwp/) data is public domain (US Government work).
