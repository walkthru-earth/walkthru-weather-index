# Global Weather Forecasts (H3-indexed)

AI-powered weather forecasts from [NOAA AI-NWP](https://registry.opendata.aws/noaa-oar-mlwp/) (GraphCast) downscaled to [H3](https://h3geo.org/) hexagonal cells with topographic corrections, in [native Parquet 2.11+ GEOMETRY](https://github.com/apache/parquet-format/blob/master/Geospatial.md) format. Updated automatically every 12 hours, 5-day forecasts at 6-hour intervals, ~2M cells per timestep at H3 resolution 5.

| | |
|---|---|
| **Source** | [NOAA AI-NWP](https://registry.opendata.aws/noaa-oar-mlwp/) GraphCast_GFS, 0.25° global grid, 6-hourly timesteps |
| **Terrain** | Topographic corrections via [walkthru-earth/dem-terrain](https://source.coop/walkthru-earth/dem-terrain) ([code](https://github.com/walkthru-earth/dem-terrain)) — H3-indexed [GEDTM-30m](https://doi.org/10.5281/zenodo.14900181) elevation, slope, aspect |
| **Format** | Apache Parquet with native GEOMETRY logical type (DuckDB 1.5), Hive-partitioned |
| **CRS** | EPSG:4326 (WGS 84) |
| **Update** | Automated every 12 hours via GitHub Actions + HuggingFace Jobs (A10G GPU) |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by [walkthru.earth](https://walkthru.earth/links) |
| **Code** | [walkthru-earth/walkthru-weather-index](https://github.com/walkthru-earth/walkthru-weather-index) |

## Quick Start

```sql
-- DuckDB
INSTALL spatial; LOAD spatial;
INSTALL httpfs;  LOAD httpfs;
SET s3_region = 'us-west-2';

SELECT h3_index, lat, lon, timestamp,
       temperature_2m_C, wind_speed_10m_ms, precipitation_mm_6hr,
       pressure_msl_hPa
FROM read_parquet(
    's3://us-west-2.opendata.source.coop/walkthru-earth/indices/weather/model=GraphCast_GFS/date=2026-03-02/hour=0/h3_res=5/*.parquet',
    hive_partitioning = true
)
WHERE lat BETWEEN 35 AND 45 AND lon BETWEEN -10 AND 5
ORDER BY timestamp, temperature_2m_C DESC
LIMIT 20;
```

```python
# Python
import duckdb

con = duckdb.connect()
for ext in ("spatial", "httpfs"):
    con.install_extension(ext); con.load_extension(ext)
con.sql("SET s3_region = 'us-west-2'")

df = con.sql("""
    SELECT h3_index, lat, lon, timestamp,
           temperature_2m_C, wind_speed_10m_ms, pressure_msl_hPa
    FROM read_parquet(
        's3://us-west-2.opendata.source.coop/walkthru-earth/indices/weather/model=GraphCast_GFS/date=2026-03-02/hour=0/h3_res=5/*.parquet',
        hive_partitioning = true
    ) WHERE lat BETWEEN 20 AND 35 AND lon BETWEEN 68 AND 90
""").fetchdf()
```

## Files

```
walkthru-earth/indices/weather/
  model=GraphCast_GFS/
    date=2026-03-02/
      hour=0/
        h3_res=5/
          part-0.parquet .. part-84.parquet    ~3.6 GB   42M rows (2M cells × 21 timesteps)
      hour=12/
        h3_res=5/
          part-0.parquet .. part-84.parquet    ~3.6 GB
    date=2026-03-03/
      hour=0/
        h3_res=5/
          part-0.parquet .. part-84.parquet    ~3.6 GB
    ...
```

Each forecast run: **85 part files**, ~3.6 GB total, ~42M rows (2,016,842 unique H3 cells × 21 timesteps over 5 days). New forecasts are appended every 12 hours. Compression: ZSTD. Sorted by `h3_index`.

## Schema

### Grid metadata

| Column | Type | Description |
|--------|------|-------------|
| `h3_index` | VARCHAR | H3 cell ID (hex string) |
| `geometry` | GEOMETRY | Cell center point (native Parquet 2.11+ GEOMETRY, EPSG:4326) |
| `lat` | FLOAT | Cell center latitude (degrees) |
| `lon` | FLOAT | Cell center longitude (degrees) |
| `area_km2` | FLOAT | H3 cell area (km²) |
| `timestamp` | TIMESTAMPTZ | Forecast valid time (UTC) |

### Partition columns

| Column | Type | Description |
|--------|------|-------------|
| `model` | VARCHAR | Model name (e.g. `GraphCast_GFS`) |
| `date` | DATE | Forecast run start date |
| `hour` | BIGINT | Forecast run start hour (UTC) |
| `h3_res` | BIGINT | H3 resolution (currently 5) |

### Weather variables — primary (topographically corrected)

| Column | Units | Topographic correction |
|--------|-------|------------------------|
| `temperature_2m_C` | °C | Variable lapse rate (derived from T850 − T2m per timestep) |
| `wind_u_10m_ms` | m/s | Elevation + slope channelling |
| `wind_v_10m_ms` | m/s | Elevation + slope channelling |
| `precipitation_mm_6hr` | mm | Dynamic orographic enhancement (wind-direction-aware) |
| `specific_humidity_gkg` | g/kg | Exponential elevation adjustment (scale height 2 km) |
| `pressure_msl_hPa` | hPa | None (sea-level reference) |
| `temperature_850hPa_C` | °C | None (free-atmosphere) |
| `wind_u_850hPa_ms` | m/s | None (free-atmosphere) |
| `wind_v_850hPa_ms` | m/s | None (free-atmosphere) |
| `vertical_velocity_500hPa_Pas` | Pa/s | None (negative = upward motion) |
| `geopotential_500hPa_m` | m | None (geopotential height) |

### Weather variables — derived

| Column | Units | Formula |
|--------|-------|---------|
| `wind_speed_10m_ms` | m/s | sqrt(u10² + v10²) |
| `wind_direction_10m_deg` | ° (meteorological) | Direction wind blows *from*, 0° = N |
| `wind_speed_850hPa_ms` | m/s | sqrt(u850² + v850²) |
| `wind_direction_850hPa_deg` | ° (meteorological) | Same convention at 850 hPa |
| `wind_shear_magnitude_ms` | m/s | sqrt((u850−u10)² + (v850−v10)²) |
| `wind_shear_direction_deg` | ° (meteorological) | Shear vector direction |
| `temp_diff_850hPa_2m_C` | °C | T850 − T2m (stability indicator) |
| `moisture_flux_u` | g/kg·m/s | q × u10 |
| `moisture_flux_v` | g/kg·m/s | q × v10 |
| `moisture_flux_magnitude` | g/kg·m/s | sqrt((qu)² + (qv)²) |
| `geopotential_anomaly_500hPa_m` | m | H500 − 5574 (ICAO standard reference) |

**Sample values** (GraphCast_GFS, 2026-03-02 00Z, hottest cells):

| h3_index | lat | lon | temp_C | wind_ms | press_hPa | precip_mm |
|----------|-----|-----|--------|---------|-----------|-----------|
| 85becd33fffffff | -28.9 | +143.5 | 33.1 | 8.8 | 1006.3 | 0.00 |
| 85becd37fffffff | -28.9 | +143.3 | 33.0 | 9.0 | 1006.1 | 0.00 |
| 85becd23fffffff | -28.8 | +143.3 | 33.0 | 9.1 | 1006.3 | 0.00 |

## How It Works

1. GitHub Actions polls NOAA S3 every 12 hours for new AI-NWP forecasts (GraphCast_GFS NetCDF)
2. When a new forecast is detected, a HuggingFace Job is triggered on an A10G GPU
3. The 0.25° global NetCDF is interpolated onto ~2M H3 cells (resolution 5) using GPU-accelerated bilinear interpolation
4. Topographic corrections are applied using the [walkthru-earth/dem-terrain](https://source.coop/walkthru-earth/dem-terrain) dataset — H3-indexed terrain derivatives (elevation, slope, aspect, TRI, TPI) computed from the [GEDTM-30m](https://doi.org/10.5281/zenodo.14900181) global DEM. The terrain data is joined by `h3_index` to apply physically-based corrections: variable lapse rates for temperature, slope channelling for wind, and orographic enhancement for precipitation
5. 21 timesteps (6-hourly, 5-day forecast) are written as Hive-partitioned Parquet to S3
6. DuckDB spatial adds native GEOMETRY with per-row-group bounding box statistics

## More Examples

```sql
-- Latest 5-day temperature forecast for a city (e.g. London)
SELECT timestamp, temperature_2m_C, wind_speed_10m_ms,
       precipitation_mm_6hr, pressure_msl_hPa
FROM read_parquet(
    's3://us-west-2.opendata.source.coop/walkthru-earth/indices/weather/model=GraphCast_GFS/date=2026-03-03/hour=0/h3_res=5/*.parquet',
    hive_partitioning = true
)
WHERE lat BETWEEN 51.0 AND 52.0 AND lon BETWEEN -0.5 AND 0.5
ORDER BY timestamp
LIMIT 100;

-- Wind shear analysis (severe weather indicator)
SELECT h3_index, lat, lon, timestamp,
       wind_shear_magnitude_ms, temp_diff_850hPa_2m_C
FROM read_parquet(
    's3://us-west-2.opendata.source.coop/walkthru-earth/indices/weather/model=GraphCast_GFS/date=2026-03-02/hour=0/h3_res=5/*.parquet',
    hive_partitioning = true
)
WHERE wind_shear_magnitude_ms > 20
ORDER BY wind_shear_magnitude_ms DESC
LIMIT 20;

-- DuckDB-WASM (browser) — use HTTPS URL
SELECT h3_index, timestamp, temperature_2m_C, wind_speed_10m_ms
FROM read_parquet(
    'https://data.source.coop/walkthru-earth/indices/weather/model=GraphCast_GFS/date=2026-03-02/hour=0/h3_res=5/part-0.parquet'
)
WHERE lat BETWEEN 35 AND 45 AND lon BETWEEN -10 AND 5
LIMIT 100;
```

## Geometry Format

The `geometry` column uses the [native Parquet 2.11+ GEOMETRY logical type](https://github.com/apache/parquet-format/blob/master/Geospatial.md) with GeoParquet 1.0 file-level metadata for backwards compatibility (`GEOPARQUET_VERSION 'BOTH'`). DuckDB 1.5+ writes per-row-group bounding box statistics automatically.

Supported by: DuckDB 1.5+, Apache Arrow (Rust), Apache Iceberg, GDAL 3.12+.

## Sources

**Weather**: [NOAA AI-NWP](https://registry.opendata.aws/noaa-oar-mlwp/) — AI Neural Weather Prediction models hosted on AWS Open Data. GraphCast_GFS produces 0.25° global forecasts on 13 pressure levels + surface variables, 6-hourly timesteps out to 10 days, initialized from GFS analysis. NetCDF format, updated twice daily.

> Lam, R., Sanchez-Gonzalez, A., Willson, M., et al. (2023). Learning skillful medium-range global weather forecasting. *Science*, 382(6677), 1416–1421. [doi:10.1126/science.adi2336](https://doi.org/10.1126/science.adi2336)

**Terrain**: [GEDTM-30m](https://doi.org/10.5281/zenodo.14900181) via [walkthru-earth/dem-terrain](https://source.coop/walkthru-earth/dem-terrain) ([code](https://github.com/walkthru-earth/dem-terrain)) — Global Ensemble Digital Terrain Model at 30m resolution, converted to H3-indexed Parquet with elevation, slope, aspect, TRI, and TPI derivatives. Used for all topographic corrections in this dataset.

> Ho, Y., Grohmann, C. H., Lindsay, J., Reuter, H. I., Parente, L., Witjes, M., & Hengl, T. (2025). GEDTM30: global ensemble digital terrain model at 30 m and derived multiscale terrain variables. *PeerJ*, 13, e19673. [doi:10.7717/peerj.19673](https://doi.org/10.7717/peerj.19673)

## License

This dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by [walkthru.earth](https://walkthru.earth/links). The source [NOAA AI-NWP](https://registry.opendata.aws/noaa-oar-mlwp/) data is public domain (US Government work).
