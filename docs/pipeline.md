# Pipeline Architecture

## Overview

```
NOAA S3 (public)
  +- GraphCast / FourCastNet / Pangu-Weather NetCDF (0.25 deg global grid)
       |
       v  pipeline/weather.py
  Download + region clip (BBOX +/- WEATHER_PADDING)
       |
       v  pipeline/h3_grid.py
  Generate H3 hexagonal grid (multi-resolution)
       |
       v  pipeline/dem.py
  Load H3-indexed terrain from Source Cooperative Parquet (or STAC fallback)
       |
       v  pipeline/variables.py + interpolation.py + corrections.py
  GPU bilinear interpolation  ->  topographic corrections  ->  unit conversion
       |
       v  pipeline/export.py
  PyArrow -> Hive-partitioned Parquet -> S3
```

---

## Step 1 -- Weather data (`pipeline/weather.py`)

### Source

NOAA `noaa-oar-mlwp-data` S3 bucket (public, no credentials). Available models:

| Model key | S3 prefix | Grid | Update cycle |
|---|---|---|---|
| `GraphCast_GFS` | `GRAP_v100_GFS/` | 0.25 deg global | 6-hourly |
| `GraphCast_IFS` | `GRAP_v100_IFS/` | 0.25 deg global | 6-hourly |
| `FourCastNet` | `FOUR_v200_GFS/` | 0.25 deg global | 6-hourly |
| `Pangu_Weather` | `PANG_v100_GFS/` | 0.25 deg global | 6-hourly |

### Caching

The downloaded NetCDF is cached in `weather_cache/`. On subsequent runs the file is reused if the S3 key matches. Old files for the same model are evicted automatically.

### Region clipping

When a regional BBOX is configured, the global file is clipped to `BBOX +/- WEATHER_PADDING` (default 2 deg) before processing. This reduces memory while ensuring every H3 target point is fully surrounded by source grid points -- the essential condition for NaN-free interpolation. For global BBOX (the default), the full dataset is used without clipping.

**Why padding matters (regional BBOX):**

Without padding, a 0.5 deg BBOX on a 0.25 deg grid gives only 2x2 = 4 source points. H3 cell centres near the edges fall outside this tiny square and produce NaN. With `WEATHER_PADDING = 2.0` the clipped region contains >= 18x18 = 324 source points.

---

## Step 2 -- H3 grid (`pipeline/h3_grid.py`)

H3 (Uber's hexagonal hierarchical spatial index) is used as the output grid because:

- Hexagonal cells have equal area and equal nearest-neighbour distance
- Hierarchical: res 5 cells nest perfectly inside res 4, etc.
- Native support for geospatial joins in DuckDB, Athena, BigQuery

For each configured resolution the module:

1. **Global BBOX**: Uses `h3.uncompact_cells(get_res0_cells(), res)` to expand all 122 base cells to the target resolution (LatLngPoly cannot represent the full globe)
2. **Regional BBOX**: Defines a `LatLngPoly` with 4 corners and calls `h3.h3shape_to_cells(polygon, res)`
3. Vectorises `h3.cell_to_latlng()` over all cells to extract centre coordinates
4. Computes cell area via `h3.cell_area(cell, 'km^2')`

**No WKT geometries are generated.** Only cell ID, centre lat/lon, area, and resolution are stored.

---

## Step 3 -- DEM (`pipeline/dem.py`)

### Source priority

1. **Pre-computed H3 Parquet** on [Source Cooperative](https://source.coop/walkthru-earth/dem-terrain) (primary) -- terrain derivatives already computed at H3 cell centres, loaded per resolution. No interpolation needed.
2. **Copernicus GLO-30** via Microsoft Planetary Computer STAC (fallback for resolutions > 7 or when Parquet is unavailable)
3. **OpenLandMap merged 30 m DEM** via `stac.openlandmap.org` (fallback #2)

### H3 Parquet path (default)

For resolutions 1-7, the pipeline reads pre-computed terrain from [walkthru-earth/dem-terrain](https://github.com/walkthru-earth/dem-terrain):

```
s3://us-west-2.opendata.source.coop/walkthru-earth/dem-terrain/h3_res={res}/data.parquet
```

Each file contains `h3_index, elev, slope, aspect, tri, tpi` already at H3 cell centres. The pipeline joins on `h3_index` to align DEM values with the H3 grid -- no raster loading, no GPU terrain computation, no `RegularGridInterpolator`. This reduces the DEM step from ~30 min to ~5 sec for global runs.

### STAC raster fallback

When Parquet is unavailable (resolution > 7, or `--no-parquet-dem` flag), the pipeline falls back to STAC raster loading. All terrain derivatives are computed in a single GPU pass:

| Output | Computation |
|---|---|
| `slope` | `arctan(sqrt((dz/dx)^2 + (dz/dy)^2))` in degrees |
| `aspect` | `(arctan2(-dz/dx, -dz/dy) * 180/pi + 360) mod 360` |
| `tri` | Terrain Ruggedness Index -- RMS deviation in 3x3 kernel |
| `tpi` | Topographic Position Index -- elevation minus 5x5 neighbourhood mean |

Gradients are computed with correct metric spacing:

```
dx = d_lon * 111320 * cos(lat_centre)   [metres]
dy = d_lat * 111320                     [metres]
```

---

## Step 4 -- Interpolation + corrections (`pipeline/variables.py`, `interpolation.py`, `corrections.py`)

### 4a. Variable extraction

For each variable in `VARIABLE_SPEC` the module:

1. Locates the variable in the xarray Dataset (trying the exact key first, then a case-insensitive substring match)
2. Selects the correct pressure level for upper-air variables (index into the `level` dimension)
3. Ensures a time dimension exists (adds one for 2-D single-time fields)

### 4b. Bilinear interpolation

Each variable array `(T, Ny, Nx)` is interpolated onto H3 cell centres `(N,)`, producing `(T, N)`.

**GPU path**: CuPy `map_coordinates` with `order=1` (bilinear). Target lat/lon are converted to fractional grid indices. Longitude wrap-around is handled via circular padding (3 columns from each edge). Latitude is clamped at poles. Complexity: O(N) per timestep.

**CPU fallback**: `scipy.interpolate.RegularGridInterpolator` with `method='linear'`, with a nearest-neighbour second pass to fill any residual NaN values.

See [Mathematics > Bilinear Interpolation](mathematics.md#4-bilinear-interpolation) for equations.

### 4c. Topographic corrections

After interpolation, physically motivated corrections are applied to account for elevation differences between the coarse model grid and the actual H3 cell terrain. See [Mathematics > Topographic Corrections](mathematics.md#5-topographic-corrections).

### 4d. Unit conversions

| From | To | Condition |
|---|---|---|
| Kelvin | Celsius | if mean > 200 K |
| Pascal | hPa | if mean > 50 000 Pa |
| m (precip) | mm | always |
| m2/s2 (geopotential) | m (geopotential height) | divide by g0 = 9.80665 |
| kg/kg (humidity) | g/kg | if mean < 1.0 |

### 4e. Derived variables

Computed after all primary variables are interpolated:

| Variable | Formula |
|---|---|
| `wind_speed` | `sqrt(u10^2 + v10^2)` |
| `wind_direction` | `(arctan2(-u10, -v10) * 180/pi + 360) mod 360` |
| `wind_speed_850hPa` | `sqrt(u850^2 + v850^2)` |
| `wind_shear_magnitude` | `sqrt((u850-u10)^2 + (v850-v10)^2)` |
| `temp_diff_850hPa_2m` | `T850 - T2m` |
| `moisture_flux_magnitude` | `sqrt((qu)^2 + (qv)^2)` |
| `geopotential_anomaly_500hPa` | `H500 - 5574 m` |

---

## Step 5 -- Export (`pipeline/export.py`)

### Progressive writes

Each H3 resolution is written to S3 immediately after interpolation completes. This ensures partial results survive if a later resolution fails (e.g. res 5 is safely persisted before res 7 starts).

### Partition scheme

```
s3://{bucket}/weather/
  model={model_name}/
    date={YYYY-MM-DD}/
      hour={HH}/
        h3_res={resolution}/
          part-00000.parquet
```

Partition columns (`model`, `date`, `hour`, `h3_res`) are also present as regular columns inside the file to support predicate pushdown without full partition path parsing.

**Do not** partition by `h3_index`. At res 9 there are ~4.8 billion possible cells; creating one directory per cell would destroy S3 LIST API performance. Instead `h3_index` is stored as a column with `write_statistics=True` enabling min/max pushdown.

### Write settings

| Setting | Value | Reason |
|---|---|---|
| Compression | ZSTD level 3 | Best ratio for float32 weather data |
| Row group size | 50k-100k rows | Balance pushdown vs read amplification |
| Max rows per file | 500 000 | ~128 MB files, optimal for Athena/Spark |
| Statistics | Enabled | Predicate pushdown on `h3_index`, `timestamp` |

### Direct S3 streaming

`pyarrow.fs.S3FileSystem` uses the AWS C++ SDK with native multipart upload. No local temp files are written; data streams directly from memory to S3.

---

## Module dependency graph

```
main.py
  +-- pipeline/config.py          (constants, no imports from pipeline/)
  +-- pipeline/gpu.py             (CuPy / numpy abstraction)
  +-- pipeline/weather.py         (-> config)
  +-- pipeline/h3_grid.py         (-> config)
  +-- pipeline/dem.py             (-> config, gpu)
  +-- pipeline/interpolation.py   (-> gpu)
  +-- pipeline/corrections.py     (-> config)
  +-- pipeline/variables.py       (-> corrections, interpolation)
  +-- pipeline/export.py          (standalone, no pipeline/ imports)
```
