# Pipeline Architecture

## Overview

```
NOAA S3 (public)
  └─ GraphCast / FourCastNet / Pangu-Weather NetCDF (0.25° global grid)
       │
       ▼  pipeline/weather.py
  Download + region clip (BBOX ± WEATHER_PADDING)
       │
       ▼  pipeline/dem.py
  Load 30 m Copernicus DEM → GPU terrain derivatives
       │
       ▼  pipeline/h3_grid.py
  Generate H3 hexagonal grid (multi-resolution)
       │
       ▼  pipeline/variables.py + interpolation.py
  GPU RBF interpolation  →  topographic corrections  →  unit conversion
       │
       ▼  pipeline/export.py
  PyArrow → Hive-partitioned Parquet → S3
```

---

## Step 1 — Weather data (`pipeline/weather.py`)

### Source

NOAA `noaa-oar-mlwp-data` S3 bucket (public, no credentials). Available models:

| Model key | S3 prefix | Grid | Update cycle |
|---|---|---|---|
| `GraphCast_GFS` | `GRAP_v100_GFS/` | 0.25° global | 6-hourly |
| `GraphCast_IFS` | `GRAP_v100_IFS/` | 0.25° global | 6-hourly |
| `FourCastNet` | `FOUR_v200_GFS/` | 0.25° global | 6-hourly |
| `Pangu_Weather` | `PANG_v100_GFS/` | 0.25° global | 6-hourly |

### Caching

The downloaded NetCDF is cached in `weather_cache/`. On subsequent runs the file is reused if the S3 key matches. Old files for the same model are evicted automatically.

### Region clipping

When a regional BBOX is configured, the global file is clipped to `BBOX ± WEATHER_PADDING` (default 2°) before processing. This reduces memory while ensuring every H3 target point is fully surrounded by source grid points — the essential condition for NaN-free interpolation. For global BBOX (the default), the full dataset is used without clipping.

**Why padding matters (regional BBOX):**

Without padding, a 0.5° BBOX on a 0.25° grid gives only 2×2 = 4 source points. H3 cell centres near the edges fall outside this tiny square and produce NaN. With `WEATHER_PADDING = 2.0°` the clipped region contains ≥ 18×18 = 324 source points.

---

## Step 2 — DEM (`pipeline/dem.py`)

### Source priority

1. **Copernicus GLO-30** via Microsoft Planetary Computer STAC (primary)
2. **OpenLandMap merged 30 m DEM** via `stac.openlandmap.org` (automatic fallback)

### Terrain derivatives computed on GPU

All derivatives are computed in a single GPU pass immediately after loading:

| Output | Computation |
|---|---|
| `slope` | `arctan(√((∂z/∂x)² + (∂z/∂y)²))` in degrees |
| `aspect` | `(arctan2(-∂z/∂x, -∂z/∂y) × 180/π + 360) mod 360` |
| `tri` | Terrain Ruggedness Index — RMS deviation in 3×3 kernel |
| `tpi` | Topographic Position Index — elevation minus 5×5 neighbourhood mean |

Gradients are computed with correct metric spacing:

```
dx = Δlon × 111320 × cos(lat_centre)   [metres]
dy = Δlat × 111320                     [metres]
```

---

## Step 3 — H3 grid (`pipeline/h3_grid.py`)

H3 (Uber's hexagonal hierarchical spatial index) is used as the output grid because:

- Hexagonal cells have equal area and equal nearest-neighbour distance
- Hierarchical: res 5 cells nest perfectly inside res 4, etc.
- Native support for geospatial joins in DuckDB, Athena, BigQuery

For each configured resolution the module:

1. Defines a `LatLngPoly` matching the BBOX
2. Calls `h3.h3shape_to_cells(polygon, resolution)` to get all cell IDs
3. Vectorises `h3.cell_to_latlng()` over the cell list (no per-cell Python loop)

**No WKT geometries are generated.** Only cell ID, centre lat/lon, area, and resolution are stored. This keeps the DataFrame lean and fast.

---

## Step 4 — Interpolation + corrections (`pipeline/variables.py`, `interpolation.py`, `corrections.py`)

### 4a. Variable extraction

For each variable in `VARIABLE_SPEC` the module:

1. Locates the variable in the xarray Dataset (trying the exact key first, then a case-insensitive substring match)
2. Selects the correct pressure level for upper-air variables (index into the `level` dimension)
3. Ensures a time dimension exists (adds one for 2-D single-time fields)

### 4b. GPU RBF interpolation

Each variable array `(T, Ny, Nx)` is interpolated onto H3 cell centres `(N,)`, producing `(T, N)`.

See [Mathematics → RBF Interpolation](mathematics.md#rbf-interpolation) for equations.

**CPU fallback:** If no GPU is present, `scipy.interpolate.RegularGridInterpolator` (bilinear) is used, with a nearest-neighbour second pass to fill any residual NaN values.

### 4c. Topographic corrections

After interpolation, physically motivated corrections are applied to account for elevation differences between the coarse model grid and the actual H3 cell terrain. See [Mathematics → Topographic Corrections](mathematics.md#topographic-corrections).

### 4d. Unit conversions

| From | To | Condition |
|---|---|---|
| Kelvin | Celsius | if mean > 200 K |
| Pascal | hPa | if mean > 50 000 Pa |
| m (precip) | mm | always |
| m²/s² (geopotential) | m (geopotential height) | divide by g₀ = 9.80665 |

### 4e. Derived variables

Computed after all primary variables are interpolated:

| Variable | Formula |
|---|---|
| `wind_speed` | `√(u₁₀² + v₁₀²)` |
| `wind_direction` | `(arctan2(-u₁₀, -v₁₀) × 180/π + 360) mod 360` |
| `wind_speed_850hPa` | `√(u₈₅₀² + v₈₅₀²)` |
| `wind_shear_magnitude` | `√((u₈₅₀−u₁₀)² + (v₈₅₀−v₁₀)²)` |
| `temp_diff_850hPa_2m` | `T₈₅₀ − T₂ₘ` |
| `moisture_flux_magnitude` | `√((qu)² + (qv)²)` |
| `geopotential_anomaly_500hPa` | `H₅₀₀ − 5574 m` |

---

## Step 5 — Export (`pipeline/export.py`)

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
| Row group size | 100 000 rows | Balance pushdown vs read amplification |
| Max rows per file | 500 000 | ~128 MB files, optimal for Athena/Spark |
| Statistics | Enabled | Predicate pushdown on `h3_index`, `timestamp` |

### Direct S3 streaming

`pyarrow.fs.S3FileSystem` uses the AWS C++ SDK with native multipart upload. No local temp files are written; data streams directly from memory to S3.

---

## Module dependency graph

```
main.py
  ├── pipeline/config.py          (constants, no imports from pipeline/)
  ├── pipeline/gpu.py             (CuPy / numpy abstraction)
  ├── pipeline/weather.py         (→ config)
  ├── pipeline/dem.py             (→ config, gpu)
  ├── pipeline/h3_grid.py         (→ config)
  ├── pipeline/interpolation.py   (→ config, gpu)
  ├── pipeline/corrections.py     (→ config)
  ├── pipeline/variables.py       (→ corrections, interpolation)
  └── pipeline/export.py          (standalone, no pipeline/ imports)
```
