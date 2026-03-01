# Weather Variables Reference

## Input variables (from NOAA NetCDF)

| NetCDF key | Long name | Pressure level | Units (raw) |
|---|---|---|---|
| `t2` | 2-metre temperature | Surface | K |
| `u10` | 10-metre U-component of wind | Surface | m/s |
| `v10` | 10-metre V-component of wind | Surface | m/s |
| `apcp` | Total precipitation (6 h accum.) | Surface | m |
| `q` | Specific humidity | Surface | kg/kg |
| `msl` | Mean sea-level pressure | Surface | Pa |
| `t` | Temperature | 850 hPa (level index 2) | K |
| `u` | U-component of wind | 850 hPa (level index 2) | m/s |
| `v` | V-component of wind | 850 hPa (level index 2) | m/s |
| `w` | Vertical velocity (pressure coords) | 500 hPa (level index 6) | Pa/s |
| `z` | Geopotential | 500 hPa (level index 6) | m²/s² |

---

## Output columns (Parquet)

### Grid metadata columns

| Column | Type | Description |
|---|---|---|
| `h3_index` | string | H3 cell ID (hex string) |
| `lat` | float32 | Cell centre latitude (°N) |
| `lon` | float32 | Cell centre longitude (°E) |
| `area_km2` | float32 | H3 cell area (km²) |
| `timestamp` | timestamp[s, UTC] | Forecast valid time |

### Partition columns (also present as regular columns)

| Column | Type | Description |
|---|---|---|
| `model` | string | e.g. `GraphCast_GFS` |
| `date` | string | `YYYY-MM-DD` of run start |
| `hour` | uint8 | Hour of run start (UTC) |
| `h3_res` | uint8 | H3 resolution (5–10) |

### Weather columns — primary (after topographic correction)

| Column | Units | Correction applied |
|---|---|---|
| `temperature_2m_C` | °C | Variable lapse rate (derived from T_850 − T_2m per timestep) |
| `wind_u_10m_ms` | m/s | Elevation + slope channelling |
| `wind_v_10m_ms` | m/s | Elevation + slope channelling |
| `precipitation_mm_6hr` | mm | Dynamic orographic enhancement (wind-direction-aware) |
| `specific_humidity_gkg` | g/kg | Exponential elevation adjustment (H_q = 2 km scale height) |
| `pressure_msl_hPa` | hPa | None |
| `temperature_850hPa_C` | °C | None (free-atmosphere variable) |
| `wind_u_850hPa_ms` | m/s | None (free-atmosphere variable) |
| `wind_v_850hPa_ms` | m/s | None (free-atmosphere variable) |
| `vertical_velocity_500hPa_Pas` | Pa/s | None (free-atmosphere variable) |
| `geopotential_500hPa_m` | m | None (geopotential height) |

> **Sign convention for vertical velocity:** negative Pa/s = upward motion (lower pressure above).

### Weather columns — derived

| Column | Units | Formula |
|---|---|---|
| `wind_speed_10m_ms` | m/s | `√(u₁₀² + v₁₀²)` |
| `wind_direction_10m_deg` | ° (met.) | `(arctan2(−u,−v)×180/π+360) mod 360` |
| `wind_speed_850hPa_ms` | m/s | `√(u₈₅₀² + v₈₅₀²)` |
| `wind_direction_850hPa_deg` | ° (met.) | same formula at 850 hPa |
| `wind_shear_magnitude_ms` | m/s | `√((u₈₅₀−u₁₀)²+(v₈₅₀−v₁₀)²)` |
| `wind_shear_direction_deg` | ° (met.) | direction of shear vector |
| `temp_diff_850hPa_2m_C` | °C | `T₈₅₀ − T₂ₘ` |
| `moisture_flux_u` | g/kg·m/s | `q × u₁₀` |
| `moisture_flux_v` | g/kg·m/s | `q × v₁₀` |
| `moisture_flux_magnitude` | g/kg·m/s | `√((qu)²+(qv)²)` |
| `geopotential_anomaly_500hPa_m` | m | `H₅₀₀ − 5574` (ICAO Standard Atmosphere reference) |

---

## Meteorological wind direction convention

Wind direction follows the meteorological convention: **direction *from* which the wind blows**, measured clockwise from north.

| Degrees | Direction |
|---|---|
| 0° / 360° | From north (northerly wind) |
| 90° | From east (easterly wind) |
| 180° | From south (southerly wind) |
| 270° | From west (westerly wind) |

---

## Interpreting stability variables

### `temp_diff_850hPa_2m_C` (T₈₅₀ − T₂ₘ)

| Value | Interpretation |
|---|---|
| Strongly negative (< −15°C) | Highly unstable, convection likely |
| Moderately negative (−5 to −15°C) | Conditionally unstable |
| Near zero (−5 to 0°C) | Near-neutral stability |
| Positive | Temperature inversion, very stable |

### `wind_shear_magnitude_ms` (850 hPa − 10 m, low-level shear)

This is a **low-level** (0–1.5 km) shear metric, not the standard 0–6 km bulk shear used for supercell forecasting. The depth of the layer depends on the surface elevation of the target region.

| Value | Interpretation |
|---|---|
| < 5 m/s | Weak low-level shear |
| 5–10 m/s | Moderate; mesoscale organisation possible |
| > 10 m/s | Strong low-level shear; low-level jet signature |

### `geopotential_anomaly_500hPa_m`

| Value | Interpretation |
|---|---|
| Strongly positive | Upper-level ridge, subsidence, dry conditions |
| Near zero | Near-normal synoptic pattern |
| Strongly negative | Upper-level trough, lift, precipitation risk |

### `vertical_velocity_500hPa_Pas`

| Value | Interpretation |
|---|---|
| < −0.5 Pa/s | Strong ascent, deep convection |
| −0.5 to 0 Pa/s | Weak ascent |
| 0 to +0.5 Pa/s | Weak descent |
| > +0.5 Pa/s | Subsidence, suppressed convection |
