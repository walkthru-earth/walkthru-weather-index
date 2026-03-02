"""Weather variable extraction, unit conversion, and derived quantities."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from pipeline.config import AI_MODELS, GEOPOTENTIAL_500_REF, TOPO_PARAMS
from pipeline.corrections import apply as topo_apply, interp_dem_field
from pipeline.interpolation import interpolate_to_points

log = logging.getLogger(__name__)

# variable key → (output_name, correction_type, unit_conversion, pressure_level_index)
VARIABLE_SPEC: dict[str, tuple[str, str, str, int | None]] = {
    "t2": (
        "temperature_2m",
        "none",
        "K→C",
        None,
    ),  # lapse rate applied post-hoc via variable rate
    "u10": ("wind_u_10m", "wind_elevation", "m/s", None),
    "v10": ("wind_v_10m", "wind_elevation", "m/s", None),
    "apcp": (
        "precipitation",
        "none",
        "m→mm",
        None,
    ),  # orographic applied post-hoc with dynamic wind
    "q": ("specific_humidity_surface", "elevation", "kg/kg→g/kg", None),
    "msl": ("pressure_msl", "none", "Pa→hPa", None),
    "w": ("vertical_velocity_500hPa", "none", "Pa/s", 5),
    "t": ("temperature_850hPa", "none", "K→C", 2),
    "u": ("wind_u_850hPa", "none", "m/s", 2),
    "v": ("wind_v_850hPa", "none", "m/s", 2),
    "z": ("geopotential_500hPa", "none", "m²/s²→m", 5),
}


def extract_all(
    ds: xr.Dataset,
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
    dem: dict,
    model_name: str = "GraphCast_GFS",
    reference_elevation: float = 0.0,
) -> dict[str, np.ndarray]:
    """Interpolate all known variables, apply corrections and unit conversions.

    Returns dict of output_name → ndarray (T, N_targets).
    """
    src_lats = ds.latitude.values
    src_lons = ds.longitude.values
    results: dict[str, np.ndarray] = {}

    # Model-specific variable exclusions
    skip_vars = set(AI_MODELS.get(model_name, {}).get("skip_vars", []))
    if skip_vars:
        log.info("[SKIP] Variables not available in %s: %s", model_name, skip_vars)

    # Validate pressure level dimension ordering
    if "level" in ds.dims:
        levels = ds.level.values
        log.info("Pressure levels: %s", levels)

    for key, (name, correction, units, level_idx) in VARIABLE_SPEC.items():
        if key in skip_vars:
            log.info("[SKIP] %s: not available in %s", name, model_name)
            continue

        var = _find_var(ds, key)
        if var is None:
            continue

        values = _select_level(var, level_idx).values
        if values.ndim == 2:
            values = values[np.newaxis]  # add time dim

        interp = interpolate_to_points(values, src_lons, src_lats, tgt_lons, tgt_lats)
        interp = topo_apply(
            interp, tgt_lats, tgt_lons, dem, correction, reference_elevation
        )
        interp = _convert_units(interp, units)

        results[name] = interp
        log.info("[INTERP] %s: %d ts x %d pts", name, interp.shape[0], interp.shape[1])

    # Compute variable lapse rate BEFORE applying lapse rate correction
    _apply_variable_lapse_rate(results, tgt_lats, tgt_lons, dem, reference_elevation)

    # Apply orographic precipitation correction with dynamic wind direction
    _apply_dynamic_orographic(results, tgt_lats, tgt_lons, dem, reference_elevation)

    _add_derived(results)
    return results


# ── helpers ───────────────────────────────────────────────────────────────────


def _find_var(ds: xr.Dataset, key: str):
    if key in ds.data_vars:
        return ds[key]
    for v in ds.data_vars:
        if key in v.lower():
            return ds[v]
    return None


def _select_level(var: xr.DataArray, level_idx: int | None) -> xr.DataArray:
    if "level" not in var.dims:
        return var
    idx = level_idx if level_idx is not None else -1
    return var.isel(level=idx)


def _convert_units(arr: np.ndarray, units: str) -> np.ndarray:
    if units == "K→C" and np.nanmean(arr) > 200:
        return arr - 273.15
    if units == "Pa→hPa" and np.nanmean(arr) > 50000:
        return arr / 100.0
    if units == "m→mm":
        return arr * 1000.0
    if units == "m²/s²→m":
        return arr / 9.80665
    if units == "kg/kg→g/kg" and np.nanmean(arr) < 1.0:
        return arr * 1000.0
    return arr


def _apply_variable_lapse_rate(
    results: dict[str, np.ndarray],
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
    dem: dict,
    reference_elevation: float = 0.0,
) -> None:
    """Apply variable lapse-rate correction to 2m temperature.

    Derives the lapse rate per timestep from T_850 and T_2m instead of using
    the fixed ISA value of -6.5 °C/km.  Falls back to the fixed rate when
    T_850 is not available.

    References
    ----------
    Karger et al. (2023), ESSD, doi:10.5194/essd-15-2445-2023  (CHELSA-W5E5)
    Dutra et al. (2020), Earth & Space Science, doi:10.1029/2019EA000984
    """
    t2m = results.get("temperature_2m")
    if t2m is None:
        return

    elev = interp_dem_field(dem, "elev", tgt_lats, tgt_lons)
    dz = elev - reference_elevation

    t85 = results.get("temperature_850hPa")
    if t85 is not None:
        # Height of 850 hPa above the reference surface
        # 850 hPa ≈ 1500 m MSL in standard atmosphere
        z_850 = 1500.0 - reference_elevation
        # Variable lapse rate per timestep: Γ = (T_850 - T_2m) / z_850  [°C/m]
        # Shape: (T, N) → take spatial mean per timestep → (T, 1)
        gamma = np.nanmean(t85 - t2m, axis=1, keepdims=True) / z_850  # °C/m
        # Clamp to physically plausible range: -9.8 to +5 °C/km
        gamma = np.clip(gamma, -9.8 / 1000.0, 5.0 / 1000.0)
        correction = gamma * dz[None, :]
        log.info("Variable lapse rate: mean = %.1f C/km", np.nanmean(gamma) * 1000)
    else:
        # Fallback to fixed ISA lapse rate (no slope enhancement — lacks literature basis)
        correction = (TOPO_PARAMS["temp_lapse_rate"] / 1000.0) * dz[None, :]
        log.info(
            "Fixed lapse rate: %.1f C/km (T_850 unavailable)",
            TOPO_PARAMS["temp_lapse_rate"],
        )

    results["temperature_2m"] = t2m + correction


def _apply_dynamic_orographic(
    results: dict[str, np.ndarray],
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
    dem: dict,
    reference_elevation: float = 0.0,
) -> None:
    """Apply orographic precipitation enhancement using dynamic wind direction.

    Uses actual model u10/v10 per timestep to determine windward vs leeward
    slopes, instead of a fixed prevailing wind assumption.

    References
    ----------
    Karger et al. (2017), Scientific Data, doi:10.1038/sdata2017122  (CHELSA)
    Roe & Baker (2019), Scientific Reports, doi:10.1038/s41598-019-49974-5
    """
    precip = results.get("precipitation")
    if precip is None:
        return

    elev = interp_dem_field(dem, "elev", tgt_lats, tgt_lons)
    slop = interp_dem_field(dem, "slope", tgt_lats, tgt_lons)
    aspc = interp_dem_field(dem, "aspect", tgt_lats, tgt_lons)
    dz = elev - reference_elevation

    # Base elevation enhancement (unchanged)
    base = 1.0 + np.maximum(dz / 500.0, 0) * 0.15

    # Dynamic wind direction from model output (per timestep)
    u10 = results.get("wind_u_10m")
    v10 = results.get("wind_v_10m")

    if u10 is not None and v10 is not None:
        # Wind direction "from" per timestep: (T,N) → (T,N)
        wind_dir = (np.arctan2(-u10, -v10) * 180 / np.pi) % 360
        # Windward factor: compare slope aspect to wind direction per timestep
        # When slope faces INTO the wind (aspect ≈ wind_dir), maximum enhancement
        adiff = np.abs(aspc[None, :] - wind_dir)
        adiff = np.minimum(adiff, 360 - adiff)
        wind_f = 1.0 + (1.0 - adiff / 180.0) * 0.5
    else:
        # No wind data — skip directional enhancement (neutral factor)
        wind_f = np.ones((1, len(tgt_lats)))

    slop_f = 1.0 + np.clip(slop / 45.0, 0, 1) * 0.3

    # Apply: base and slop_f are (N,), wind_f is (T,N)
    results["precipitation"] = precip * (base * slop_f)[None, :] * wind_f


def _add_derived(results: dict[str, np.ndarray]) -> None:
    u, v = results.get("wind_u_10m"), results.get("wind_v_10m")
    if u is not None and v is not None:
        results["wind_speed"] = np.sqrt(u**2 + v**2)
        results["wind_direction"] = (np.arctan2(-u, -v) * 180 / np.pi) % 360

    u8, v8 = results.get("wind_u_850hPa"), results.get("wind_v_850hPa")
    if u8 is not None and v8 is not None:
        results["wind_speed_850hPa"] = np.sqrt(u8**2 + v8**2)
        results["wind_direction_850hPa"] = (np.arctan2(-u8, -v8) * 180 / np.pi) % 360
        if u is not None and v is not None:
            su = u8 - u
            sv = v8 - v
            results["wind_shear_magnitude"] = np.sqrt(su**2 + sv**2)
            results["wind_shear_direction"] = (np.arctan2(-su, -sv) * 180 / np.pi) % 360

    t2m = results.get("temperature_2m")
    t85 = results.get("temperature_850hPa")
    if t2m is not None and t85 is not None:
        results["temp_diff_850hPa_2m"] = t85 - t2m

    q = results.get("specific_humidity_surface")
    if q is not None and u is not None and v is not None:
        results["moisture_flux_u"] = q * u
        results["moisture_flux_v"] = q * v
        results["moisture_flux_magnitude"] = np.sqrt((q * u) ** 2 + (q * v) ** 2)

    z500 = results.get("geopotential_500hPa")
    if z500 is not None:
        results["geopotential_anomaly_500hPa"] = z500 - GEOPOTENTIAL_500_REF
