"""Topographic corrections applied after interpolation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pipeline.config import TOPO_PARAMS


def apply(
    data: np.ndarray,  # (T, N)
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
    dem: dict,
    correction_type: str,
    reference_elevation: float = 0.0,
) -> np.ndarray:
    """Apply a topographic correction to interpolated *data*."""
    if correction_type == "none":
        return data

    elev = _interp_dem(dem["lat"], dem["lon"], dem["elev"], tgt_lats, tgt_lons)
    slop = _interp_dem(dem["lat"], dem["lon"], dem["slope"], tgt_lats, tgt_lons)
    dz = elev - reference_elevation

    if correction_type == "wind_elevation":
        enh = np.maximum(1.0 + TOPO_PARAMS["wind_height_factor"] * (dz / 1000.0), 0.1)
        enh *= 1.0 + np.clip(slop / 30.0, 0, 1) * 0.3
        return data * enh[None, :]

    if correction_type == "elevation":
        # Exponential moisture scale-height profile (H_q ≈ 2000 m)
        # Held & Soden (2006), J. Climate, doi:10.1175/JCLI3990.1
        fac = np.clip(np.exp(-dz / 2000.0), 0.05, 1.5)
        return data * fac[None, :]

    return data


def interp_dem_field(
    dem: dict,
    field: str,
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
) -> np.ndarray:
    """Interpolate a single DEM field onto target points (public helper)."""
    return _interp_dem(dem["lat"], dem["lon"], dem[field], tgt_lats, tgt_lons)


def _interp_dem(
    dem_lats: np.ndarray,
    dem_lons: np.ndarray,
    values: np.ndarray,
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
) -> np.ndarray:
    interp = RegularGridInterpolator(
        (dem_lats, dem_lons),
        values.astype(np.float32),
        bounds_error=False,
        fill_value=np.nan,
    )
    return interp(np.column_stack([tgt_lats, tgt_lons]))
