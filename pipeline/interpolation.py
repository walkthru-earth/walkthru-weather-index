"""Interpolation from regular source grids onto irregular target points.

GPU path  : CuPy map_coordinates (bilinear on GPU) -- O(N) per timestep.
CPU path  : scipy RegularGridInterpolator (bilinear + nearest fallback).

Both paths produce (T, N_targets) float32 output from (T, Ny, Nx) input.
"""

from __future__ import annotations

import logging

import numpy as np

from pipeline.gpu import free_gpu_memory, gpu_available

log = logging.getLogger(__name__)


def interpolate_to_points(
    data_3d: np.ndarray,
    src_lons: np.ndarray,
    src_lats: np.ndarray,
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
) -> np.ndarray:
    """Interpolate *data_3d* (T, Ny, Nx) onto target points.

    Returns ndarray of shape (T, N_targets) in float32.
    Uses GPU bilinear (map_coordinates) if available, otherwise CPU bilinear.
    """
    T, Ny, Nx = data_3d.shape
    N = len(tgt_lons)

    if gpu_available():
        log.info(
            "[INTERP] GPU bilinear: %d ts x (%d,%d) grid -> %s pts",
            T,
            Ny,
            Nx,
            f"{N:,}",
        )
        return _bilinear_gpu(data_3d, src_lons, src_lats, tgt_lons, tgt_lats)
    log.info(
        "[INTERP] CPU bilinear: %d ts x (%d,%d) grid -> %s pts",
        T,
        Ny,
        Nx,
        f"{N:,}",
    )
    return _bilinear_cpu(data_3d, src_lons, src_lats, tgt_lons, tgt_lats)


# -- GPU path: CuPy map_coordinates (bilinear) --------------------------------


def _bilinear_gpu(
    data_3d: np.ndarray,
    src_lons: np.ndarray,
    src_lats: np.ndarray,
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
) -> np.ndarray:
    """GPU-accelerated bilinear interpolation using map_coordinates.

    Converts target lat/lon to fractional grid indices, then uses CuPy's
    map_coordinates (order=1 = bilinear) to interpolate.  Handles longitude
    wrap-around by padding the grid circularly.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates

    T, Ny, Nx = data_3d.shape

    # Grid spacing
    dlat = float(src_lats[1] - src_lats[0]) if Ny > 1 else 1.0
    dlon = float(src_lons[1] - src_lons[0]) if Nx > 1 else 1.0
    lat0 = float(src_lats[0])
    lon0 = float(src_lons[0])

    # Normalize target longitudes to match source grid range
    # Source might be 0..360 or -180..180
    lon_max = float(src_lons[-1])
    lon_range = lon_max - lon0 + dlon  # full periodic range

    tgt_lon_norm = (tgt_lons - lon0) % lon_range + lon0

    # Convert to fractional grid indices
    lat_idx = (tgt_lats - lat0) / dlat  # shape (N,)
    lon_idx = (tgt_lon_norm - lon0) / dlon  # shape (N,)

    # Clamp latitude indices to valid range (no wrap at poles)
    lat_idx = np.clip(lat_idx, 0, Ny - 1)

    # Transfer coordinates to GPU once
    lat_idx_gpu = cp.asarray(lat_idx, dtype=cp.float32)
    lon_idx_gpu = cp.asarray(lon_idx, dtype=cp.float32)
    coords_gpu = cp.stack([lat_idx_gpu, lon_idx_gpu])  # (2, N)

    # Pad longitude axis by a few columns for wrap-around interpolation
    pad = 3
    out = np.empty((T, len(tgt_lons)), dtype=np.float32)

    for t in range(T):
        slab = data_3d[t].astype(np.float32)

        # Circular padding on longitude axis
        slab_padded = np.concatenate([slab[:, -pad:], slab, slab[:, :pad]], axis=1)
        slab_gpu = cp.asarray(slab_padded)

        # Shift longitude indices to account for padding
        coords_shifted = cp.stack([lat_idx_gpu, lon_idx_gpu + pad])

        # Bilinear interpolation (order=1), nearest at boundaries
        result = map_coordinates(slab_gpu, coords_shifted, order=1, mode="nearest")
        out[t] = result.get()

        del slab_gpu, coords_shifted, result

    del lat_idx_gpu, lon_idx_gpu, coords_gpu
    free_gpu_memory()

    # Fill any remaining NaN with nearest-neighbour (edge cases at poles)
    nan_count = np.isnan(out).sum()
    if nan_count > 0:
        log.warning(
            "[INTERP] %d NaN values after GPU bilinear, filling with nearest", nan_count
        )
        _fill_nan_nearest(out, src_lons, src_lats, tgt_lons, tgt_lats)

    log.info("[INTERP] GPU bilinear complete: %d timesteps", T)
    return out


# -- CPU fallback: bilinear interpolation --------------------------------------


def _bilinear_cpu(
    data_3d: np.ndarray,
    src_lons: np.ndarray,
    src_lats: np.ndarray,
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    T = data_3d.shape[0]
    tgt_pts = np.column_stack([tgt_lats, tgt_lons])
    out = np.empty((T, len(tgt_lons)), dtype=np.float32)

    for t in range(T):
        slab = data_3d[t].astype(np.float32)

        # Primary: bilinear interpolation
        interp_lin = RegularGridInterpolator(
            (src_lats, src_lons),
            slab,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        row = interp_lin(tgt_pts)

        # Fallback: nearest-neighbour for any remaining NaN
        # (handles target points that fall just outside the padded source grid)
        nan_mask = np.isnan(row)
        if nan_mask.any():
            interp_nn = RegularGridInterpolator(
                (src_lats, src_lons),
                slab,
                method="nearest",
                bounds_error=False,
                fill_value=None,
            )
            row[nan_mask] = interp_nn(tgt_pts[nan_mask])

        out[t] = row

    log.info("[INTERP] CPU bilinear complete: %d timesteps", T)
    return out


# -- helpers -------------------------------------------------------------------


def _fill_nan_nearest(
    out: np.ndarray,
    src_lons: np.ndarray,
    src_lats: np.ndarray,
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
) -> None:
    """In-place fill NaN values using nearest-neighbour on CPU."""
    from scipy.interpolate import RegularGridInterpolator

    tgt_pts = np.column_stack([tgt_lats, tgt_lons])
    T = out.shape[0]

    for t in range(T):
        nan_mask = np.isnan(out[t])
        if not nan_mask.any():
            continue
        interp_nn = RegularGridInterpolator(
            (src_lats, src_lons),
            out[t],
            method="nearest",
            bounds_error=False,
            fill_value=0.0,
        )
        out[t, nan_mask] = interp_nn(tgt_pts[nan_mask])
