"""Bilinear interpolation with GPU-accelerated Gaussian kernel smoothing fallback.

GPU path  : Nadaraya-Watson kernel regression with Gaussian kernel.
CPU path  : scipy RegularGridInterpolator (bilinear + nearest-neighbour fallback).

Both paths produce (T, N_targets) float32 output from (T, Ny, Nx) input.

The GPU path uses a Gaussian kernel with cos(lat)-corrected longitude distances
to avoid anisotropy in degree-space. The shape parameter eps defaults to the
source grid spacing (0.25° for NOAA AI-NWP models) following standard
bandwidth-selection practice for Nadaraya-Watson regression.
"""

from __future__ import annotations

import logging

import numpy as np

from pipeline.config import GPU_CHUNK_SIZE, PRECISION
from pipeline.gpu import free_gpu_memory, gpu_available, to_numpy, xp

log = logging.getLogger(__name__)


def interpolate_to_points(
    data_3d: np.ndarray,
    src_lons: np.ndarray,
    src_lats: np.ndarray,
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
    eps: float | None = None,
    chunk: int = GPU_CHUNK_SIZE,
) -> np.ndarray:
    """Interpolate *data_3d* (T, Ny, Nx) onto target points.

    Returns ndarray of shape (T, N_targets) in float32.
    Uses GPU Gaussian kernel smoothing if available, otherwise bilinear on CPU.
    """
    if eps is None:
        # Default epsilon = source grid spacing (0.25° for NOAA AI-NWP)
        eps = float(abs(src_lons[1] - src_lons[0])) if len(src_lons) > 1 else 0.25

    T, Ny, Nx = data_3d.shape
    N = len(tgt_lons)

    if gpu_available():
        log.info(
            "[INTERP] GPU kernel smoothing: %d ts x (%d,%d) grid -> %s pts (eps=%.3f, chunk=%d)",
            T,
            Ny,
            Nx,
            f"{N:,}",
            eps,
            chunk,
        )
        return _kernel_smooth_gpu(
            data_3d, src_lons, src_lats, tgt_lons, tgt_lats, eps, chunk
        )
    log.info(
        "[INTERP] CPU bilinear: %d ts x (%d,%d) grid -> %s pts",
        T,
        Ny,
        Nx,
        f"{N:,}",
    )
    return _bilinear_cpu(data_3d, src_lons, src_lats, tgt_lons, tgt_lats)


# ── GPU path: Nadaraya-Watson kernel regression ─────────────────────────────


def _kernel_smooth_gpu(
    data_3d: np.ndarray,
    src_lons: np.ndarray,
    src_lats: np.ndarray,
    tgt_lons: np.ndarray,
    tgt_lats: np.ndarray,
    eps: float,
    chunk: int,
) -> np.ndarray:
    _xp = xp()
    dtype = getattr(_xp, PRECISION)

    T, Ny, Nx = data_3d.shape
    S = Ny * Nx
    N = len(tgt_lons)

    # cos(lat) correction for longitude distances to remove anisotropy
    lat_center = float((src_lats.max() + src_lats.min()) / 2)
    cos_lat = float(np.cos(np.radians(lat_center)))

    grid_lon, grid_lat = np.meshgrid(src_lons, src_lats)
    # Store as (lon_corrected, lat) pairs
    src_coords = _xp.asarray(
        np.stack([grid_lon.ravel() * cos_lat, grid_lat.ravel()], axis=1), dtype=dtype
    )

    data_flat = _xp.asarray(data_3d.reshape(T, S), dtype=dtype)
    out = _xp.zeros((T, N), dtype=dtype)

    import cupy as cp

    n_chunks = (N + chunk - 1) // chunk
    for ci, i in enumerate(range(0, N, chunk)):
        j = min(i + chunk, N)
        tgt_chunk = _xp.asarray(
            np.column_stack([tgt_lons[i:j] * cos_lat, tgt_lats[i:j]]), dtype=dtype
        )

        diff = tgt_chunk[:, None, :] - src_coords[None, :, :]
        distances = _xp.sqrt(_xp.sum(diff**2, axis=2))

        weights = _xp.exp(-((distances / eps) ** 2))
        w_sum = _xp.maximum(_xp.sum(weights, axis=1, keepdims=True), 1e-10)
        weights = weights / w_sum

        out[:, i:j] = _xp.matmul(data_flat, weights.T)

        del diff, distances, weights, w_sum, tgt_chunk
        cp.cuda.Stream.null.synchronize()

        if (ci + 1) % 500 == 0 or ci + 1 == n_chunks:
            log.info(
                "[INTERP] GPU chunk %d/%d (%.0f%%)",
                ci + 1,
                n_chunks,
                100 * (ci + 1) / n_chunks,
            )

    result = to_numpy(out)
    del data_flat, src_coords, out
    free_gpu_memory()
    return result


# ── CPU fallback: bilinear interpolation ─────────────────────────────────────


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

    return out
