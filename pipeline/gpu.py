"""GPU / CPU abstraction.

Import this module everywhere instead of importing cupy directly.
If CUDA is not available, all operations transparently fall back to numpy.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage

    _GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None  # type: ignore[assignment]
    cp_ndimage = None  # type: ignore[assignment]
    _GPU = False


def gpu_available() -> bool:
    return _GPU


def to_device(arr: np.ndarray, dtype=np.float32):
    """Send a numpy array to GPU (or keep on CPU if no GPU)."""
    arr = arr.astype(dtype)
    return cp.asarray(arr) if _GPU else arr


def to_numpy(arr) -> np.ndarray:
    """Pull an array back to numpy (no-op if already numpy)."""
    if _GPU and isinstance(arr, cp.ndarray):
        return arr.get()
    return np.asarray(arr)


def xp():
    """Return the active array module (cupy or numpy)."""
    return cp if _GPU else np


def xp_ndimage():
    """Return the active ndimage module."""
    if _GPU:
        return cp_ndimage
    from scipy import ndimage

    return ndimage


def free_gpu_memory():
    if _GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
