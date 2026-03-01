"""DEM loading with full GPU processing.

Primary source  : Copernicus GLO-30 via Microsoft Planetary Computer STAC
Fallback source : openlandmap merged 30 m DEM (stac.openlandmap.org)
"""

from __future__ import annotations

import numpy as np

from pipeline.config import BBOX
from pipeline.gpu import to_device, to_numpy, xp, xp_ndimage


def load_dem(bbox: dict = BBOX, resolution: float | None = None) -> dict:
    """Load DEM and compute terrain derivatives on GPU (or CPU).

    When *resolution* is None it is auto-computed to target ~2000×2000 pixels.

    Returns a dict with keys: elev, lat, lon, slope, aspect, tri, tpi, dx, dy.
    All array values are numpy (CPU) arrays — GPU tensors are only used
    transiently during derivative computation.
    """

    # Auto-scale resolution to target ~2000×2000 px
    if resolution is None:
        lat_span = bbox["max_lat"] - bbox["min_lat"]
        lon_span = bbox["max_lon"] - bbox["min_lon"]
        resolution = max(lat_span, lon_span) / 2000
        resolution = max(resolution, 0.0003)  # floor at ~30 m

    # DEM bbox: add a small buffer for edge interpolation,
    # but skip for global bbox (edges wrap at ±180/±90).
    is_global = (bbox["max_lat"] - bbox["min_lat"]) >= 170
    buffer = 0.0 if is_global else 0.05
    dem_bbox = {
        "min_lat": max(bbox["min_lat"] - buffer, -90.0),
        "max_lat": min(bbox["max_lat"] + buffer, 90.0),
        "min_lon": max(bbox["min_lon"] - buffer, -180.0),
        "max_lon": min(bbox["max_lon"] + buffer, 180.0),
    }

    print("🏔️  Loading DEM …")

    dem_ds = _load_from_planetary_computer(dem_bbox, resolution)

    if dem_ds is None:
        print("   ⚠️  Planetary Computer unavailable, trying openlandmap …")
        dem_ds = _load_from_openlandmap(dem_bbox, resolution)

    dem = dem_ds["data"].squeeze()

    lons = dem.longitude.values if "longitude" in dem.coords else dem.x.values
    lats = dem.latitude.values if "latitude" in dem.coords else dem.y.values

    # Ensure latitude is ascending (south→north) so np.gradient dz/dy
    # points northward, which the aspect formula requires.
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        elev_np = dem.values[::-1]
    else:
        elev_np = dem.values

    elev_gpu = to_device(elev_np)

    _xp = xp()
    _ndi = xp_ndimage()

    lat0 = (lats.max() + lats.min()) / 2
    dx = abs(lons[1] - lons[0]) * 111_320 * np.cos(np.radians(lat0))
    dy = abs(lats[1] - lats[0]) * 111_320

    dz_dy, dz_dx = _xp.gradient(elev_gpu, dy, dx)
    slope = _xp.arctan(_xp.sqrt(dz_dx**2 + dz_dy**2)) * 180 / _xp.pi
    aspect = (_xp.arctan2(-dz_dx, -dz_dy) * 180 / _xp.pi + 360) % 360

    # TRI: Riley et al. (1999) — RMS of elevation differences between
    # center cell and each neighbor in a 3×3 window.
    # Compute sum of squared differences via convolution identity:
    #   Σ(z_i - z_c)² = Σz_i² - 2·z_c·Σz_i + 9·z_c²
    #                 = conv(z², ones) - 2·z·conv(z, ones) + 9·z²
    kernel_ones = _xp.ones((3, 3), dtype=_xp.float32)
    sum_z2 = _ndi.convolve(elev_gpu**2, kernel_ones, mode="nearest")
    sum_z = _ndi.convolve(elev_gpu, kernel_ones, mode="nearest")
    # 9 neighbors including center; subtract center contribution for pure Riley
    # But Riley (1999) includes the center in the sum, so keep all 9 terms
    tri = _xp.sqrt(_xp.maximum(sum_z2 - 2 * elev_gpu * sum_z + 9 * elev_gpu**2, 0) / 9)

    tpi = elev_gpu - _ndi.uniform_filter(elev_gpu, size=5, mode="nearest")

    print(
        f"   ✅ DEM {elev_gpu.shape}  "
        f"elev {float(_xp.min(elev_gpu)):.0f}–{float(_xp.max(elev_gpu)):.0f} m  "
        f"slope max {float(_xp.max(slope)):.1f}°"
    )

    return {
        "elev": to_numpy(elev_gpu),
        "lat": lats,
        "lon": lons,
        "slope": to_numpy(slope),
        "aspect": to_numpy(aspect),
        "tri": to_numpy(tri),
        "tpi": to_numpy(tpi),
        "dx": dx,
        "dy": dy,
    }


# ── private helpers ───────────────────────────────────────────────────────────


def _load_from_planetary_computer(bbox: dict, resolution: float):
    try:
        import odc.stac
        import pystac_client
        import planetary_computer

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        items = list(
            catalog.search(
                collections=["cop-dem-glo-30"],
                bbox=[
                    bbox["min_lon"],
                    bbox["min_lat"],
                    bbox["max_lon"],
                    bbox["max_lat"],
                ],
            ).items()
        )

        return odc.stac.load(
            items,
            bands=["data"],
            crs="EPSG:4326",
            resolution=resolution,
            resampling="bilinear",
            bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
        )
    except Exception as e:
        print(f"   ⚠️  Planetary Computer error: {e}")
        return None


def _load_from_openlandmap(bbox: dict, resolution: float):
    """Fallback: OpenLandMap merged 30 m DEM via STAC."""
    import odc.stac
    import pystac_client

    catalog = pystac_client.Client.open("https://stac.openlandmap.org")
    items = list(
        catalog.search(
            collections=["gedtm-30m"],
            bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
        ).items()
    )

    if not items:
        raise RuntimeError("No DEM tiles found in openlandmap either.")

    return odc.stac.load(
        items,
        bands=["data"],
        crs="EPSG:4326",
        resolution=resolution,
        resampling="bilinear",
        bbox=[bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]],
    )
