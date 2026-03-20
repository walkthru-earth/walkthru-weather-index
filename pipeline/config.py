"""Central configuration — edit this file to change location, resolutions, models."""

from pathlib import Path

FORECAST_DAYS = 5

# Default bounding box: global
BBOX = {
    "min_lat": -90.0,
    "max_lat": 90.0,
    "min_lon": -180.0,
    "max_lon": 180.0,
}

# Extra margin added when loading weather source data.
# Must be large enough so every H3 cell centre (at any configured resolution)
# is fully surrounded by source grid points — prevents NaN at the BBOX edges.
# Rule of thumb: ≥ (H3_cell_diameter / 2) + source_grid_resolution
# GraphCast/FourCastNet are 0.25°; H3 res-5 cell diameter ≈ 2.5° → use 2.0°
WEATHER_PADDING: float = 2.0

# ── H3 resolutions ────────────────────────────────────────────────────────────
# Provide one or more; each resolution produces a separate parquet partition.
# res 5 ≈ 253 km² (~2M cells globally)
# res 7 ≈ 5 km²  (~99M cells globally) — needs A10G-large (46 GB RAM)
H3_RESOLUTIONS: list[int] = [8]

# ── Pre-computed DEM terrain (H3-indexed Parquet) ────────────────────────────
# Source Cooperative public bucket — no credentials needed.
# Each resolution has a single data.parquet file sorted by h3_index.
DEM_PARQUET_BASE = "s3://us-west-2.opendata.source.coop/walkthru-earth/dem-terrain"
DEM_PARQUET_REGION = "us-west-2"
# Resolutions available as pre-computed Parquet (v2: all 1-10).
DEM_PARQUET_MAX_RES = 10

# ── NOAA AI-NWP S3 bucket ─────────────────────────────────────────────────────
S3_BUCKET = "noaa-oar-mlwp-data"

AI_MODELS: dict[str, dict] = {
    "GraphCast_GFS": {
        "code": "GRAP_v100_GFS",
        "interval_hours": 6,
        # Variables NOT available or unreliable for this model (empty = all available)
        "skip_vars": [],
    },
    "GraphCast_IFS": {
        "code": "GRAP_v100_IFS",
        "interval_hours": 6,
        "skip_vars": [],
    },
    "FourCastNet": {
        "code": "FOUR_v200_GFS",
        "interval_hours": 6,
        # FourCastNet outputs relative humidity (not specific humidity),
        # does not output accumulated precipitation or vertical velocity.
        "skip_vars": ["q", "apcp", "w"],
    },
    "Pangu_Weather": {
        "code": "PANG_v100_GFS",
        "interval_hours": 6,
        # Pangu does not output accumulated precipitation or vertical velocity.
        "skip_vars": ["apcp", "w"],
    },
}

# ── Topographic correction parameters ────────────────────────────────────────
TOPO_PARAMS = {
    "temp_lapse_rate": -6.5,  # °C per 1000 m (default; overridden by variable lapse rate when available)
    "wind_height_factor": 0.3,  # fractional increase per 1000 m
}

# ── Geopotential anomaly reference ──────────────────────────────────────────
# ICAO Standard Atmosphere: 500 hPa geopotential height ≈ 5574 m.
# Used as a sensible global default for anomaly computation.
GEOPOTENTIAL_500_REF: float = 5574.0

# ── Local dirs (used when running outside HF Jobs) ────────────────────────────
CACHE_DIR = Path("weather_cache")
OUTPUT_DIR = Path("output")
DEM_CACHE = Path("dem_cache")


def ensure_dirs() -> None:
    """Create local cache/output directories (call from main, not at import time)."""
    for d in (CACHE_DIR, OUTPUT_DIR, DEM_CACHE):
        d.mkdir(parents=True, exist_ok=True)
