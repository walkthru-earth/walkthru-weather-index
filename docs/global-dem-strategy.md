# Global DEM Strategy

How to make the pipeline run globally without the DEM becoming a bottleneck.

---

## The Problem

The pipeline currently loads DEM on-the-fly via STAC (Copernicus GLO-30 from Planetary Computer). For a regional bbox this takes seconds. For a **global bbox** it becomes the dominant bottleneck:

| Step | Regional (0.5°) | Global (180° x 360°) |
|---|---|---|
| Download NetCDF | ~2 min (always 5.8 GB) | ~2 min (same file) |
| **Load DEM via STAC** | **~5 sec** | **~20-40 min** |
| H3 grid (res 5) | ~1 sec (10 cells) | ~5 sec (12K cells) |
| GPU interpolation | ~2 sec | ~2 min |
| Export to S3 | ~1 sec | ~30 sec |
| **Total** | **~2.5 min** | **~25-45 min** |

The STAC approach fetches hundreds of COG tiles, reprojects them, and computes terrain derivatives (slope, aspect, TRI, TPI) from scratch every single run. For a twice-daily global pipeline, this is unacceptable.

---

## Options Evaluated

### Option A: Pre-processed DEM as H3 GeoParquet (Recommended)

Pre-compute terrain derivatives for global coverage at the resolution the pipeline needs, store as H3-indexed GeoParquet, and read only the needed columns at runtime.

```
s3://bucket/dem-terrain/
  h3_res=3/
    part-00000.parquet
    # columns: h3_index, lat, lon, elev, slope, aspect, tri, tpi
```

**One-time preprocessing:**

1. Load GEDTM30 global DEM (COG tiles from OpenLandMap or Zenodo)
2. Resample to target resolution (~0.09° for a ~2000x2000 global grid)
3. Compute slope, aspect, TRI, TPI on the resampled grid
4. Assign each pixel to an H3 cell at the target resolution
5. Aggregate (mean elevation, mean slope, etc.) per H3 cell
6. Write as Parquet, partitioned by H3 parent resolution

**At pipeline runtime:**

1. Read Parquet (scan only needed columns)
2. DEM is already at H3 cell resolution -- no interpolation needed
3. Pass directly to topographic corrections

**Pros:**

- DEM load drops from ~30 min to **~5 seconds** (Parquet column scan)
- No STAC API calls, no COG fetching, no on-the-fly reprojection
- Same data reused every run -- compute terrain derivatives once, use forever
- Natural fit: pipeline already outputs H3, so DEM should be H3 too
- Proven pattern: FOSS4G 2025 demonstrated GEDTM30 -> H3 res 12 -> GeoParquet using GDAL + DuckDB

**Cons:**

- One-time preprocessing cost (~1-2 hours on a big machine)
- Need to host the Parquet files (~1-5 GB depending on resolution)
- Loses per-pixel terrain detail within each H3 cell (acceptable for weather downscaling where the weather grid is 0.25°)

**Estimated global run time after preprocessing:**

| Step | Time |
|---|---|
| Download NetCDF | ~2 min |
| Load DEM from Parquet | ~5 sec |
| H3 grid | ~5 sec |
| GPU interpolation | ~2 min |
| Export | ~30 sec |
| **Total** | **~5 min on A10G** |

**Cost: ~$0.10 per global run. Twice daily = ~$6/month.**

---

### Option B: One Big Machine with Pre-merged COG (~350 GB)

Use a pre-merged global COG file on a high-memory instance.

**How it works:**

1. Pre-merge all Copernicus GLO-30 tiles into a single global COG (~350 GB at 30 m)
2. At runtime, load via windowed reads and compute derivatives

**Pros:**

- Conceptually simple -- just point at the file
- Preserves full 30 m resolution
- No H3 pre-aggregation -- terrain derivatives at native resolution

**Cons:**

- 350 GB cannot fit in GPU VRAM (A100 = 80 GB) or most instance RAM
- Even with windowed reads, computing global slope/aspect requires reading neighbouring pixels across the full raster
- Needs a CPU-heavy instance (~256+ GB RAM) -- slower and more expensive
- This cost is paid **every single run** (twice daily)
- Mapterhorn (global terrain tile pipeline) also chunks by tiles rather than loading the full globe

**Estimated cost per run:**

| Machine | RAM | Time | Cost/run |
|---|---|---|---|
| Hetzner CCX63 (CPU) | 256 GB | ~45 min | ~$0.35 |
| A100 80 GB (GPU) | ~200 GB system | may OOM | ~$2.00 |
| A100 80 GB + 500 GB RAM instance (RunPod) | 500 GB | ~30 min | ~$1.50 |

---

### Option C: Distributed Bbox Tile Jobs

Split the globe into regional tiles (e.g., 30° x 30°), run parallel HF Jobs, merge Parquet outputs.

**How it works:**

1. Orchestrator job divides globe into N tiles (e.g., 72 tiles of 30° x 30°)
2. Each tile is submitted as a separate HF Job with `--bbox`
3. Each job fetches only its regional DEM via STAC (fast, ~5 sec each)
4. Parquet outputs land in the same S3 prefix and are union-readable

**Pros:**

- Each job is small and cheap (A10G is fine)
- Natural parallelism -- all tiles run concurrently
- No preprocessing needed
- Uses existing pipeline code unchanged

**Cons:**

- More orchestration complexity (submit N jobs, monitor, handle failures)
- Edge effects at tile boundaries (H3 cells spanning two tiles need overlap or deduplication)
- More HF job submissions, more points of failure
- Total cost scales with number of tiles (N x ~$0.10 = ~$7.20 for 72 tiles)
- STAC API rate limits could become a factor with 72 concurrent requests

---

## Comparison Matrix

| Criterion | A: H3 GeoParquet | B: Big COG Machine | C: Distributed Tiles |
|---|---|---|---|
| **Global run time** | ~5 min | ~30-45 min | ~5 min (parallel) |
| **Cost per run** | ~$0.10 | ~$0.35-2.00 | ~$7.20 |
| **Monthly cost (2x/day)** | ~$6 | ~$21-120 | ~$432 |
| **One-time setup** | 1-2 hours preprocessing | Merge COG (~2 hours) | Orchestrator code |
| **Machine needed** | A10G 24 GB | 256+ GB RAM | N x A10G |
| **Complexity** | Low (read Parquet) | Low (read COG) | High (orchestration) |
| **Resolution preserved** | H3 cell mean | Full 30 m | Full 30 m per tile |
| **Failure modes** | Single job | Single job | N jobs to monitor |
| **DEM freshness** | Reprocess when DEM updates | Reprocess when DEM updates | Always fresh from STAC |

---

## Prior Art and References

### FOSS4G 2025: H3 Spatial Indexing for Global Raster Data

[Talk link](https://talks.osgeo.org/foss4g-2025/talk/TJTC3X/)

Demonstrated converting GEDTM30 (20+ GB global DEM) to H3 hexagonal grids at resolution 12 using GDAL, DuckDB, and GeoParquet. Key findings:

- O(1) cell data retrieval via H3 hash lookup
- Eliminates projection boundary issues
- Consistent global coverage with hierarchical multi-scale analysis
- Pipeline: raster -> H3 hashes Parquet -> DuckDB processing -> GeoParquet storage

### GEDTM30: Global Ensemble Digital Terrain Model

[GitHub](https://github.com/openlandmap/GEDTM30) | [Paper (PeerJ)](https://peerj.com/articles/19673/) | [Zenodo](https://zenodo.org/records/15689805)

- 30 m global DEM built from ICESat-2 + GEDI + multisource fusion
- 15 pre-computed terrain parameters at 6 scales (30-960 m)
- Available as Cloud-Optimized GeoTIFF via OpenLandMap STAC
- Uses 5° x 5° tiling with Equi7 grid system for processing
- Total dataset: ~80 TB across all versions and derivatives

### Mapterhorn: Global Terrain Tile Pipeline

[GitHub](https://github.com/mapterhorn/mapterhorn/tree/main/pipelines)

- Transforms 130+ DEM sources into Terrain RGB PMTiles for web maps
- Key architectural patterns relevant to our use case:
  - **Does NOT load the full globe at once** -- processes in Z12 macrotile chunks
  - Memory bounded: max 32,768 px wide tiles (~4 GB uncompressed)
  - Merging requires ~20 GB RAM per thread
  - Incremental/dirty processing: only reprocesses changed data
  - Does NOT use Parquet -- outputs PMTiles (visual tiles, not analytics)
  - Does NOT pre-compute slope/aspect -- expects client-side derivation

### GeoParquet Partitioning

[Optimal GeoParquet Partitioning Strategy](https://medium.com/center-for-coastal-climate-resilience-visualizatio/optimal-geoparquet-partitioning-strategy-33331874ef6c)

- H3 partitioning (6.48 s) performs close to hybrid approaches (5.36 s)
- Both significantly faster than attribute-only (12.2 s) or no partitioning (32.6 s)
- Parquet 2.11+ supports native GEOMETRY/GEOGRAPHY types

### OpenLandMap STAC

[Collection](https://stac.openlandmap.org/gedtm-30m/collection.json)

- GEDTM30 available via STAC catalog as COG tiles
- Current pipeline fallback source (after Planetary Computer)
- Global extent: -180 to 180 lon, -65 to 85 lat

---

## Recommendation

**Option A (H3 GeoParquet)** is the clear winner for a twice-daily global pipeline:

1. **Lowest marginal cost** -- ~$0.10/run vs $0.35-2.00 (Option B) or $7.20 (Option C)
2. **Fastest runtime** -- ~5 min vs 30-45 min (Option B)
3. **Simplest operations** -- single job, no orchestration, no high-memory instances
4. **Natural H3 alignment** -- DEM data arrives pre-indexed to match the output grid

The trade-off (losing per-pixel terrain detail within each H3 cell) is acceptable because:

- The weather source data is 0.25° (~28 km) resolution
- H3 res 5 cells are ~253 km² -- terrain detail within a cell averages out
- Even H3 res 7 cells (~5 km²) are far larger than 30 m DEM pixels
- The topographic corrections use cell-mean elevation, slope, and aspect -- not pixel-level detail

### Next Steps

1. Build a one-time DEM preprocessing script using DuckDB + GDAL
2. Process GEDTM30 COG tiles -> compute derivatives -> aggregate to H3 -> write GeoParquet
3. Host on S3 alongside the weather output
4. Update `pipeline/dem.py` to read from Parquet when available, fall back to STAC for regional bbox
