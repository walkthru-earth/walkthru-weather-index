# Global DEM Strategy

Design rationale for the H3 GeoParquet terrain approach used by the pipeline.

---

## The Problem

Loading DEM on-the-fly via STAC (Copernicus GLO-30) works for regional bounding boxes but becomes the dominant bottleneck for global runs:

| Step | Regional (0.5 deg) | Global (180 x 360 deg) |
|---|---|---|
| Download NetCDF | ~2 min | ~2 min |
| **Load DEM via STAC** | **~5 sec** | **~20-40 min** |
| H3 grid (res 5) | ~1 sec | ~5 sec |
| GPU interpolation | ~2 sec | ~10-30 sec |
| Export to S3 | ~1 sec | ~30 sec |
| **Total** | **~2.5 min** | **~25-45 min** |

The STAC approach fetches hundreds of COG tiles, reprojects them, and computes terrain derivatives (slope, aspect, TRI, TPI) from scratch every run. For a twice-daily global pipeline, this is unacceptable.

---

## Solution: Pre-computed H3 GeoParquet

The pipeline reads pre-computed terrain derivatives from [walkthru-earth/dem-terrain](https://github.com/walkthru-earth/dem-terrain), hosted on [Source Cooperative](https://source.coop/walkthru-earth/dem-terrain) (public, no auth):

```
s3://us-west-2.opendata.source.coop/walkthru-earth/dem-terrain/
  h3_res=1/data.parquet
  h3_res=2/data.parquet
  ...
  h3_res=7/data.parquet
```

Each file contains `h3_index, elev, slope, aspect, tri, tpi` already at H3 cell centres.

**Runtime impact:**

| Step | Time |
|---|---|
| Download NetCDF | ~2 min |
| Load DEM from Parquet | ~5 sec |
| H3 grid | ~5 sec |
| GPU interpolation | ~10-30 sec |
| Export | ~30 sec |
| **Total** | **~3-5 min on A10G** |

**Cost: ~$0.10 per global run. Twice daily = ~$6/month.**

### Key benefits

- DEM load drops from ~30 min to **~5 seconds** (Parquet column scan)
- No STAC API calls, no COG fetching, no on-the-fly reprojection
- Same data reused every run -- compute terrain derivatives once, use forever
- Natural fit: pipeline already outputs H3, so DEM is H3 too
- When `dem["h3_native"] == True`, downstream code skips `RegularGridInterpolator` entirely

### Trade-offs

- One-time preprocessing cost (~1-2 hours on a large machine)
- Loses per-pixel terrain detail within each H3 cell (acceptable for weather downscaling where the source grid is 0.25 deg ~ 28 km)
- Need to host the Parquet files (a few GB total)

---

## Alternatives Evaluated

### Option B: One Big COG Machine

Pre-merge all Copernicus GLO-30 tiles into a single global COG (~350 GB at 30 m), load via windowed reads at runtime.

- 350 GB cannot fit in GPU VRAM (A100 = 80 GB) or most instance RAM
- Needs a CPU-heavy instance (256+ GB RAM) -- slower and more expensive
- This cost is paid **every single run**
- Estimated: ~$0.35-2.00 per run, ~30-45 min

### Option C: Distributed Bbox Tile Jobs

Split the globe into regional tiles (e.g., 30 x 30 deg), run parallel HF Jobs, merge outputs.

- More orchestration complexity (submit N jobs, monitor, handle failures)
- Edge effects at tile boundaries (H3 cells spanning two tiles need overlap)
- Total cost scales with number of tiles (72 tiles x ~$0.10 = ~$7.20)

### Comparison

| Criterion | H3 GeoParquet | Big COG Machine | Distributed Tiles |
|---|---|---|---|
| **Global run time** | ~5 min | ~30-45 min | ~5 min (parallel) |
| **Cost per run** | ~$0.10 | ~$0.35-2.00 | ~$7.20 |
| **Monthly cost (2x/day)** | ~$6 | ~$21-120 | ~$432 |
| **Machine needed** | A10G 24 GB | 256+ GB RAM | N x A10G |
| **Complexity** | Low | Low | High |

---

## DEM Source Dataset

The [walkthru-earth/dem-terrain](https://github.com/walkthru-earth/dem-terrain) project generates the H3 Parquet terrain data:

- **Source**: GEDTM-30m global DEM (OpenGeoHub)
- **Processing**: DuckDB 1.5 with native Parquet 2.11+ GEOMETRY support
- **Schema**: h3_index, geometry (native GEOMETRY), lat, lon, elev, slope, aspect, tri, tpi
- **Resolutions**: 1-7 uploaded and live. 8-10 in progress.
- **Hosting**: Source Cooperative (public, anonymous S3 access)

### Prior art

- FOSS4G 2025: [H3 Spatial Indexing for Global Raster Data](https://talks.osgeo.org/foss4g-2025/talk/TJTC3X/) demonstrated GEDTM30 > H3 res 12 > GeoParquet using GDAL + DuckDB
- GeoParquet partitioning studies show H3-based partitioning is competitive with hybrid approaches and significantly faster than no partitioning
- [GEDTM30 paper](https://peerj.com/articles/19673/): 30 m global DEM from ICESat-2 + GEDI + multisource fusion
