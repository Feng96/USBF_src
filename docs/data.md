# Data and output roles

`external_assets.csv` inventories names, roles, versions, byte counts, SHA-256 digests, cache locations, access requirements and missing-asset behavior. Downloads are pinned release artifacts; they do not query current OSM as a substitute for the archived reference.

| Directory | Role |
|---|---|
| `example/sensors/` | Real NAIP RGB, signed nDSM and separate RGB/height validity arrays |
| `example/targets.parquet` | Three original GBA objects, exact identifiers and original generation anchors |
| `example/neighbors.parquet` | Complete original neighbor context for those objects |
| `example/regression/` | Historical outputs used only for post-generation regression |
| `primary/` | Fixed 2,019 matches, original/pre-confirmation/final geometries, raw references, shifts and ROIs |
| `osm/` | Historical verified local OSM snapshot, 27,001 features in Los Angeles/Bay Area |
| `registration/` | Frozen real 1,024 m ground-registration image context and masks |
| `curved_reference/` | Separate 96-screening/13-target/9-match/7-registered supplementary records |
| `weights/` | Exact B0/B1 inference checkpoints |
| `full/` | Optional complete 32-frame generation inputs and their own checksum index |

RGB is stored as three floating bands in 0..1. Signed nDSM is in metres; valid measurements may be zero or negative. Height validity distinguishes missing observations from real zero. The model receives valid height clipped to 0..30 m and divided by 30, plus a separate validity channel. Physical geometry rules use signed metre values. All-missing height reproduces each model's own RGB pathway; this is checked with the real frozen weights.

The main example uses 2015 NAIP imagery `nv_m_3611554_ne_11_1_20150525_20150923`; the recorded contributing DSM/DTM project is `USGS_LPC_NV_LasVegas_QL1_2016_LAS_2018`. Original product identifiers, unsigned URLs, compound CRSs, vertical units, resampling and valid-pixel accounting are preserved in `example/upstream_sources.json`. This prepared snapshot is distinguished from downloading current source collections. The [NAIP catalog](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery) and [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) document the source products. SpaceNet references and image context originate from [SpaceNet-2 Las Vegas](https://spacenet.ai/spacenet-buildings-dataset-v2/).

GBA is the direct footprint inventory. OSM/Microsoft labels are inherited source information, not three independently fused inventories or certificates of truth independence. The archived OSM reference does not cover Las Vegas. Only an eligible exact inherited OSM type/ID can confirm a changed circle; missing, ambiguous or non-OSM evidence returns that candidate to the exact original GBA footprint.

The output's `gba_geometry` stores the original input. `pre_confirmation_geometry` stores the guarded proposal before OSM confirmation, and the active `geometry` stores the final footprint. `key`, `source` and `id` retain object identity. `modified` is actual geometry change; candidate flags are separate. Areas are computed in EPSG:5070 square metres. `height` inherits a supplied source height, or remains null if the historical example input lacks that attribute. `height_valid_fraction`, `height_missing` and `height_incomplete` describe raster evidence and do not replace inherited building height.

`usbf prepare --rgb image.tif --dsm dsm.tif --dtm dtm.tif --output sensor.npz` is an explicit convenience for new, north-up user rasters. The source RGB grid is retained, elevation must be metres, DSM/DTM are aligned horizontally, and their signed difference and validity are saved. Omit both elevation paths for the missing-height path. This is not advertised as byte-exact regeneration of the historical source-selection and acquisition process.

The small input is not national USBF, and its modification proportion must not be extrapolated nationally. Third-party source licenses and attribution remain applicable; the software license does not relicense GBA/OSM/SpaceNet-derived assets. The [GBA project's current license notice](https://github.com/zhu-xlab/GlobalBuildingAtlas/blob/main/README.md) distinguishes product components, so a single blanket license is not asserted for every upstream asset.
