# Measured local environment

Tests used Linux x86_64, glibc 2.28, Python 3.11.5 and CPU-only Torch 2.5.1+cpu / torchvision 0.20.1+cpu. The installation began in a new virtual environment and installed actual pinned binary distributions. It was not copied from the author's existing scientific environment. The CPU lock records that measured environment; Windows, macOS and other Python versions have not passed the same release acceptance.

| Dependency | Observed version |
|---|---|
| NumPy / SciPy | 2.2.5 / 1.15.3 |
| rasterio / GDAL | 1.4.3 / 3.9.3 |
| pyproj / PROJ | 3.7.1 / 9.5.1 |
| Shapely / GEOS | 2.1.1 / 3.13.1 |
| GeoPandas / pyogrio | 1.1.1 / 0.11.0 |
| pandas / pyarrow | 2.2.3 / 19.0.1 |
| OpenCV / scikit-image | 4.11.0 / 0.25.2 |

The first complete clean application run, with two Torch threads, took 9.45 s for real inference, three footprint generations, GeoParquet readback and example geometry scoring. Peak process resident memory was 1,125,860 KiB (about 1.07 GiB); this excludes installation and operating-system cache. Full fixed-geometry scoring plus tables/figures took 34.81 s. The real missing-height control took 9.67 s, ground-image registration 6.57 s, one explicit training software-control update 5.08 s, and fresh notebook Run All 13.48 s. These are measurements, not speed guarantees. Use the versioned command/resource logs for later measurements.

The default public asset set is 625,150,026 bytes: two checkpoints plus a 23,878,248-byte example/evaluation archive. The optional 32-frame archive adds 81,341,643 bytes. Public archives include attribution and license notices; scientific member bytes are unchanged from the tested local snapshot. The software environment and its package downloads are additional. Plan for at least 4 GiB RAM and several GiB of free storage for installation, extracted inputs, cached archives and outputs; that planning allowance is not a measured minimum. No GPU is required. CUDA inference is exposed explicitly but this release's clean acceptance was run on CPU, not a newly tested CUDA environment.
