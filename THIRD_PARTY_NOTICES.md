# Third-party notices

The geometric source subset includes:

- Geodan `building-boundary`, copyright 2018 Geodan, MIT; full notice in `licenses/building_boundary.txt`.
- DPIRD-DMA `Building-Regulariser`, copyright 2025 DPIRD-DMA, MIT; full notice in `licenses/buildingregulariser.txt`.

The original runtime imported only pure geometric submodules. The package retains the needed submodules and their nested initialization, with package-relative imports; optional top-level package initialization is omitted. Scientific rule extraction and source hashes are recorded separately.

PyTorch/torchvision, NumPy/SciPy, GDAL/rasterio, PROJ/pyproj, GEOS/Shapely, GeoPandas/pyogrio, OpenCV and scikit-image retain their own licenses. They are installed from pinned distributions rather than copied as unlicensed project code.

Input imagery, elevation, geometry and labels are distinct assets. Source terms and attribution for NAIP/3DEP, GBA and inherited OSM/Microsoft geometry, and SpaceNet remain applicable. The current GBA component-license notice is linked in `docs/data.md`; no blanket relicensing of mixed upstream data is implied by an original-code license. Author approval of the original-code and frozen-checkpoint release is pending, so no new project LICENSE is invented here.
