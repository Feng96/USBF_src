# Third-party notices

The geometric source subset includes:

- Geodan `building-boundary`, copyright 2018 Geodan, MIT; full notice in `licenses/building_boundary.txt`.
- DPIRD-DMA `Building-Regulariser`, copyright 2025 DPIRD-DMA, MIT; full notice in `licenses/buildingregulariser.txt`.

The original runtime imported only pure geometric submodules. The package retains the needed submodules and their nested initialization, with package-relative imports; optional top-level package initialization is omitted. Scientific rule extraction and source hashes are recorded separately.

PyTorch/torchvision, NumPy/SciPy, GDAL/rasterio, PROJ/pyproj, GEOS/Shapely, GeoPandas/pyogrio, OpenCV and scikit-image retain their own licenses. They are installed from pinned distributions rather than copied as unlicensed project code.

Original USBF code is licensed under MIT; the root `LICENSE` applies to original contributions. The two frozen model checkpoints are licensed under CC BY 4.0, as specified in `licenses/MODEL_WEIGHTS.txt`. Vendored packages retain the notices above.

Input imagery, elevation, geometry and labels are distinct assets. Source terms and attribution for NAIP/3DEP, GBA and inherited OSM/Microsoft geometry, and SpaceNet remain applicable. See `docs/asset_terms.md` and the `DATA_LICENSES.txt` supplied inside each data archive. GBA/OSM-origin footprint databases retain ODbL; SpaceNet reference labels and imagery retain CC BY-SA 4.0. The software and checkpoint licenses do not relicense these input datasets or the separate national product.
