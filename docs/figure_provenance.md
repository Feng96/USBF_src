# Figure provenance

`usbf evaluate` produces new Figures 3/S1–S6 directly from the newly recalculated metric table, using fixed sample and group definitions. Both original and registered frames are retained. Empty categories remain n=0 and mean=NA; no positive point is plotted for an absent category.

The historical Figure 1 source is `manuscript_simple_map_20261025/make_map.py`; Figure 2 and manuscript model-role/selected-example assembly sources are `manuscript_model_roles_20261025/build.py` and `fix_flow.py`. Their exact historical source hashes are recorded in `manuscript_mapping.csv` and the release evidence manifest. These figures are not new experiments.

Figures 4–6 use the manuscript's archived object selections and image/geometry evidence. The exact six object IDs and selection roles are in `illustrated_example_registry.json`. Figures 4/5 were rendered by `build_data_paper_assets_20260911.py`; Figure 6 was revised by `make_figure6.py` to retain two local-boundary failure examples. The old third failure case was removed in that revision and is not silently reintroduced. `figure_source_registry.json` records the actual renderer and revision source hashes. Complete rendering of those historical figures has not been rerun here.

The three-object local preview is an independently named reproducibility illustration and does not replace those manuscript cases or imply additional validation observations. Run `usbf preview` to regenerate it from actual RGB and newly generated geometries. Table S8 is the separate curved-reference comparison; the extra classification-threshold table is named `CLASSIFICATION_THRESHOLD_SENSITIVITY.csv` and supports Figure S6.
