# USBF: local manuscript reproduction

Generate building footprints locally from real RGB imagery, signed nDSM, original GBA instances, and the two frozen manuscript models. The example includes a changed outline, an unchanged building, and a circle candidate that returns to GBA because exact OSM confirmation is unavailable.

The manuscript configuration retains circle/ellipse candidate detection and the September 18 OSM confirmation rule. B1 generates candidate outlines; B0 and B1 probabilities jointly assess local changes. Building identifiers and supplied GBA height attributes are retained.

This release candidate has passed local clean-environment validation. Public repository, asset URLs, and author licenses are awaiting confirmation; `fetch` will not silently substitute other data or weights. See [the actual validation report](docs/release_validation.json).

## Quick start

Use Linux x86_64 and Python 3.11. A CPU is sufficient. After obtaining and unpacking this source package:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cpu-linux-py311.lock
python -m pip install --no-deps .
usbf doctor
usbf fetch --data-dir data
usbf example --data-dir data --output outputs/example --device cpu --threads 2
usbf regression --data-dir data --product outputs/example/USBF.parquet --output outputs/regression.json
usbf preview --data-dir data --product outputs/example/USBF.parquet --output outputs/preview.png
```

The initial download contains two approximately 300 MB checkpoints and a small real-data/evaluation snapshot. After downloading, the core example works offline:

```bash
usbf fetch --offline --data-dir data
usbf example --data-dir data --output outputs/offline-example --device cpu
```

For a privately supplied review bundle before public publication, put the three staged files in the repository root and prepare the same cache explicitly:

```bash
mkdir -p data/weights
tar -xzf usbf-manuscript-inputs-20261025.tar.gz -C data
cp B0_control_best_shape_naip_val.pth B1_shape_best_shape_naip_val.pth data/weights/
usbf fetch --offline --data-dir data
usbf example --data-dir data --output outputs/local-review --device cpu --threads 2
```

This local-file route does not count as validation of a public download endpoint. [Measured versions and resources](docs/measured_environment.md) and [the local-versus-national output schema](docs/local_output_schema.md) describe the tested scope.

Use a new output directory for each generation. `outputs/example/USBF.parquet` is a readable GeoParquet containing final, original, and pre-confirmation geometries, stable input IDs, accepted-change status, areas, source labels, and elevation-coverage information. `AUDIT.json`, `GENERATION.json`, `INFERENCE.json`, and `evaluation/` document actual execution. Original height values absent from an input remain null; nDSM statistics do not replace them.

Measured in a fresh Python 3.11 CPU environment, the real one-frame/three-building generation and example scoring took 9.45 seconds with two Torch threads, after installation and data acquisition. Runtime depends on hardware and cache state. Exact resource measurements and dependency versions accompany the validation logs. A dedicated GPU is not required for this example; `--device cuda` is an explicit optional path, not a silently selected backend.

## Recompute the paper evaluation

```bash
usbf evaluate --data-dir data --output outputs/paper
usbf register --data-dir data --output outputs/ground-registration.json
```

`evaluate` reads frozen footprint and reference geometries plus fixed matches and shifts, re-rasterizes the boundaries, and recomputes polygon IoU and boundary F1. It produces Table 3, S2 Tables S4–S9, Figures 3/S1–S6, per-object scores, and separate curved-reference results. Existing score CSVs are used only afterward to check regression, never as the computed scores.

The primary sample remains 2,019 pairs: 171 changed and 1,848 unchanged. Accepted-operation groups contain 144 additions only, 20 removals only, seven combined changes, and zero accepted circle/ellipse replacements. Both original-coordinate and NAIP-registered reference comparisons are reported. The separate curved-reference sample has nine raw-coordinate pairs, seven registered pairs, and four unmatched targets with `NA` scores; it does not enlarge the primary denominator.

For optional full-input generation:

```bash
usbf fetch --full-inputs --data-dir data
usbf generate --dataset full --data-dir data --output outputs/full-generation --device cpu
usbf evaluate --data-dir data --generated-product outputs/full-generation/USBF.parquet --output outputs/full-generation-evaluation
```

The optional manifest includes all 32 historical sensor frames and original generation targets. Full-image generation is provided as an executable route and input inventory; it has **not** been run as part of the current acceptance. Only small-example generation and full fixed-geometry evaluation replay are verified. Do not describe the latter as reconstructing the complete paper from imagery.

## Notebook and tests

```bash
python scripts/run_notebook.py notebooks/USBF_local_example.ipynb --output outputs/executed-example.ipynb
python -m pytest -q
```

The notebook starts a fresh kernel, performs real model inference and geometry evaluation, and displays the newly generated footprints. Its default data directory is `data/` at the repository root. `.[test]` dependencies are included in the tested CPU lock file. Contract tests include clearly labeled synthetic controls for circle confirmation, holes, missing elevation, invalid inputs, absent weights, unmatched references, and geometry readback. These are software tests, not additional accuracy samples.

See [reproduction details](docs/reproduction.md), [training scope](docs/training.md), [data roles](docs/data.md), [scientific scope](docs/scientific_scope.md), [source mapping](docs/source_mapping.csv), and [manuscript mapping](docs/manuscript_mapping.csv). Dataset access remains at [Source Cooperative](https://source.coop/opengeos/us-buildings); this example does not independently certify national release counts or coverage.
