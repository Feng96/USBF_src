# Reproduction protocol

The fixed manuscript science is `GBA_V6circle_V7_reference_night_v003_20260911` with observed-curvature candidate search and `V7_OSM_circle_confirmation_v1_20260918`. Local generation uses the original rule definitions extracted into importable modules; source hashes and symbol hashes are listed in `source_mapping.csv`. Input/output adapters and package imports are separated from scientific functions.

The two checkpoints are validated before deserialization:

| Model | Selected epoch | Updates | SHA-256 |
|---|---:|---:|---|
| B0 | 5 | 4,330 | `1912c45a663ab95962449960791396eacfc2fa9f5236b0abcce6bb5716c71238` |
| B1 | 49 | 42,434 | `65b86df2d146f4074eea700cd6cd8ff2d3839e9a837c3dc0d947211b6e87bcc9` |

Different selected epochs are disclosed; comparing these checkpoints is not a strict equal-budget architectural ablation. B1's edge and fourfold-direction outputs feed a learned mask residual. They are not a third segmentation model or a replacement with a complete frame-field polygonizer.

The example is fixed before its release scoring. Among historical contexts containing a changed noncircular building, an unchanged building, and a reverted circle candidate, the hash-first anchor under `usbf-paper-release-20261025:` is selected. A hash-first key is then chosen within each role. No IoU/F1 value selects an example. The resulting context is `Vegas_SN2_1226`, a 512 m square on a 1 m UTM11 grid. All 8,397 original context-neighbor geometries are retained for the example's spatial guard.

The generation chain is input schema/hash checking → actual B0/B1 forward passes → B1 contour candidates → frozen local geometry/evidence rules → simultaneous/original-neighbor guards → exact-ID OSM circle confirmation → GeoParquet export and readback → scoring against fixed references. The `--historical-probabilities` option is explicitly regression-only and is never counted as actual model end-to-end validation.

Evaluation preserves both coordinate frames. Only reference polygons receive the archived UTM11 translation. Neither GBA nor USBF moves. No footprint-to-footprint registration, rematching on refined polygons, or score-based shift selection occurs. Per-object IoU and boundary F1 are recalculated with the original world-anchored 0.5 m grid and the original fixed ROI. Boundary tolerances are 1 m and 2 m; arithmetic building means are primary. The stability band is ±0.01 metric units, one percentage point. Bootstrap summaries use 5,000 source-image cluster draws, seed 20261024, retaining six parent images. A singleton source-image group has no estimated interval.

The raw-reference comparison remains negative for modified objects. In the registered-reference frame, the regenerated modified-group means are IoU 0.6973765936048105 → 0.7250759348180107 and boundary F1@1 m 0.5440859123332865 → 0.5993448188467065. Their paired changes are +2.7699341213200275 and +5.525890651342003 percentage points. The corresponding percentile intervals are [1.1024549625565354, 5.181428632271803] and [−0.4297466961952849, 11.084904449702625] percentage points. These numbers are regression targets verified after geometry scoring, not values copied into the calculated result table.

The clean CPU example produces exactly the historical final WKB for all three objects. Compared with historical GPU probabilities, maximum absolute probability difference is approximately 3.16×10⁻⁶, with no threshold crossing at 0.25, 0.35, 0.45, 0.50, or 0.65 in this frame. Geometry tolerances were not loosened. Other hardware or library versions must be evaluated and reported rather than assumed equivalent.

The supplementary reference-selected cohort is handled independently. It retains 96 screening records, 13 fixed targets, nine frozen GBA matches and seven accepted registered matches. Missing matches and unavailable registered scores remain `NA`. The primary sample's nine reverted circle candidates and these nine shape-selected matches have different selection definitions and must not be merged or treated as the same sample.

The real registration command reruns SIFT/NCC fitting and spatial holdout checks on a frozen actual image context and its archived ground mask. It does not claim to rebuild the ground mask from newly downloaded NIR and height products. Source IDs/unsigned URLs and archived preprocessing parameters distinguish the prepared snapshot from a new live-data run.

The historical local acceptance levels and their completed substeps are preserved in `release_validation.json`; its pending-publication entries describe the pre-release run, not the current repository availability. The public-release workflow separately checks a fresh GitHub checkout, a new dependency installation, anonymous downloads, real model execution, geometric regression, complete fixed-geometry evaluation and a fresh notebook kernel. Its logs and generated artifacts are available through the repository's Actions page. No DOI has been assigned to this software release.
