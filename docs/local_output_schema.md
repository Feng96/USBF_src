# Local example and national product attributes

The minimal example exports one row per original input ID, with valid geometries and explicit CRS metadata. It is not a national product exporter. Table A1 describes the national release schema; the small generation adapter uses the following explicit correspondences.

| Local example field | Meaning / national correspondence |
|---|---|
| `key`, `source`, `id` | Stable original sample identity and inherited source. `key` is not presented as a newly reconstructed national `usbf_id`. |
| `geometry` | Final EPSG:4326 geometry. |
| `gba_geometry` | Exact original geometry, corresponding to the national `original_geometry` role. |
| `pre_confirmation_geometry` | Guarded candidate before OSM circle confirmation, retained for audit. |
| `modified`, `modification_type` | Actual final change and broad local category; detailed accepted decision and added/removed areas remain available. The national category vocabulary is not silently substituted. |
| `added_area_m2`, `removed_area_m2` | Polygon differences in EPSG:5070 square metres. |
| `decision`, `processing_status` | Recorded local decision and outcome. |
| `circle_candidate`, `circle_proposed_before_gate`, `circle_gate_status`, `circle_gate_reason` | Candidate, confirmation and fallback information, separate from actual final change. |
| `holes_before`, `holes_after` | Actual ring counts. |
| `height`, `height_inherited` | Supplied GBA height, if present; absent input height remains null. |
| `height_valid_fraction`, `height_missing`, `height_incomplete` | Evidence availability on the local original-footprint analysis raster, independent of inherited building height. |
| `science_version`, `circle_gate_policy` | Frozen manuscript science and exact-ID confirmation policy. |

The source-frame anchor is retained in the input attributes. Full inference/frame associations and overlap-guard reversions are also recorded in `GENERATION.json`, `AUDIT.json` and `JOINT_GUARDS.json`. The example does not invent missing national region identifiers, source-height values, original WKB hashes or release-ledger metadata merely to populate a national schema. The output keeps original, candidate and final WKB through GeoParquet save/readback checks.
