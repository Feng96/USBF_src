# B0 vs B1 native detail + fourfold pilot

Matched epochs: 49. Independent GT: false. Per-domain shape-selected checkpoints; no production GBA edits.

|Arm|Domain|Epoch|IoU|Boundary F1 1m|Boundary p95 m|
|---|---|---:|---:|---:|---:|
|B0_control|naip_val|5|84.8940%|86.7608%|2.160|
|B0_control|ma_val|0|77.7168%|81.2723%|5.385|
|B1_shape|naip_val|49|84.9675%|87.0251%|2.160|
|B1_shape|ma_val|0|77.7168%|81.2723%|5.385|

Orientation supervision is derived from existing masks. Validation is development data, not proof of correcting GBA.
Metric spacing is tile-center geodesic dx/dy. Frame-guided polygon optimization and independent edit auditing are NOT implemented by this pilot.
Previews: previews/*.png; vectors/*.gpkg contain reference_mask, B0_control, B1_shape (not original GBA vector geometry).
