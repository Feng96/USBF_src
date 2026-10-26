# Training and checkpoint provenance

Fixed-weight inference is the main reproduction route. `configs/training_paired.json` records the actual original training settings and selected checkpoints. Original training used 4,164 NAIP and 2,757 Massachusetts 1 m patches, with 231 NAIP and 75 Massachusetts validation patches. B0/B1 share the epoch65 parent with SHA-256 `cf96686c920a4bbe27b3ec00161f6d855f73a5a94f1da107fa2353040a0f48f8`. It is different from either selected inference checkpoint and is not silently substituted.

The paired recipe used batch eight, AdamW with weight decay 10⁻⁴, backbone/head learning rates 2×10⁻⁶/2×10⁻⁵ and new B1 head rate 10⁻⁴, cosine decay to 0.1 over 60 planned epochs, gradient clipping at five, BF16 training and FP32 validation. The seed is 20260909. Geometric augmentation rotates/flips all spatial arrays together; RGB perturbation is separate. The selected B0/B1 epochs are five and 49. Original selection used domain-specific boundary F1@1 m subject to an IoU guard. They are not an equal-training-budget causal ablation.

`configs/training_data_inventory.json` records all 7,227 original train/validation patch IDs, source-image groups, grids and recorded component hashes. No test records are included. This is provenance metadata, distinct from an executable normalized-NPZ training manifest; the original raw training assets are not bundled with the minimal example.

The local configurable update entry preserves the published architecture, losses, optimizer groups and explicit data validity:

```bash
usbf train --data-dir training-data --manifest training-data/manifest.csv \
  --parent training-data/weights/B0_parent_epoch65.pth \
  --arm B1_shape --device cuda --epochs 60 --max-updates 100 \
  --output outputs/training-run
```

The manifest has `name,path,domain,split,sha256`; paths are relative to `--data-dir`. NPZ patches require `rgb` (normalized three-band input), `height` (normalized model height), `valid`, `target` and `supervision_valid`. Heights and supervision have separate validity masks. The published full raw training corpus and warm-start parent are additional assets, outside the minimal download. Missing inputs fail explicitly. This finite update entry does not claim that it has replayed the complete historical domain-specific checkpoint-selection process. Complete retraining, all historical segmentation-validation inputs and the height-perturbation suite have not been rerun for this release.

A bounded software-control step can be run on the real local image:

```bash
usbf train --data-dir data --software-smoke --arm B1_shape \
  --output outputs/training-software-control.json
```

This uses original GBA rasterized on the real image solely to check loss, gradients and a real parameter update. It is explicitly not part of the published training corpus, not new accuracy evidence, and does not write to the frozen checkpoints. The actual CPU control passed. Synthetic tests of auxiliary contour targets remain separate from scientific evaluation.

`historical_training_evidence/` contains the original selected-checkpoint comparison and height-condition results with source hashes. Table S3 compares a different full20 clean/perturbed recipe (3,905/231 train/validation patches, batch four, FP32); it must not be relabeled as the selected paired B0/B1 experiment or mixed with SpaceNet footprint scores.
