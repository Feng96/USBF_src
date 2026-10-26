# Historical height experiment

Recorded evidence, not a newly executed release experiment. The original launch note is identified by SHA-256 in PROVENANCE.json; scheduler, process and private path details are omitted here.

The two full20 recipes each used 3,905 training and 231 validation patches, batch four, FP32, 20 epochs and 19,540 updates, seed 20260906. AdamW used learning rate 0.0002 (backbone at 0.1 times that rate) and weight decay 0.0001. Initialization used the same seeded decoder and ImageNet backbone. Checkpoint selection used normal-condition masked validation IoU at threshold 0.5; earliest wins ties.

B0 uses validity-gated fusion without height perturbations; C combines boundary supervision and random height perturbations. This is a complete-recipe comparison. Seven evaluation conditions are none, missing, block_missing, bias, spike, blur and shift. Historical comparison counts and scores are in height_comparison.csv. Pixel boundary F1 here is the historical one-pixel L-infinity metric, separate from the SpaceNet 0.5 m grid and one-metre Euclidean metric.

Complete raw validation inputs, selected full20 checkpoints and the perturbation rerun are outside the minimal fixed B0/B1 inference download. The published scores in this file are not claimed as a fresh geometric or model rerun.
