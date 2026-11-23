# RGBT-Tiny training launch evidence

These files record the frozen protocol, completed data preparation and successful
training preflight on November 23, 2026. They are launch evidence, not completed
training or evaluation results.

The first epoch completed with validation mAP50–95 of 0.0675; this is an early
training diagnostic, not a final result. One target-trained model is running,
followed automatically by nine development-only fixed/controller comparisons.
The fresh holdout remains unscored.

Protocol: [RGBT_TINY_TRAINING_PROTOCOL.md](../../../docs/RGBT_TINY_TRAINING_PROTOCOL.md).

Live local status:
`/media/imsel/4TBSSD/EMCH585_Artifacts/rgbt_tiny_yolo_20261123/status.json`.

Remote root on the training workstation:
`/mnt/12TB_Drive/whisenaj/EMCH585_training/rgbt_tiny_yolo_20261123`.

The local watcher archives study outputs after completion or failure. Designing
the proposed reliability-aware controller and its final holdout evaluation are
subsequent stages, not part of this automatic baseline pipeline.
