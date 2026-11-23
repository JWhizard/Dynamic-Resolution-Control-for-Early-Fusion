# RGBT-Tiny training and development protocol

This first stage fine-tunes one four-channel YOLO26n and measures fixed-resolution
and existing-controller development baselines. It reserves an unscored holdout
for the later reliability-aware controller study. It does not yet establish a
strong detector, implement the new controller or provide final test results.

## Data and conservative groups

Only the original visible-coordinate official training annotations supply the
new partitions. Images with the same `DJI_nnnn` prefix stay together. Prefixes
shared with the previously examined official test are quarantined. Decoded
RGB/thermal pairs are hashed; matching candidate groups are merged, and any
exact content overlap with the old official test is quarantined. These checks
do not prove geographic or weather independence.

The frozen partition is:

| Partition | Paired images | Capture groups | Use |
|---|---:|---:|---|
| Training | 20,218 | 46 | Detector optimization |
| Validation | 4,419 | 10 | Checkpoint, baseline and later controller development |
| Fresh holdout | 4,294 | 10 | Final evaluation after the new method is frozen |

All seven classes occur in each partition: ship, car, cyclist, pedestrian, bus,
drone and plane. The partition algorithm selects among 5,000 seeded whole-group
permutations using a predefined class/frame-balance objective. Labels contribute
to stratification; no model predictions or scores contribute to the split.
The 5,180 quarantined official training pairs are not fitted. The previously
examined official test remains exploratory evidence, not an untouched test.

The training YAML contains only train/validation image paths. No holdout image
directory is exposed to the loader. Native COCO annotation geometry is preserved
for exact development scoring. Training and Ultralytics-validation labels clip
boxes to image boundaries, with conversion counts recorded.

## Model and training recipe

The initialization is the frozen M3FD multi-scale YOLO26n checkpoint. The target
model has seven output classes. Compatible weights are transferred, and a
preflight check requires the four-channel stem to match the initialization
exactly. The class order and train/validation file membership are also checked.

The input is B,G,R,IR. A paired JPEG loader adds the thermal channel on demand
and uses the same linear resize for both modalities. RGB augmentation preserves
the thermal channel. This avoids an additional TIFF copy of the image corpus.

The pinned environment uses Ultralytics 8.4.14 and CUDA PyTorch on the workstation.
One GPU is selected by UUID only after three consecutive idle observations;
other jobs are not terminated. The recipe uses seed 42, batch 16, four workers,
AdamW at learning rate 0.001, weight decay 0.0005, three warmup epochs and AMP.
Training draws uniformly from 480/544/576/640 pixels. Mosaic and mixup are off.

Training is capped at 100 epochs or six hours of the optimization loop, with
patience 20 on validation mAP50–95. Validation and final checkpoint processing
can extend wall time beyond the optimization cap. Best-checkpoint selection
uses the pinned Ultralytics validation fitness at 640; exact native COCO scores
are computed separately after training. Scoped settings disable remote experiment
logging without changing the user's shared Ultralytics configuration.

## Automatic development comparisons

After training, the pipeline freezes a copy and SHA-256 of the selected model.
It evaluates fixed 320/480/544/576/640, the frozen sequence controller, guarded
controller, deadline EMA and AIMD on validation data only. Fixed 320 is an extra
diagnostic; the existing adaptive controls retain their original four-action
480/544/576/640 space.

All seven classes are scored using pycocotools IoU .50:.95 AP and AP-small.
Pedestrian recall uses IoU .5, confidence .25 and maxDets100, including small
pedestrians separately. The old source-trained scene controller receives only
the explicit shared-class feedback mapping: target bus (4) → source bus (0),
car (1) → car (1) and pedestrian (3) → People (4). Accuracy still scores all seven classes.

Measurements are offline serial service time and NVML GPU-device energy. They
do not measure real-time dropped frames, output age or energy of the full host.
Profiling uses a training image, outside the timed development loop. One run per
policy gives no repeat-run uncertainty. The fixed reference is selected by
highest validation COCO AP among settings with p95 service ≤25 ms, with lower
mean service as a tie-break; if no setting is feasible, the fastest is selected
and infeasibility remains explicit.

## Evidence and next stage

Remote study root:
`/mnt/12TB_Drive/whisenaj/EMCH585_training/rgbt_tiny_yolo_20261123`

Local status and completion archive:
`/media/imsel/4TBSSD/EMCH585_Artifacts/rgbt_tiny_yolo_20261123`

The remote root contains source snapshots and `pipeline_status.json`; its
`study` directory contains the frozen plan, pixel-hash inventory, split
manifests, preparation proof, training progress, epoch ledger and selected-model
hash. Development results are written after training. A separate local watcher
mirrors status and archives study outputs when the pipeline reaches a terminal
state. Raw images stay in their existing dataset location.

The next stage is to diagnose the trained detector's development failures and
design a targeted probing/fallback controller. Its fitting, published baseline
comparison and settings must be frozen before unlocking the fresh holdout.
The automatic first-stage pipeline never unlocks or evaluates it.
