# Held-out detector comparison — November 16, 2026

All three frozen checkpoints evaluated on the same 419 M3FD grouped test pairs (3,376 boxes), at a fixed 576-pixel long side on workstation GPU 0 (RTX 6000 Ada). No model fitting, threshold search, or checkpoint selection used these results.

| Model | Test AP | AP50 | AP-small |
|---|---:|---:|---:|
| yolo26n_ms | 41.21% | 63.58% | 14.58% |
| fasterrcnn | 40.16% | 65.18% | 18.38% |
| retinanet | 30.37% | 49.02% | 4.06% |

AP is COCO AP averaged over IoU 0.50:0.95. AP-small uses native box area below 32² pixels. See comparison.csv for unrounded fractions and metrics.json for per-class AP and recall.

## Protocol and limits

- Checkpoints, split hashes, confidence 0.001, resolution, and six-class mapping were frozen in plan.json before inference. Faster R-CNN selected epoch 62; RetinaNet selected epoch 31 by validation AP.
- Original YOLO26n multi-scale project checkpoint is used, not the separately reported fixed-576 fine-tuned checkpoint. YOLO test results were already examined in this repository; only the new families receive their first held-out test evaluation here.
- Input channel order is B,G,R,IR for all models. Native coordinates, class IDs 1–6, and COCO maxDets [1,10,100] are common. Detector-specific trained interpolation, padding, normalization, and output decoding remain in use; long-side size alone does not equalize tensor shape or compute.
- GT follows the original repository evaluator without clipping. 103 boxes extend outside image bounds. Their geometry is retained identically for all three models. Original additional-family validation metrics used clipped GT and therefore are not directly identical to this scoring convention.
- YOLO and the two-hour additional-family training runs have different recipes and budgets. The table compares these frozen checkpoints; it is not a matched-budget architecture study or evidence of adaptive-controller generalization.
- One grouped M3FD test split is not cross-environment robustness evidence. Timing wall_seconds includes model loading and scoring; it is not a latency benchmark. No new Jetson latency/energy claims are made.

## Artifacts and verification

- plan.json, environment.json, metrics.json, comparison.csv and raw_index.json capture configuration, hashes, scores, and provenance.
- Full predictions, native GT, all input-file hashes, and the evaluation log: /media/imsel/4TBSSD/EMCH585_Artifacts/heldout_20261116.
- Additional-family checkpoint backups with training metadata and epoch logs: /media/imsel/4TBSSD/EMCH585_Artifacts/trained_families_20261116. All checkpoint copies verified against the frozen plan.
- Input dimensions match the committed per-image sizes for all 419 pairs. Saved prediction IDs, classes, counts, finite values, and completion markers verified. Two evaluator tests cover native dimensions, unmodified label geometry, perfect/empty/wrong-class predictions and small-object scoring.

## Reproduction

On the workstation, from /mnt/12TB_Drive/whisenaj/EMCH585_training/families_20261113_1745, select an available GPU and run into a NEW output directory:

```bash
CUDA_VISIBLE_DEVICES=0 /home/whisenaj/anaconda3/envs/YOLO_Workspace/bin/python -m experiments.evaluate_families \
  --plan heldout_plan_20261116.json --output heldout_reproduction_NEW --device cuda:0
```

Requires the exact data/checkpoints from plan.json. Source: experiments/evaluate_families.py; source hash recorded in environment.json. No training or downloads are needed.

The next research milestone is to evaluate adaptive resolution and fixed/adaptive baselines across these three detectors, then measure the new models on the Jetson.
