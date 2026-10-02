# Controller results (test split, MS model, 3 detector seeds)

All controllers are fit on train, selected on val, and scored once on test with detections from seeds
42/123/456. "Gain" is AP minus the **best-fixed hull** at the same mean CPU cost: the best of every fixed
size in {320, 400, 480, 544, 576, 640} and every two-size random mix (4 pinned CPU threads, frozen cost
model `results/processed/cpu_cost_model_v3.json`). Bootstrap CIs resample test sequences (1,000 reps;
`bootstrap_vs_hull.py`, validated against pycocotools in `tests/test_bootstrap_ap.py`).

## Without the controller's own cost (as fit; optimistic for anything but brightness)

| Controller | Gain vs hull | 95% CI | Per seed |
|---|---|---|---|
| Learned CNN ensemble (5×w24, 96×128), seed-avg labels | **+0.65** | −0.09 to +1.34 | +0.22 / +0.63 / +1.10 |
| … retrained (init offsets 5, 10) | +0.88 / +0.85 | — | all positive |
| Learned kNN, cross-fit labels | +0.44 | −0.16 to +1.08 | +0.37 / +0.56 / +0.39 |
| Learned CNN, cross-fit labels | +0.23 | −0.47 to +1.06 | +0.02 / −0.14 / +0.80 |
| Learned kNN, seed-avg labels | +0.18 | −0.21 to +0.54 | |
| v3 brightness tree, seed-avg / raw / cross-fit labels | 0.00 / −0.26 / −0.44 | all include 0 or are below it | |

## Including the controller's measured CPU cost (the fair comparison)

| Controller | Controller cost | Gain vs hull |
|---|---|---|
| CNN ensemble 5×w24, 96×128 | 5.15 ms | −0.08 |
| Single CNN w24, 96×128 | 1.09 ms | −0.06 |
| Single CNN w12, 48×64 (3 inits) | 0.35–0.38 ms | +0.38 / −0.14 / −0.21 |
| Single CNN w8, 48×64 | 0.34 ms | +0.42 |
| kNN, 48×64 | 0.97 ms | +0.01 |
| Brightness tree (fast luminance) | 0.07 ms | 0.00 |

Near 18–25 ms the hull rises by about 0.56 AP per ms, so 1 ms of controller compute costs about 0.6 AP.
Once the controller pays for itself, every learned controller ties the best fixed resolution
(about 576 px). The cheap CNNs scatter between −0.2 and +0.4 across retrains.

## Conclusion so far

- Per-frame headroom over the best fixed resolution is real but small: +0.6 to +1.8 AP (cross-seed oracle).
  It is mostly per-scene: one decision per sequence keeps 60–90% of it.
- Signal that predicts it exists (learned controllers gain +0.4 to +0.9 AP when free), but on a
  compute-bound CPU the controller's own cost cancels the benefit.
- Open option (needs a scope decision; uses video): **amortised per-scene control**, i.e. run the
  expensive controller once per K frames.
