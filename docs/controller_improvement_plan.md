# Plan: a resolution controller that beats random (and fixed) resolution

Status 2026-10-01. Diagnostics: `results/processed/diag_oracle_noise.csv`, `diag_headroom_vs_hull.csv`.

## What the diagnostics say

| Question | Measurement | Answer |
|---|---|---|
| How much of the oracle's +3.5 AP is real? | Choose per-image resolutions on seed 42, score on seeds 123/456 | **About half is noise** (winner's curse over 3 noisy per-image APs). Real headroom over a random 320/480/640 mix: +1.3–2.2 AP |
| Are random mixes of 320/480/640 the right baseline? | Fixed 400/544/576 px on the same multi-scale model, pinned CPU | **No.** The frontier is concave: **576 px = 0.417 AP at 23.5 ms beats 640 px (0.415 at 28.0 ms)**; 544 px = 0.414 at 21.8 ms. Intermediate fixed sizes sit +1.5–1.7 AP above the mix line |
| Is there real headroom above the *best fixed* frontier? | Cross-seed oracle vs the upper hull of all fixed sizes and their mixes | **Yes, but small: +0.6 to +1.8 AP** (per image), +0.4 to +1.3 AP (one choice per sequence). Largest with a finer action set {400, 480, 544, 576, 640} |
| Where does the headroom live? | Per-sequence oracle (one resolution per scene) | **Mostly at scene level**: 60–90% of the per-image headroom survives one decision per sequence |
| Does merging the probe into an escalated pass help? | NMS-merge of 320 + 640 detections | +0.4 AP only. Not enough to rescue the cascade on its own |

Implication: the achievable gain is about **+1 AP at about 20% lower cost than 640 px**. The bar is the
best fixed resolution, not a random mix. The current v2 rule (+0.3 AP over a 320/480/640 mix) actually loses to
fixed 544/576 px.

## Success criterion

A controller succeeds only if all of the following hold, on test, at the same mean CPU cost:

1. It beats **the best-fixed hull**: every fixed resolution and every random mix of two fixed resolutions.
2. It beats **cost-matched random**.
3. The margin is **≥ +0.5 AP**, and its 95% sequence-bootstrap CI excludes 0.
4. It is scored on detections from **seeds 123/456**, not only the seed used to fit it.
5. Its online **p95 latency** is no worse than the fixed resolution of equal mean cost.

Selection stays val-only, and every variant tried is reported (F6 style).

## Methods, in order

### M0: Fix the baseline and the action set (first; about 0.5 day, mostly done)
- Add fixed 400/544/576 px as B3 operating points in T1/F1/F6, and use the best-fixed hull as the frontier everywhere.
- Controllers choose from **{400, 480, 544, 576, 640}**, not {320, 480, 640}. 320 px is never on the hull.
- Optional: retrain MS with per-batch sizes {320, 352, …, 640} (about 1 h on one GPU). This may lift the whole
  frontier, which then becomes the baseline.

### M1: Denoise the training targets (about 0.5 day, inference only)
Half the raw oracle gain is noise, so a controller fit to raw per-image AP learns noise.
- **Seed-averaged utilities**: run MS seeds 123/456 on train and val, then average per-image AP over the 3 seeds.
- **Sequence shrinkage**: q̃(i,r) = α·q(i,r) + (1−α)·mean over i's sequence; choose α by val CV. Most real
  headroom is sequence-level.
- Refit the cost-aware policy tree (existing `controller_fit.py`) on q̃ with the new action set.
- **Go/no-go:** if this doesn't beat the best-fixed hull on val by ≥ 0.3 AP, skip to M3/M4 and stop tuning trees.

### M2: Cheaper decision inputs (about 0.5 day)
The CPU analyzer still costs 2.3 ms, 8% of a 640 pass.
- Compute luminance during the resize kernel, which already reads every pixel, or take it from the camera's
  auto-exposure statistics (free on a real sensor). Target analyzer cost ≈ 0 on both platforms.

### M3: Learned controller, RQ3 (2–3 days)
- Input: 4-channel thumbnail (128×96). The IR channel sees small warm objects that whole-frame statistics miss.
- Model: about 50k-parameter CNN predicting the utility vector over the action set, plus a cost term at
  decision time. Budget ≤ 0.5 ms GPU / ≤ 2 ms CPU, measured online.
- Labels:
  - **Cross-fitted**: train two MS models on halves of train (about 1 h each on separate GPUs), so all 2,940 train
    images get labels from a detector that never saw them.
  - Then apply M1 denoising. Labels from the full model's own train predictions are inflated.
- Cheap variant first: **scene retrieval (kNN)**. Embed the thumbnail, find the nearest train sequences, use
  their mean utility. This targets exactly the sequence-level structure.

### M4: Amortised (per-scene) control (2 days; scope decision needed)
Headroom is mostly per scene, so decide once per scene instead of per frame:
- Run an expensive probe (a 640 pass plus probe features, ρ = 0.39) on 1 frame in K.
- Apply its decision to the next K frames, and re-trigger on scene change (thumbnail correlation, already computed).
- The probe cost is divided by K.
- Simulate on M3FD sequence groups first. Real deployment needs video. The proposal lists temporal adaptation as
  out of scope, so **this needs your sign-off**.

### M5: Tail-latency-aware decisions (0.5 day, with M1/M3)
Mixes inherit the tail of their largest resolution, which is why v2 matched fixed-640 on p95.
- Add a p95 or deadline constraint to the policy objective: cap the share of the largest resolution, or prefer
  576 over 640 when within ε.

### Low priority
- Probe-merge cascades (+0.4 AP, verified).
- Region-of-interest resolution: out of scope.

## Order and decision points

| Week | Work | Decision |
|---|---|---|
| 1 | M0 + M1 + M2 (offline, no new detector training) | Does a denoised policy beat the best-fixed hull on val? |
| 2 | M3 (cross-fit 2 MS models; kNN, then CNN) + Orin bring-up | Does the learned controller capture ≥ 50% of the cross-seed headroom (H3)? |
| 3 | M4 simulation (if approved) + M5; online benchmarks on CPU and Orin | Final F1/F6 against the best-fixed hull |

If nothing clears the bar, the finding stands as a result: on M3FD, a well-chosen fixed resolution
(about 576 px) captures most of the benefit, and per-frame adaptation is worth at most about 1 AP.
