# Jetson GPU follow-up — November 13, 2026

29 completed TensorRT GPU runs; 31044 processed frames.

This study uses one detector family on one physical Jetson AGX Orin.
The KAIST data are a frozen-model check on the first 1,200 frames of one night training-preview sequence, not official KAIST test evaluation.
The FLIR endurance data repeat 120 unique indoor pairs. These are not independent environments.
Strict FP32 TensorRT disables TF32; preprocessing remains the original CPU torch path.
The proposed guarded policy is experimental and is not a published-baseline reproduction.

## Measured runs

| Run | p95 / p99 ms | Controller mean ms | Rail-sum J/frame | Person AP | Causal aligned AP |
|---|---:|---:|---:|---:|---:|
| endurance_sequence_rep0 | 37.93 / 39.77 | 0.5545 | 0.6762 | — | — |
| endurance_sequence_rep1 | 37.80 / 39.75 | 0.5593 | 0.6757 | — | — |
| endurance_sequence_rep2 | 37.52 / 39.59 | 0.5484 | 0.6732 | — | — |
| endurance_sequence_rep3 | 37.74 / 39.79 | 0.5484 | 0.6763 | — | — |
| endurance_sequence_rep4 | 37.87 / 40.34 | 0.5523 | 0.6763 | — | — |
| kaist_aimd | 47.56 / 50.73 | 0.1010 | 0.8674 | 0.0150 | — |
| kaist_deadline | 44.72 / 48.52 | 0.1313 | 0.8222 | 0.0149 | — |
| kaist_fixed-480 | 45.00 / 47.28 | 0.1069 | 0.8320 | 0.0149 | — |
| kaist_fixed-544 | 49.55 / 51.79 | 0.1079 | 0.9269 | 0.0147 | — |
| kaist_fixed-576 | 56.22 / 59.63 | 0.1125 | 1.0416 | 0.0093 | — |
| kaist_fixed-640 | 51.08 / 53.39 | 0.1069 | 0.9553 | 0.0102 | — |
| kaist_guarded-no-hysteresis | 45.17 / 47.74 | 0.3722 | 0.8229 | 0.0149 | — |
| kaist_guarded-no-resource | 50.26 / 53.43 | 0.3454 | 0.9304 | 0.0113 | — |
| kaist_guarded-no-scene | 44.52 / 48.02 | 0.1382 | 0.8209 | 0.0149 | — |
| kaist_guarded-no-switch-cost | 44.91 / 47.35 | 0.3771 | 0.8296 | 0.0149 | — |
| kaist_guarded | 46.47 / 48.82 | 0.3726 | 0.8359 | 0.0149 | — |
| kaist_sequence | 50.39 / 53.50 | 0.3413 | 0.9395 | 0.0113 | — |
| resource_cpu_deadline | 40.65 / 44.34 | 0.1382 | 0.7636 | — | — |
| resource_cpu_guarded | 40.58 / 55.15 | 0.5944 | 0.7567 | — | — |
| resource_cpu_sequence | 39.55 / 44.00 | 0.5480 | 0.7056 | — | — |
| resource_gpu_deadline | 39.31 / 129.76 | 0.1389 | 1.0818 | — | — |
| resource_gpu_guarded | 38.06 / 125.64 | 0.5787 | 1.0278 | — | — |
| resource_gpu_sequence | 37.89 / 132.69 | 0.5438 | 1.0852 | — | — |
| resource_memory_deadline | 41.53 / 59.55 | 0.1513 | 0.8363 | — | — |
| resource_memory_guarded | 38.39 / 71.87 | 0.6078 | 0.7832 | — | — |
| resource_memory_sequence | 38.75 / 48.45 | 0.5861 | 0.7972 | — | — |
| stream_fixed-576 | 56.71 / 59.04 | 0.1129 | 1.0499 | 0.0094 | 0.0005 |
| stream_guarded | 50.94 / 53.80 | 0.4001 | 0.9963 | 0.0149 | 0.0095 |
| stream_sequence | 52.37 / 54.29 | 0.3493 | 1.0112 | 0.0113 | 0.0042 |

## Repeated-run uncertainty

Intervals below bootstrap means of run-level statistics across fresh process repetitions. They do not provide sequence-level or cross-environment confidence intervals.
- p95_ms: mean 37.7712; 95% bootstrap interval [37.63485818373738, 37.87845555401873]; n=5 runs.
- p99_ms: mean 39.8483; 95% bootstrap interval [39.66481718371502, 40.10520231322148]; n=5 runs.
- controller_mean_ms: mean 0.5526; 95% bootstrap interval [0.5491645599504409, 0.5561329773154285]; n=5 runs.
- rail_sum_J_per_frame: mean 0.6755; 95% bootstrap interval [0.6743124874113275, 0.6762783961368093]; n=5 runs.

## Measurement scope

- Stage timing includes CPU preprocessing/postprocessing, synchronized GPU execution span and host transfers. Service adds JPEG decoding and controller work.
- Rail sums are not wall-plug/model-only energy. CPU/GPU/memory contention energy includes the competing workload.
- Phase setup is included in service cost; worker initialization can produce a large transition outlier. Warmup/profile wall time is separate in each raw record.
- COCO evaluation uses native coordinates, an explicit People→person mapping, and ignores other human-like/uncertain regions. This is not official KAIST log-average miss rate.
- Causal aligned AP uses only the latest completed result at each recorded arrival; no tracking or future predictions. It is distinct from scoring skipped frames as empty.
- Resource conditions are short diagnostics. Correlation, shared-host load, thermal drift and small repetition counts limit causal/performance claims.
- Natural scene transition annotations, external power calibration, independently held-out domains and trained additional detector families are still required.
- Workstation 10.173.92.11 is authorized for training, but SSH access must be configured before remote jobs can run.

The source study plan was saved before execution. No model or controller parameters were fitted to KAIST labels.

## Fresh-process startup and gate checks

Five fresh-process/context/engine repetitions were measured. GPU driver and filesystem caches were retained; this is not a hardware-cold boot. Initialization ranged from 289.2–309.5 ms; first inference from 58.5–66.6 ms. Warm samples and complete subprocess duration are retained separately. Engine-build wall time is in the engine manifest.

Numerical gates passed on 32 FLIR and 32 KAIST native-shape cases at confidence .10, plus 64 KAIST cases at .01. The .10 KAIST gate contains only two matched detections, so the lower-confidence check supplies additional nonempty output coverage. These remain sampled numerical checks, not AP-equivalence confidence bounds.

Lower-confidence KAIST gate: 5 reference detections, 5 class/IoU-matched predictions.

## Important negative result

Frozen-model person AP on the night prefix was approximately 0.009–0.015 across the main configurations, with very low recall at confidence .25. Class names and image/label dimensions were verified; annotated examples were inspected. This is weak transfer under the declared adaptation, not evidence of robustness or usefulness. No test-data normalization, confidence tuning or fine-tuning was applied to conceal the result.

The two extra scene/resource-removal ablations are a supplementary block. Source snapshots v1/v2 are preserved; the v2 change adds those branches and CLI names, rather than fitting model/controller parameters to KAIST. Comparisons remain diagnostic on a single prefix.

The observed fixed-point Pareto set is in `observed_fixed_frontier.json`. It covers the four configured resolutions, is descriptive (one run each), and does not establish a statistically superior operating point or the best configuration outside this action space.

## Setup energy and retrospective content transitions

A separate fresh builder process (OS/CUDA caches retained) measured 57.23 s and 1140.17 J named-rail energy for the shared dynamic engine. Runtime/context initialization plus the same twelve warmup calls measured 0.624 s and 12.63 J. The validated canonical engine was not replaced. These are repeat-procedure measurements, not exact energy of the original build, and the short warmup integral is coarse at 10 Hz.

For an explicitly attributed deployment lifecycle, report steady J/frame + (attributed build J + startup/warmup J)/N. Do not silently hide setup, but do not assume an engine is rebuilt on every video session either. Training energy and external wall-power calibration remain unmeasured. Full setup records are in `setup_energy.json`.

There are 11 stable person-presence/absence transitions in the evaluated prefix, derived retrospectively from non-ignored ground truth with at least ten consecutive frames per state. The controller was not given these labels. One-second windows describe resolution mixes, service time and first-switch source-time delay in `content_transitions.json`. They are correlated diagnostic windows, not independently annotated weather/day/night cuts or transition-window AP estimates.
