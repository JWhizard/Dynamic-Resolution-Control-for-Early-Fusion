# Jetson AGX Orin pilot results

40 completed runs; 4769 processed frames.

These are CPU pilot measurements on one unannotated indoor FLIR preview clip.
Accuracy, natural scene-transition robustness, and cross-platform generality are unmeasured.
Short-run p99 values are descriptive. Named-rail energy is not wall-plug energy.
Thread-trace and CPU-contention tests are separate conditions. The C++ capture service remains active.

| Run | p50 / p95 / p99 ms | Controller mean ms | Budget misses | Rail-sum J/frame |
|---|---:|---:|---:|---:|
| aimd_contention | 153.77 / 1306.57 / 1330.61 | 0.1037 | 70.8% | 10.5332 |
| aimd_rep0_threads | 116.95 / 158.86 / 191.91 | 0.1070 | 37.5% | 2.0146 |
| aimd_rep1_threads | 149.32 / 237.88 / 291.25 | 0.1047 | 93.3% | 3.4749 |
| deadline_clean_constant | 156.15 / 258.38 / 326.26 | 0.1375 | 78.3% | 3.5175 |
| deadline_contention | 124.69 / 1321.15 / 1356.49 | 0.1237 | 53.3% | 8.5677 |
| deadline_ir_dropout | 164.79 / 235.93 / 249.45 | 0.1310 | 85.8% | 3.5703 |
| deadline_ir_shift | 157.88 / 287.19 / 501.02 | 0.1276 | 76.7% | 3.7434 |
| deadline_realtime | 126.60 / 173.25 / 187.37 | 0.1265 | 63.6% | 2.2498 |
| deadline_rep0_threads | 147.61 / 248.65 / 303.60 | 0.1469 | 94.2% | 3.3835 |
| deadline_rep1_threads | 148.46 / 298.17 / 701.01 | 0.1459 | 88.3% | 3.9295 |
| deadline_rgb_blur | 173.22 / 268.83 / 314.83 | 0.1367 | 90.8% | 3.8616 |
| deadline_rgb_dark | 144.37 / 249.83 / 342.46 | 0.1320 | 80.0% | 3.3511 |
| fixed-480_rep0_threads | 122.98 / 166.38 / 189.30 | 0.1023 | 57.5% | 2.1540 |
| fixed-480_rep1_threads | 148.94 / 224.75 / 249.61 | 0.1173 | 91.7% | 3.3010 |
| fixed-544_rep0_threads | 171.63 / 256.92 / 330.75 | 0.1069 | 97.5% | 3.8572 |
| fixed-544_rep1_threads | 142.08 / 184.05 / 206.29 | 0.1063 | 85.8% | 2.4844 |
| fixed-576_clean_constant | 206.24 / 303.90 / 349.89 | 0.1137 | 100.0% | 4.7272 |
| fixed-576_contention | 160.49 / 765.48 / 820.76 | 0.1031 | 100.0% | 5.9717 |
| fixed-576_ir_dropout | 142.76 / 196.55 / 209.04 | 0.1029 | 100.0% | 2.7695 |
| fixed-576_ir_shift | 141.70 / 200.48 / 210.56 | 0.1039 | 99.2% | 2.6747 |
| fixed-576_realtime | 165.27 / 219.13 / 233.29 | 0.1076 | 100.0% | 2.8733 |
| fixed-576_rep0_threads | 162.21 / 212.14 / 226.54 | 0.1049 | 100.0% | 2.8080 |
| fixed-576_rep1_threads | 152.43 / 206.78 / 237.80 | 0.1022 | 98.3% | 2.7061 |
| fixed-576_rgb_blur | 194.18 / 285.63 / 333.19 | 0.1102 | 100.0% | 4.2905 |
| fixed-576_rgb_dark | 139.04 / 189.24 / 241.91 | 0.1042 | 98.3% | 2.6365 |
| fixed-640_rep0_threads | 230.07 / 330.45 / 458.12 | 0.1670 | 100.0% | 5.1730 |
| fixed-640_rep1_threads | 220.51 / 324.99 / 556.56 | 0.1114 | 100.0% | 5.0436 |
| random_rep0_threads | 197.19 / 273.63 / 300.46 | 0.2744 | 98.3% | 4.3501 |
| random_rep1_threads | 152.57 / 205.49 / 239.52 | 0.2719 | 87.5% | 2.6597 |
| sequence_clean_constant | 167.82 / 239.97 / 280.03 | 0.5260 | 85.8% | 3.7984 |
| sequence_contention | 163.36 / 1315.97 / 1338.67 | 0.5141 | 70.0% | 9.2187 |
| sequence_ir_dropout | 199.13 / 289.22 / 397.70 | 0.4788 | 98.3% | 4.3441 |
| sequence_ir_shift | 164.72 / 245.85 / 283.74 | 0.5101 | 87.5% | 3.5880 |
| sequence_realtime | 129.87 / 173.32 / 209.86 | 0.5004 | 77.4% | 2.3770 |
| sequence_rep0_threads | 149.05 / 223.03 / 241.24 | 0.5314 | 92.5% | 3.3786 |
| sequence_rep1_threads | 125.21 / 176.12 / 191.79 | 0.5143 | 60.8% | 2.2579 |
| sequence_rgb_blur | 141.88 / 236.95 / 274.56 | 0.5018 | 69.2% | 2.9618 |
| sequence_rgb_dark | 126.64 / 187.77 / 197.16 | 0.4877 | 54.2% | 2.3398 |
| switch_rep0_threads | 190.68 / 271.63 / 296.09 | 0.1137 | 97.5% | 4.2440 |
| switch_rep1_threads | 184.81 / 265.73 / 290.37 | 0.1131 | 100.0% | 4.1130 |

Switch deltas match frame, repetition, resource phase, and destination resolution against a fixed run.
They are descriptive paired comparisons across separate runs; DVFS, temperature, and order can confound them.
Use a randomized same-frame switch/stay microbenchmark for a causal switching-cost estimate.
421 matched switch observations recorded in `switch_matched_deltas.json`.


## Interpretation and validation

The 4,769 processed replay frames reuse **120 unique indoor frame pairs**. They
are not 4,769 independent observations or multiple environments. Two repetitions
of the thread-trace conditions expose substantial run-to-run variation. This
pilot does not establish a consistent adaptive-policy advantage.

The sequence policy's mean controller cost was **0.514–0.531 ms** in the two
thread-trace runs. Its service p95 was **176–223 ms**, and p99 **192–241 ms**.
These service costs include decoding and controller work, differ from dashboard
inference-only timings, and include the prescribed resource changes.

In the realtime runs, deadline feedback processed 118 inputs and dropped 2;
sequence processed 115 and dropped 5; fixed-576 processed 96 and dropped 24.
All use the same recorded host-arrival trace. Detection accuracy is unmeasured,
so those counts alone do not establish which policy is best.

With two competing CPU processes, the AIMD controller stayed at its configured
480-pixel floor throughout the contention phase and missed every 120 ms service
budget in that phase. Mean service latency there was approximately 1,237 ms.
This is a concrete failure of resolution-only adaptation within this action
space; it is not proof that a lower resolution or a different detector cannot
meet a different budget.

The separate paired experiment measured all 12 directed warm resolution
transitions, with five randomized same-frame stay/switch pairs per transition.
Directed mean deltas ranged from **−30.7 to +57.7 ms**. Negative deltas and the
large spread indicate noisy short-run effects. Do not claim a stable universal
switching penalty, or extrapolate these CPU observations to TensorRT engines.
Raw paired observations are in `paired_switch_cost.json`.

All 40 runs have covered named-rail energy integrals and no recorded power read
errors. Every run satisfies processed + dropped = input frames. All 240 input
files passed SHA-256 verification. All 24 targeted tests passed, including video
import order, power units/integration, adaptive feedback, censored transition
response, rejection of unlabeled accuracy, and perfect/empty synthetic COCO
scorer fixtures. Those fixtures do not supply real FLIR accuracy evidence.

`raw_index.json` lists the full local raw files and their hashes. Raw predictions,
power samples and frame timings remain under `captures/orin_pilot_results/`;
the captured clip remains under `captures/research_flir_pilot/`.

## Work still required for the requested research claims

- Validate and benchmark Jetson CUDA/TensorRT execution, longer tails and
  calibrated external power measurements.
- Obtain labeled continuous RGB/IR sequences, annotate natural scene transitions,
  and score accuracy plus time-aligned streaming behavior.
- Reproduce strong published adaptive baselines with equivalent training,
  profiling and action-space budgets; the local feedback policies are diagnostic
  baselines, not faithful published-system reproductions.
- Test independent real environments and held-out locations/sequences. Synthetic
  darkening, blur, IR dropout and misalignment are stress tests only.
- Evaluate multiple trained detector families, multiple labeled datasets, and
  additional physical hardware. The adapter interface enables this work; only
  one detector, one indoor clip and one physical platform were exercised here.

The complete protocol and evidence gates are in
[`docs/RESEARCH_EVALUATION.md`](../../../docs/RESEARCH_EVALUATION.md).
