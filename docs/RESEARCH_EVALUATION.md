# EMCH585 evaluation expansion

Status updated November 13, 2026: the CPU pilot is followed by a validated
TensorRT GPU study, labeled KAIST night-preview transfer, repeated-run endurance,
separate CPU/GPU/memory contention, causal streaming accuracy, cold-process tests
and guarded-controller ablations. This remains an incomplete generalization study.

Current results: [November 13 GPU report](../results/processed/orin_followup_20261113/REPORT.md).
Deployment/reproduction: [GPU setup](GPU_SETUP.md). Review drafts are in
[deliverables](deliverables/REPORT_DRAFT.md). The working browser demo remains CPU.
The following CPU-pilot section is retained as historical protocol documentation;
it must not be mistaken for the complete current evidence.

The local physical machine is an NVIDIA Jetson AGX Orin Developer Kit in MAXN mode.
The installed project environment uses PyTorch 2.10 CPU. Consequently these runs
originally measured CPU inference on the Orin. The separate November 13 runs use native TensorRT GPU execution with CPU preprocessing; CPU PyTorch is preserved. The live
C++ camera acquisition service stays active. UI inference is paused during the
pilot and restored afterwards. No power-mode, fan, clock, or system-wide CPU
settings are changed. Resource restrictions apply to benchmark processes only.

## Claims and evidence gates

| Requested claim | Available implementation/evidence | Required before making the full claim |
|---|---|---|
| Orin energy, tails, controller overhead | CPU/GPU stage and rail traces; five-run GPU endurance CIs; separate setup energy | External power cross-check, broader workloads, complete training/lifecycle accounting |
| Changing compute availability | 4→1→4 worker trace and separate CPU/GPU/memory pressure | Independent trace repetitions, validated DVFS/power-mode sweeps and wider contention levels |
| Continuous RGB/IR video and transitions | Ordered FLIR replay, labeled KAIST night prefix, causal alignment, retrospective person-presence events | Independent natural environment transitions, time-aligned label verification, hardware synchronization |
| Strong adaptive baselines | Fixed configurations, sequence/deadline/AIMD, guarded and four ablations; pinned upstream sources | Faithful published-system execution/training, equal profiling/action budgets |
| Robustness across environments | Explicit synthetic stress; weak labeled night transfer | Independent held-out indoor/outdoor/day/weather sequences with labels |
| General technique | Common adapter, one trained family evaluated, two additional builders/recipes prepared | Actually trained/evaluated families, multiple labeled datasets and independent physical hardware |

A second checkpoint of the same detector is not another detector family. CPU
thread restrictions on one Orin are not another hardware platform. Artificial
corruptions of one indoor clip are not new physical environments. Random policy
selection is a diagnostic, not an adequate strongest baseline. A detection
count or confidence score is not detection accuracy.

## Executable pilot

From the repository root, using the existing `.venv`:

```bash
.venv/bin/python -m pytest -q tests/test_video_experiments.py
.venv/bin/python -m experiments.capture --output captures/new_flir_clip --frames 120 --fps 8
.venv/bin/python -m experiments.pilot \
  --manifest captures/new_flir_clip/manifest.json \
  --output captures/new_orin_results --reps 2 --frames 120
.venv/bin/python -m experiments.report captures/new_orin_results
```

The pilot has 40 runs in a reproducible randomized order: nine policies × two
repetitions with 4→1→4 CPU threads; four synthetic corruptions × three policies;
four CPU-contention runs; three realtime runs; and three clean constant-thread
controls for the corruption comparisons. Corruptions apply only in the
middle third of the stream to expose entry/recovery behavior. Corruption and
contention conditions currently have one repetition each and are diagnostics.
Never interpret their differences as statistically established improvements.
Each run warms every resolution, samples a two-second background period, then
processes the clip. Output files are preserved; rerunning the pilot skips saved
results. Use a new output directory for a changed protocol or code version.

Outputs contain raw predictions and measurements, checkpoint/manifest/code
hashes, hardware and power mode, actual power samples, per-phase temperatures
and CPU frequencies, resource-transition setup costs, and per-frame decode,
choose, detector, feedback-update and service times. `choose_ms + update_ms`
includes the sequence controller's feedback feature extraction. It excludes
detector computation, which is charged identically to all policies.

`deadline` uses online EMA latency estimates, resolution-squared extrapolation,
and a 10% reserve. `aimd` reduces one resolution on a miss and increases one only
after four consecutive observations below 80% of budget. These local baselines
are explicitly not implementations of named published algorithms. Their tuning
parameters must eventually be selected on validation sequences, never test.

The runner binds its process to a recorded CPU subset. A two-worker contention
condition pins owned busy workers to the same subset only in the middle third.
Every child is terminated in `finally`. Thread-count and contention experiments
are separate so their effects can be assessed independently. Host activity,
DVFS, and temperature remain possible pilot confounds.
The 120 ms budget is an illustrative preselected pilot budget, not a validated
application SLA. The initial local execution began with 37 jobs; the three clean
constant-thread controls are recorded as a supplementary block, with their own
execution order, rather than rewriting the observed initial order.

## Measurement definitions

- Offline service latency includes phase setup, decoding, controller selection,
  preprocessing/inference/postprocessing, output transfer, and feedback update.
  Every recorded frame is processed in order. The fraction exceeding 120 ms is
  a **service-budget miss rate**, not a streaming deadline miss rate.
- Realtime replay uses recorded host-arrival intervals, keeps the newest due
  input, explicitly counts skipped frames, and records output age. This is a
  reproducible input-arrival simulation, not a hardware-synchronized live sensor
  latency measurement. The current pilot has no tracking/reuse of stale boxes.
- Power is voltage (mV) × current (mA) / 1e6 W. Integrate each named INA3221 rail
  with trapezoids over actual monotonic timestamps and interpolate boundaries.
  Out-of-coverage integrals are unavailable, not zero. Never sum a shunt-sum
  channel with the underlying named rails. The reported named-rail sum is not
  wall-plug power or model-only energy. Background-subtracted estimates can be
  negative and must retain that sign. Sampling nominally at 10 Hz cannot resolve
  individual inference energy; aggregate over long steady phases.
- Switching costs: the forced-switch policy exercises all resolutions. The
  report matches switching frames to fixed-destination runs at the same frame,
  seed and phase. Differences are descriptive because separate runs differ in
  DVFS/thermal state. Follow with a randomized, paired same-frame stay/switch
  microbenchmark and CUDA-event/host clocks; report each directed transition.
- A short clip's p99 is a near-maximum observation. Do not present a 120-frame
  pilot percentile as a stable production tail estimate.

NVIDIA documents AGX Orin INA3221 power monitoring and its rails in the
[Jetson Linux power guide](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html).

## Continuous inputs and accuracy

A manifest supplies strictly increasing `timestamp_s`, unique `id`, `rgb` and
`thermal` image paths, a `sequence` ID, and optional `scene` and `environment`.
Paths are relative to the manifest. Sequence IDs mark independent videos;
scene labels mark content transitions inside videos. Never randomly reorder
frames or reset a controller at a content transition using test annotations.
The existing sequence controller resets only at independent sequence boundaries.

For labeled data, add `coco_annotations` (relative path) and explicit
`class_to_category` (model index to COCO category ID). Labels must refer to the
same calibrated/cropped pixels as the inputs. The scorer checks IDs and native
image dimensions. Run after installing `pycocotools`:

```bash
.venv/bin/python -m experiments.score --manifest /data/test/manifest.json \
  --result /results/policy.json --output /results/policy_accuracy.json
```

The scorer reports COCO AP/AP-small and recall overall and by environment.
Realtime scoring includes all input frames, giving skipped frames no detections;
this explicit dropped-frame AP convention is not the published streaming-AP
metric. A full video evaluation also needs causal time-aligned streaming AP,
small-person recall, transition-window AP, false positives, and detection
flicker/track continuity. COCO labels need stable class mapping across datasets;
for pedestrian-only datasets score the common pedestrian class explicitly.

Capture from the current FLIR server uses independent calibrated JPEG endpoints.
The original sensor timestamps are not exposed. Record this synchronization
limitation; do not call the HTTP arrival skew sensor synchronization accuracy.

## Confirmatory experiments

Pre-register the metrics, comparisons, non-inferiority margin (e.g. ≤0.5 AP point
loss, selected with the supervisor), latency budgets, seeds and repetitions.
Collect enough independent sequences and minutes per resource phase that tail
estimates stabilize. Start with ≥5 independent run repetitions and ≥10,000
processed frames per condition; extend based on run-level CI precision, not
whether a favorable p-value appears. Bootstrap whole sequences/runs, not adjacent
frames. Report paired accuracy/energy/latency differences with 95% CIs and
Pareto frontiers, including all fixed resolutions and best-fixed hull.

1. **Orin deployment:** validate a JetPack-compatible CUDA PyTorch build and a
   TensorRT backend using equivalent inputs/predictions before benchmarking.
   Sweep supported power modes and budgets; record clocks, thermal state,
   memory, engine-build/cache state, synchronization costs and idle power.
   Power-mode changes should be a separate documented experiment, not silently
   applied to the live camera workstation. Compare cold switches, warmed shape
   switches, and static-shape engines; count warmup and calibration as setup.
2. **Resource dynamics:** replay low→high→low CPU/GPU contention, memory-bandwidth
   pressure, and thermal throttling traces. Measure budget violation duration,
   controller recovery time, oscillation, and energy through entry/recovery.
   Include predictable and previously unseen traces. Keep workload content fixed
   when testing resources and resources fixed when testing content.
3. **Temporal baselines:** reproduce a validated adaptive content baseline such
   as [AdaScale](https://arxiv.org/abs/1902.02910), and a contention-aware baseline
   such as [ApproxDet](https://arxiv.org/abs/2010.10754) or a suitable faithful
   [Chanakya](https://papers.nips.cc/paper_files/paper/2023/file/ae2d574d2c309f3a45880e4460efd176-Paper-Conference.pdf)
   adaptation. Document changed action spaces/modality handling. Use common
   splits, detector weights, resolution choices, latency budgets and profiling
   budgets; charge probing/reconfiguration overhead. Include no-hysteresis,
   no-resource-feedback, no-scene-feedback and switching-penalty ablations.
4. **Environments:** acquire labeled indoor/outdoor, day/night, glare, rain/fog,
   camera-motion, object-density and distance sequences. Annotate transition
   times without feeding those labels to the policy. Report each domain and
   worst-group performance. Separate genuine environments from synthetic
   corruption severities and modality outage/registration stress tests.
5. **Generalization:** train/evaluate at least three materially different detector
   families, e.g. early-fusion one-stage YOLO, two-stage Faster R-CNN, and an
   RGB/T transformer, with appropriate multimodal stems and dataset heads.
   Use M³FD plus [KAIST](https://github.com/SoonminHwang/rgbt-ped-detection) and
   [LLVIP](https://github.com/bupt-ai-cz/LLVIP). KAIST supplies temporal RGB/T
   sequences; LLVIP provides paired low-light images and separately released
   raw videos. Treat still-image data and continuous video as different tests.
   Evaluate on Orin GPU, a desktop discrete GPU, and an x86 CPU (optionally a
   second embedded platform). A custom adapter factory can be loaded with
   `--adapter package.module:factory --weights ... --device ...`.
   Execution backends on one machine do not establish independent physical
   platforms; record host identities and use separate machines for that claim.
6. **Transfer protocol:** split by sequence/location; fit policy and cost models
   on train/validation only. Compare frozen zero-shot transfer against an
   explicitly budgeted hardware-only recalibration and dataset fine-tuning.
   Hold out one detector, domain and platform in turn. Evaluate every cell or
   report missing cells. Do not claim universal efficacy from an adapter API.

## Candidate general technique (to validate)

Frame the contribution as a causal, detector-independent resolution controller
that combines previous detections (content utility), online measured service
cost (resources), a deadline constraint, and an explicit reconfiguration penalty.
Its interface should need only previous detections, timing/resource observations,
and candidate resolutions. The present project supplies the content controller
and this experiment harness; combining and validating those components remains
work. Detector-independent plumbing is not evidence that the policy transfers.

A credible result is a consistent accuracy/energy/tail-latency frontier improvement
against the best fixed hull and strong adaptive baselines, with all profiling,
controller and switching costs included. If adaptive policies tie the hull, report
that result and identify the conditions where adaptation helps or fails.

Paired constant-rate recordings can also be imported with
`python -m experiments.import_video --rgb rgb.mp4 --thermal ir.mp4 --output NEW_DIR
--sequence SEQUENCE_ID --environment ENVIRONMENT`. The importer rejects differing
FPS or dimensions and preserves frame order in lossless PNGs. It assumes aligned
frame indices; registration and sensor synchronization still require independent
verification. It does not generate labels or infer environment names.

After the pilot, run the complementary paired switch-cost and transition analyses:

```bash
.venv/bin/python -m experiments.switch_cost \
  --manifest captures/research_flir_pilot/manifest.json \
  --output captures/orin_switch_cost.json --reps 10
.venv/bin/python -m experiments.transitions captures/new_orin_results
```

Transition response reports time until five consecutive budget-compliant outputs,
time to first resolution change, and censoring horizon if no recovery is observed.
This is a budget-response measure, not accuracy recovery. Natural scene changes
are analyzed only when real scene annotations exist in the input manifest.

Before reusing a captured clip, verify its recorded RGB/IR byte hashes with
`python -m experiments.verify_capture captures/research_flir_pilot/manifest.json`.
Imported videos currently lack per-frame checksums; freeze and archive those
inputs separately before a confirmatory study.


## November 13 update and boundaries

The GPU follow-up completed 29 runs and 31,044 processed frames, including five
2,400-frame endurance process repetitions (12,000 timed replays of 120 unique
FLIR pairs), 1,200-frame frozen-model KAIST preview comparisons, three realtime
runs, three resource-pressure types, and all four guarded-policy ablations.
The main 27-job block and the supplementary scene/resource ablations have separate
plans and source snapshots. Fresh-process/context startup is measured five times;
GPU driver/filesystem caches are retained. Paired warm GPU switching has ten
repetitions per directed transition (120 pairs total).

The guarded policy is a candidate implementation, not an established general
technique. It caps the sequence controller's preferred resolution using online
cost estimates, a 10% budget reserve, 1 ms switch reserve and four-frame dwell.
Those values were engineering choices, not target-label-fitted parameters.
Ablations remove scene feedback, resource feedback, hysteresis and switch reserve.
Warmup/profile wall time is recorded separately; full setup/training energy
accounting remains incomplete.

The verified official KAIST preview has only night sequence set04/V001 from the
original training preview. Its actual image/label dimensions are 640x512. The
study does not use official KAIST test sequences. Only People→person is scored;
other human-like/uncertain categories are ignore regions. Model/person class
mapping and example overlays were checked. Frozen transfer was weak (person
AP roughly 0.009–0.015), so no real-environment robustness claim follows from
this experiment. The sampled native-shape CPU/TensorRT gates passed, including
an additional low-confidence check, so the weak result was not silently dismissed
or hidden with test-data tuning.

The time-aligned scorer now evaluates latest-completed boxes at each recorded
arrival, without future predictions, tracking, interpolation or sequence leakage.
Its exact convention is explicit and is not an official KAIST benchmark score.
Run-level uncertainty is reported across five fresh process repetitions; this
is not a confidence interval over independent environments or physical platforms.

Workstation 10.173.92.11 is authorized for training. It is reachable but SSH
credentials for the attempted whisenaj account were denied. No remote jobs ran.
GPU-only additional-family training and adapters are prepared in
`experiments/train_families.py` and `experiments/detector_families.py`; construction
and dataset contracts were tested, but GPU training/validation remain pending.
Published baseline source snapshots are pinned; their faithful execution remains
pending. See [baseline plan](BASELINE_REPRODUCTION_PLAN.md).
