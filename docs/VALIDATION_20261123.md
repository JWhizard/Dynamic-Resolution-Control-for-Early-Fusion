# November 23 validation

The RGBT-Tiny frozen-transfer comparison completed all three models on all
12,590 official test pairs, or 37,770 model/image evaluations. Saved M3FD
predictions were rescored on the same bus/car/person category subset. Native
predictions, per-frame timings, original annotation hashes, checkpoint hashes
and all paired image hashes are preserved outside Git. The compact result
export checks complete frame-ID coverage, finite predictions, category IDs and
prediction/timing file hashes.

The first remote evaluator assumed six classes and failed while summarizing the
shared-class source score. Its source is preserved. The corrected evaluator
supports category IDs 1/2/5; Faster R-CNN's completed native predictions were
rescored, without target fitting. A regression test checks perfect three-class
COCO scoring with these noncontiguous category IDs. Resume metadata records
the evaluator and runner source hashes.

The matched-input deployment experiment completed 18 runs: three models, two
platform/backend stacks, three fresh processes per case and 120 timed frames
per run. Input content/order and the shared plan hash match across every run.
Every run contains an energy measurement. A post-run device-identity check
confirmed CUDA's selected device and NVML index 3 refer to the same GPU UUID.
NVML GPU-device energy and Orin named host rails have different scopes, so no
cross-platform energy ratio is reported.

Command: `.venv/bin/python -m pytest tests/test_rgbt_tiny_compare.py tests/test_long_kaist_analysis.py tests/test_kaist_video.py -q`

Result: **13 passed**. New and modified Python entry points compile, and the
working-tree whitespace check passes. The November 19 full-suite record remains
in VALIDATION_20261119.md; its four missing-external-asset failures were not
reclassified as passing tests.

The updated review report and presentation were exported to PDF; the report
was also exported to editable Word. Sample report and slide pages were rendered
and visually inspected. The presentation contains both original GIF assets,
verified inside its ZIP package. A sample RGBT-Tiny native-coordinate person
overlay was visually checked across visible/thermal images; this does not
establish independent sensor calibration.

ApproxDet's upstream model archive was downloaded, hashed and its seven regular
model assets extracted into an isolated artifact directory. No upstream model
inference or scheduler ran. Its process-wide contention cleanup needs replacement
with owned-process cleanup before shared-host execution; compatible runtime,
original-case validation and multimodal profiling remain open.
