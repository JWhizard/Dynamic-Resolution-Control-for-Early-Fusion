# Repository validation — November 19, 2026

Command: `.venv/bin/python -m pytest tests -q`

Result on the Jetson checkout: **61 passed, 4 failed**. The four failures
require external assets absent from this machine:

- `test_bootstrap_ap.py::test_fast_ap_matches_pycocotools`: workstation M3FD labels.
- `test_per_image_ap.py::test_per_image_ap_matches_cocoeval`: workstation M3FD labels.
- `test_preprocess.py::test_matches_ultralytics`: `runs/b2_fusion_640_seed42/weights/last.pt`.
- `test_stem.py::test_stem_equivalence`: a pretrained checkpoint at a workstation-only path.

No test assertions were weakened to accommodate missing assets. The seven
KAIST selection, annotation, scoring and recovery tests passed in this suite.
Python compilation and `git diff --check` also passed.

The full-sequence study completion export independently checks all 162 saved
cases against the planned job matrix, score/result hashes, manifest/annotation
hash links, source-frame accounting and power-read errors. Its verification
record is `results/processed/kaist_long_video_20261118/validation.json`.
This verifies saved evidence, not full-test backend accuracy equivalence or
environmental robustness. See the study report for scope and limitations.
