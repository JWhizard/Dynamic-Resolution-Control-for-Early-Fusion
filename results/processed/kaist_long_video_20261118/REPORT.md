# Full-sequence KAIST replay completion — November 19, 2026

The completed [accuracy, streaming, latency and energy analysis](analysis_20261119/REPORT.md)
compares all fixed/adaptive policies by model and day/night, including paired
sequence-bootstrap intervals and observed fixed-resolution frontiers.

All 162 planned cases completed: three frozen detectors, six full
day/night sequences, eight fixed/adaptive configurations under CPU preprocessing
thread changes and one fixed576 constant-thread control per detector/sequence.
There are 16,873 unique paired images, replayed across
455,571 source-frame evaluations; 280,026
were processed and 175,545 were skipped by latest-due-input replay.

`cases.csv` contains per-sequence causal and skipped-empty COCO person AP,
service latency, controller cost, budget misses and named-rail energy. AP values
are fractions. These are saved scores, not newly recomputed pooled COCO metrics.
`validation.json` records verified result hashes, annotation/manifest hash links,
saved-score consistency, source-frame accounting and power read errors.

The study extends previously evaluated 600-frame prefixes. It is frozen
M3FD-to-KAIST transfer, not a new untouched confirmatory test or official KAIST
benchmark score. Labels use person class mapping and crowd-ignore handling.
Separate day/night recordings do not establish natural environment transitions.
There is one run per case. Named-rail energy includes background host activity.
The 100 ms service budget differs from the 50 ms source arrival interval.

Run-level outputs, predictions and traces remain at `/media/imsel/4TBSSD/EMCH585_Artifacts/kaist_long_video_20261118`.
Source snapshots and the launch protocol were preserved there before scoring.
Reproduce this export with `python -m experiments.summarize_long_kaist
--raw /path/to/study --output /NEW/summary` (on one shell line).
The linked analysis reports sequence-macro and worst-sequence accuracy, pooled
latency/age percentiles and resource controls. Dataset-pooled COCO AP and natural
environment-transition studies are not claimed.
