# KAIST held-out video protocol — November 16, 2026

## November 18 full-sequence extension (runs completed November 19)

The November 16 raw study has a completion marker for all 90 cases, although
its aggregate report and the main checklist had not been updated. Its 30-second
prefixes are preserved. The extension uses the full versions of the same six
sequences, selected by official IDs rather than label content:

| Sequence | Environment | Paired frames | Duration at nominal 20 Hz |
|---|---|---:|---:|
| set06/V000 | day | 1,523 | 76.15 s |
| set07/V000 | day | 2,544 | 127.20 s |
| set08/V000 | day | 2,578 | 128.90 s |
| set09/V000 | night | 3,500 | 175.00 s |
| set10/V000 | night | 4,708 | 235.40 s |
| set11/V000 | night | 2,020 | 101.00 s |

Total: 16,873 unique paired frames, about 14 minutes. This includes the
previously scored prefixes, so it is an extended frozen-transfer evaluation,
not an entirely untouched confirmatory test. No target tuning is authorized.

All three validated TensorRT detectors use fixed 480/544/576/640, sequence,
guarded, deadline EMA and AIMD policies under 4→1→4 preprocessing threads,
plus fixed576 4→4→4 controls: 162 cases, 455,571 source-frame evaluations.
All 162 cases completed on November 19: 280,026 processed arrivals and 175,545
dropped arrivals. Compact per-sequence results and completion verification are
in `results/processed/kaist_long_video_20261118`. Pooled and transition analysis
remains follow-up work.

The original 100 ms service budget, 20 Hz arrivals, person mapping, ignore
handling and causal scoring conventions remain in force. Resource phases
follow each sequence's thirds; they are not natural environmental transitions.
Each case has one run, so these runs do not provide repeated-run confidence
intervals. Sequence lengths vary; dataset-level reporting must explicitly
distinguish sequence macro means from frame-weighted or pooled metrics.

Launch entry point: `experiments/start_long_kaist.py`. The driver checks the
publisher MD5 of the existing workstation archive, extracts only selected
regular files into a new directory, copies and verifies all selected file
hashes on Jetson, creates manifests with globally unique image IDs, freezes
the job matrix and launches fresh-process inference/scoring cases. Models
and policy settings are inherited from the earlier frozen study. This does
not launch training or interrupt existing workstation training jobs.

Study directory:
`/media/imsel/4TBSSD/EMCH585_Artifacts/kaist_long_video_20261118`

Monitor `status.json` there for preparation/overall state, then
`runs/progress.json` for completed cases. The external launcher log is
`/media/imsel/4TBSSD/EMCH585_Artifacts/kaist_long_video_20261118_launcher.log`.
`runs/COMPLETED.json` is written only after all cases finish. Long-study analysis
uses `experiments/analyze_long_kaist.py`, which handles unequal sequence lengths,
sequence-macro accuracy, pooled latency/age tails and paired sequence bootstrap.
Its report is in `results/processed/kaist_long_video_20261118/analysis_20261119`.
The older `report_kaist_video.py` remains specific to the 600-frame study.
Preserve all raw outputs.

The sections below describe the original November 16 prefix study.

The full official archive was located on the authorized workstation and its
39.0 GB file verified against the publisher MD5:
`fcfc7cfcc1237ad414af64b972f8dc97`. The original source remains unchanged at
`/mnt/12TB_Drive/whisenaj/Datasets/KAIST/kaist-cvpr15.tar`.
The archive has exactly 38,999,029,760 bytes (about 36.3 GiB).

Selection was frozen before model scoring: the first 600 consecutive dense
frames from the lexicographically first sequence of each official test set
06–11. These are set06/V000 through set11/V000: three day and three night
prefixes, 3,600 unique paired frames, nominal 20 Hz and 30 seconds per sequence.
This is a bounded subset of the full test release, not the whole benchmark.

Selected JPEG, XML and split-list files were extracted directly from the verified
archive into a new directory. Archive code and links were not extracted or run.
All 10,805 selected files were checked by SHA-256 after copying to Jetson.
Actual RGB/thermal dimensions and annotation headers agree at 640×512.
Source train/test membership is disjoint, and sparse every-20th-frame lists
are not used as native-rate video.

Source and attribution: Soonmin Hwang, Jaesik Park, Namil Kim, Yukyung Choi,
In So Kweon, *Multispectral Pedestrian Detection: Benchmark Dataset and
Baselines*, CVPR 2015. [Official release](https://github.com/SoonminHwang/rgbt-ped-detection).
Dataset license: CC BY-NC-SA 4.0. Publisher README/license snapshots are retained.

## Annotation and model contracts

Frozen models are the existing M3FD-trained YOLO26n multi-scale, Faster R-CNN
and RetinaNet checkpoints and their validated strict-FP32 TensorRT engines.
No target-dataset training, threshold selection or controller fitting occurs.
Additional model/backend checks on the first frame of all six new sequences
compare 24 cases per new detector across 480/544/576/640 long-side inputs.

The model's zero-based class 4, named People in M3FD, maps to KAIST person.
All other detector classes are outside this accuracy benchmark. KAIST's
annotations named people, person?, cyclist, or marked difficult are ignore
regions. The similarly named M3FD People class does not make KAIST crowd
annotations valid individual pedestrians.

The sanitized XML x/y/w/h geometry is clipped to actual image bounds and
converted to COCO annotations. All clear person sizes and occlusions remain
in scope. This is not official KAIST reasonable-set/log-average-miss-rate
scoring. Pixel registration is publisher-provided; native-box overlay examples
were inspected, but residual misalignment is not independently certified.

## Frozen case matrix

- Three GPU detectors, six independent sequence prefixes.
- Fixed 576, frozen sequence controller, deadline EMA and guarded policies.
- Primary condition: Torch CPU preprocessing threads 4→1→4 across the three
  source thirds, at approximately 10 and 20 seconds.
- Controls: fixed 576 with constant 4→4→4 threads for every model/sequence.
- Total: 72 primary cases and 18 controls, 90 cases and 54,000 input-frame
  evaluations. The same 3,600 unique pairs are reused; these are not 54,000
  independent images or separate environments.
- One run per case; randomized order with seed 20261116.
- Four allowed CPU cores, 8–11, and one inter-op thread. The intervention limits
  Torch parallelism, not physical core ownership or GPU clock/capacity.
- Native nominal 20 Hz arrival replay, processing only the latest due input.
  The inherited 100 ms service budget is distinct from the 50 ms input period.
- Three warmup calls per resolution and one idle second are outside the timed
  input-processing window. Browser/camera services remain active.

## Measurements and interpretation

Every source frame is scored. Skipped-frame AP assigns skipped inputs empty
detections. Causal time-aligned AP uses only the latest completed boxes available
at each source arrival, with no future predictions, tracking or interpolation.
State and box reuse cannot cross sequence boundaries.

Completion age measures a processed frame's completion relative to its source
arrival. Display age measures the age of the reused boxes at an input arrival.
Service latency, its tails, controller overhead, drops, resolution choices,
phase setup, temperatures, clocks and named rail traces are retained separately.

Energy has two denominators: joules per source arrival and joules per completed
inference. Both include shared-host background activity and are neither isolated
model energy nor wall-plug power. A method cannot claim lower energy merely by
hiding that it processes fewer source frames.

Pooled COCO scores recompute a single precision/recall curve over all six
independent sequences. Day/night pools, sequence macro means and worst-sequence
scores expose differences in pedestrian density. One run per case gives no
repeated-run confidence interval. Shared thermal/I/O/load drift can confound
small differences even with matched fixed-policy controls.

Stable person-presence changes from ground truth are post-hoc content proxies,
never controller inputs. They are not independently annotated environment cuts.
Separate day/night recordings are not concatenated into fake natural transitions.

## Locations and reproduction

Selected data:
`/media/imsel/4TBSSD/EMCH585_Artifacts/datasets/kaist/heldout_video_20261116`

Labeled manifests:
`/media/imsel/4TBSSD/EMCH585_Artifacts/datasets/kaist/heldout_eval_20261116`

Plan, exact source snapshots, gates and raw results:
`/media/imsel/4TBSSD/EMCH585_Artifacts/kaist_video_20261116`

```bash
.venv/bin/python -m experiments.run_kaist_video \
  --plan /path/to/plan.json --output /NEW/output
```

This entrypoint checks engine gates and hashes, runs fresh processes, scores
all input frames and preserves completed output directories. The report is
generated by `experiments/report_kaist_video.py` after every case completes.
