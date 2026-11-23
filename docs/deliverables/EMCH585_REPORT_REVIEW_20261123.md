# Dynamic resolution control for RGB thermal detection

Review draft updated November 23, 2026. Instructor scope, submission requirements and final claims remain to be agreed.

## Research question and contribution

Can causal input-resolution control improve the accuracy, timeliness and energy cost of RGB–thermal detection under changing content and compute availability? The implemented contribution is a reproducible evaluation system with common detector adapters, native-coordinate scoring, explicit streaming arrivals, resource interventions, controller accounting and recorded power. Current measurements show useful reductions relative to some fixed settings, but no consistent adaptive-policy advantage across all three detector families and the observed fixed-resolution alternatives.

## Models and data

All fusion detectors consume B,G,R,IR. YOLO26n uses the existing multi-scale project checkpoint; Faster R-CNN and RetinaNet use frozen validation-best checkpoints from provisional two-hour training runs. Training budgets differ, so these are checkpoint comparisons rather than a matched-budget architecture study.

| Dataset | Paired images used | Purpose | Split and scope |
|---|---:|---|---|
| M3FD | 4,200 total; 419 test | Source training and held-out accuracy | Sequence-grouped 2,940/841/419 train/validation/test |
| KAIST | 16,873 | Frozen video transfer across three day and three night sequences | Full versions of six selected official test sequences; earlier prefixes were already examined |
| RGBT-Tiny | 46,701 total; 12,590 test | Frozen transfer and tiny-object comparison | Official 85 training / 30 test sequences; no target fitting |
| Local FLIR | 120 recorded indoor pairs | Deployment cost, repetition and switching | Unannotated; repeated frames do not establish accuracy or independent scene coverage |

RGBT-Tiny stores two modality frames per pair; the local release has 34,111 training pairs and 12,590 test pairs. Its native release includes modality-specific boxes. The comparison uses original visible-coordinate annotations with paired thermal inputs. Dataset counts and official split membership are audited from the local release. Eight DJI filename prefixes occur in both official partitions; this suggests possibly related captures, not proven frame leakage. Future target training should verify capture identity before selecting validation/test units.

## Deployment and measurements

Batch size is one. Jetson AGX Orin uses TensorRT 10.3 strict FP32 GPU engines and retains CPU preprocessing. The workstation uses an RTX 6000 Ada and native FP32 PyTorch. Both additional-family TensorRT exports passed sampled prediction agreement gates before timing. Backends and preprocessing contracts are recorded; cost differences are not attributed solely to hardware.

Accuracy uses pycocotools COCO AP over IoU .50:.95, AP-small and person recall at IoU .5/confidence .25. The streaming replay runs nominal 20 Hz arrivals, retains the latest due input and counts skipped arrivals. Causal scoring uses only the latest completed boxes at each arrival, without tracking or future results. Service time, completion age and display age are separate metrics. The 100 ms service budget differs from the 50 ms arrival period.

Named-rail energy includes shared-host browser/camera load. Both joules per source arrival and joules per completed inference are reported; neither is wall-plug or isolated model energy. Model loading, profiling, warmup and common engine builds are recorded separately.

## M3FD held-out accuracy

| Frozen checkpoint | Six-class AP % | AP50 % | AP-small % |
|---|---:|---:|---:|
| YOLO26n | 41.21 | 63.58 | 14.58 |
| Faster R-CNN | 40.16 | 65.18 | 18.38 |
| RetinaNet | 30.37 | 49.02 | 4.06 |

All models were evaluated on the same 419 grouped test pairs at a 576-pixel long side. These six-class scores must not be directly subtracted from person-only KAIST scores or the three-class transfer scores below.

## KAIST fixed and adaptive comparisons

All 162 cases completed: three models × six sequences × eight policies plus a fixed576 constant-thread control. The 16,873 unique pairs generate 455,571 source-arrival evaluations, of which 280,026 were processed and 175,545 skipped. The primary CPU preprocessing-thread trace is 4→1→4, within fixed process affinity; this is not a GPU-contention experiment.

| Model | Policy | Sequence-macro causal person AP % | Mean service ms | Dropped arrivals % | J/source arrival |
|---|---|---:|---:|---:|---:|
| YOLO26n | fixed:480 | 2.742 | 44.83 | 0.05 | 0.999 |
| YOLO26n | fixed:576 | 1.318 | 50.85 | 2.66 | 1.001 |
| YOLO26n | sequence | 1.558 | 47.46 | 0.09 | 1.001 |
| YOLO26n | guarded | 1.546 | 47.74 | 0.15 | 1.000 |
| Faster R-CNN | fixed:480 | 0.402 | 131.29 | 61.93 | 1.405 |
| Faster R-CNN | fixed:576 | 0.408 | 139.18 | 64.08 | 1.555 |
| Faster R-CNN | sequence | 0.377 | 137.40 | 63.61 | 1.570 |
| Faster R-CNN | guarded | 0.358 | 130.09 | 61.57 | 1.424 |
| RetinaNet | fixed:480 | 0.692 | 97.29 | 48.97 | 1.316 |
| RetinaNet | fixed:576 | 0.656 | 109.88 | 54.78 | 1.511 |
| RetinaNet | sequence | 0.645 | 108.26 | 54.06 | 1.527 |
| RetinaNet | guarded | 0.673 | 97.48 | 49.03 | 1.306 |

Daytime sequence-macro AP exceeds nighttime AP in every tested model/policy combination. YOLO fixed480 has the highest observed person AP (2.742%); its frozen sequence controller reaches 1.558%. Faster R-CNN and RetinaNet have weak absolute transfer accuracy and substantial source drops. Guarded RetinaNet reduces cost relative to fixed576, but the frozen sequence policy has no consistent advantage over the observed fixed alternatives.

Accuracy aggregates equally across sequences; latency/age percentiles pool actual samples. Paired differences use 1,000 bootstrap resamples of the six matched sequence units. With one run per case and six selected sequences, those intervals describe sequence-mix sensitivity, not repeated-run variability or broad environmental robustness.

[Detailed KAIST analysis and figures](../../results/processed/kaist_long_video_20261118/analysis_20261119/REPORT.md).

## RGBT-Tiny comparison with M3FD

The same frozen M3FD-trained checkpoints are evaluated at fixed576 on the full RGBT-Tiny test split. Both datasets are scored on bus, car and person only; original M3FD predictions are rescored to that subset. Cyclist is not treated as motorcycle. Ship, drone, plane and other unmatched classes are outside this comparison, while all test images remain included as positives or negatives.

| Model | M3FD shared-class AP % | RGBT-Tiny shared-class AP % | M3FD AP-small % | RGBT-Tiny AP-small % |
|---|---:|---:|---:|---:|
| Faster R-CNN | 51.904 | 0.114 | 20.054 | 0.239 |
| RetinaNet | 43.793 | 0.013 | 4.740 | 0.032 |
| YOLO26n | 54.730 | 0.095 | 17.338 | 0.107 |

Comparison status: all three models complete.

In the shared-class test annotations, 80.26% of RGBT-Tiny boxes have area below 16² native pixels, versus 8.25% in M3FD. Native dimensions and resized input footprints differ. Object size, viewpoint, environment and annotation conventions all change, so the size difference alone is not established as the cause of the transfer gap. [Detailed class metrics and scale audit](../../results/processed/rgbt_tiny_comparison_20261123/REPORT.md).

These are frozen cross-dataset transfer results, not evidence that one dataset is intrinsically harder or a comparison of models trained on each dataset. COCO IoU AP is distinct from RGBT-Tiny’s SAFit metric. Residual cross-camera parallax, object scale and annotation conventions can affect transfer. Test results are not used to choose a new policy.

## Matched input deployment costs

A separate fixed576 comparison uses the same 120 predecoded indoor pairs, three fresh processes per model/platform, and four pinned CPU threads. Both input hashes and ordering match. These are short deployment-stack comparisons: Orin TensorRT and RTX 6000 Ada PyTorch differ in backend as well as hardware.

| Platform | Model | Mean service ms | Mean run p95 ms | J/frame | Energy scope |
|---|---|---:|---:|---:|---|
| orin | Faster R-CNN | 95.64 | 115.64 | 2.942 | Orin named-rail host sum; shared services included |
| orin | RetinaNet | 58.15 | 75.54 | 2.009 | Orin named-rail host sum; shared services included |
| orin | YOLO26n | 26.19 | 30.19 | 0.511 | Orin named-rail host sum; shared services included |
| workstation | Faster R-CNN | 8.89 | 9.65 | 2.312 | GPU device NVML total-energy delta |
| workstation | RetinaNet | 14.64 | 15.39 | 3.896 | GPU device NVML total-energy delta |
| workstation | YOLO26n | 4.19 | 5.53 | 0.447 | GPU device NVML total-energy delta |

Workstation NVML measures GPU-device energy; Orin rails include host/background load. These scopes do not support a cross-platform energy-efficiency ratio. Prefetch, loading and warmup are excluded, so the table is not end-to-end camera latency. The 120-frame repetitions do not establish stable p99 or unseen-scene behavior. [Protocol and validation](../../results/processed/platform_comparison_20261123/REPORT.md).

## Related adaptive systems

[ApproxDet](https://arxiv.org/abs/2010.10754) and [Chanakya](https://www.microsoft.com/en-us/research/publication/chanakya-learning-runtime-decisions-for-adaptive-real-time-perception/) already study content/resource-aware inference. Local deadline, AIMD and guarded controls are diagnostic implementations, not reproductions of those published systems. The preserved upstream environments require older dependencies and original model/data assets; faithful execution and a documented multimodal adaptation remain open.

## Remaining evaluation and paper direction

- Measure longer live-camera sessions with verified synchronization, controlled changes and human-checked labels for accuracy claims.
- Extend the short matched-input platform measurements to independent scenes and longer runs; cross-check power with an external meter where available.
- Reproduce at least one published adaptive baseline under declared comparable capabilities and full overhead accounting.
- Investigate why feedback fails under transfer before designing reliability-aware probing or fallback control. Confirm any new method on untouched data.
- Agree on instructor scope, submission requirements and success margins. The evidence supports an empirical system study; it does not yet establish a new state-of-the-art controller.

## Evidence and reproduction

- [Project checklist](../EMCH585_PROJECT_TODO.txt)
- [GPU export and numerical validation](../FAMILY_TENSORRT.md)
- [KAIST video protocol](../KAIST_VIDEO_PROTOCOL.md)
- [M3FD held-out report](../../results/processed/heldout_families_20261116/REPORT.md)
- [RGBT-Tiny official release](https://github.com/XinyiYing/RGBT-Tiny) and [paper](https://arxiv.org/abs/2406.14482). Existing data-access terms remain applicable; the repository does not establish a public-domain dataset license.

The dated November 13 drafts remain preserved. This review draft incorporates the November 16 training/deployment and November 19 full-video analysis.
