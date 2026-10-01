# Dynamic-Resolution Control for RGB/IR Early-Fusion Object Detection (M³FD)

CSCE 585 course project. A four-channel (B,G,R,LWIR) early-fusion YOLO26n is served at a per-image
input resolution of 320, 480 or 640 px. A cheap Scene Analyzer and a depth-3 decision-tree
controller choose that resolution. The project measures accuracy, latency and energy against fixed
resolutions, a matched random control and a per-image oracle. See [`PROPOSAL.md`](PROPOSAL.md).

## Layout

| Path | What it is |
|---|---|
| `prepare_m3fd_fourchannel.py` | 4-channel TIFFs + **sequence-grouped** 70/20/10 split (`splits/`) |
| `fusion/stem.py`, `fusion/trainer.py` | COCO stem inflation to B,G,R,IR; Ultralytics trainer with colour aug on 4-ch input and discrete {320,480,640} multi-scale |
| `train_early_fusion.py`, `configs/*.yaml` | One entry point per configuration (B0, B1, B2, specialists, MS) |
| `fusion/preprocess.py` | The single GPU inference path shared by evaluation and benchmarking |
| `predict_cache.py`, `evaluate.py` | Per-image detection cache; pycocotools scoring of any per-image policy |
| `generate_oracle_labels.py` | Per-image oracle (B5) and matched random control (B4) |
| `scene_analyzer.py`, `controller_heuristic.py` | Scene features (+ microbenchmark), policies, CART controller (H) |
| `bench_latency_energy.py` | Batch-1 harness: per-stage CUDA-event timing, p50/p95/p99, peak memory, NVML energy; `--cpu-threads` constrained mode |
| `analysis.py` | Figures F1–F4 and tables T0/T1/A1 from `results/` |
| `run_m2.sh` | Milestone-2 pipeline after training |
| `tests/` | Stem equivalence, preprocessing vs Ultralytics, per-image AP vs pycocotools |

## Environment

Workstation: 4 × RTX 6000 Ada, Xeon w9-3495X, Linux 6.17. Actual toolchain: **PyTorch 2.10.0 / CUDA 12.8 /
Ultralytics 8.4.14**, Python 3.10. The proposal lists 2.5.1 / 12.1 / 8.4.79; those versions were not the
ones available. Milestone-2 runs used the `YOLO_Workspace` conda env as-is. It lacks scikit-learn and
nvidia-ml-py, so the controller uses a built-in CART and energy uses a ctypes NVML binding. Neither
package is required.

Data: `/mnt/12TB_Drive/whisenaj/Datasets/M3FD_Detection` (source) → `/mnt/12TB_Drive/whisenaj/Datasets/M3FD_4ch` (derived).

## Reproduce

```bash
python prepare_m3fd_fourchannel.py                       # split + TIFFs (~6 min)
for c in b0_rgb_640 b1_ir_640 b2_fusion_640 spec_fusion_480 spec_fusion_320 ms_fusion; do
  python train_early_fusion.py --config configs/$c.yaml --device 1; done   # ~95 min each
./run_m2.sh 1 1                                          # caches, oracle, controller, B4, benchmarks, figures
python analysis.py                                       # F1 (main result) from committed results/ only
```

Tests: `python tests/test_stem.py`, `python tests/test_per_image_ap.py`, `python tests/test_preprocess.py`.

**Evaluator cross-check (B2, seed 42, test @640).** Ultralytics `val` gives mAP50-95 0.4427. The same
detections from our GPU path, scored with Ultralytics' AP code, give 0.4369. That 0.6-point pipeline
difference comes from antialiased bilinear resize vs cv2 INTER_AREA; mAP50 agrees to 0.001. Scored with
pycocotools (what we report), the same detections give 0.4274; that 1.0-point gap is the metric definition
(interpolation and maxDets=100). An earlier version assumed 1024×768 for every frame and under-reported by
5.8 points; that was caught by this check and fixed.

## Design refinements relative to the proposal

1. **Grouped split.** M³FD frames come from sequences, so under the proposal's stratified random split
   **48.2% of test frames have a near-duplicate in train** (thumbnail correlation > 0.95). Frames are
   grouped by connected components of thumbnail correlation > 0.85 (767 groups), and whole groups are
   split. The result is 2940/841/419 frames with 0% near-duplicate leakage (`splits/m3fd_grouped_seed42.json`,
   visual check in `splits/group_contact_sheet.jpg`).
2. **Native resolution is mostly 1024×768**, not 640×480. However, 274 of the 4,200 frames are smaller
   (400×280 to 880×520; 76 of the 419 test frames), and every box conversion uses per-frame sizes
   (`splits/image_sizes.json`). Small objects are area < 32² at native resolution: 14,073 of 34,407
   boxes (40.9%).
3. **Ultralytics 4-channel caveats handled:**
   - the pretrained stem is dropped on shape mismatch, so it is inflated in B,G,R,IR order;
   - RandomHSV skips non-3-channel images, so it is applied to channels 0–2;
   - `multi_scale` is continuous, so a discrete {320,480,640} draw is used per batch.
4. **Scene Analyzer.** The spectral feature is measured on native-resolution patches, because a 64-px
   thumbnail cannot contain frequencies above the 320-px Nyquist limit. M³FD has no scene tags, so
   day/night/adverse is a documented proxy (`scene_analyzer.SCENE_THRESH`, contact sheets in `splits/`).
5. **Oracle labels are computed on val, not train.** The detector memorises its training images.
6. **YOLO26 is NMS-free.** The last stage is timed as "postprocess" (confidence filter, rescale, D2H).
   The harness uses the deployment threshold of 0.25; mAP uses the standard conf=0.001 cache.
7. **No Orin yet.** The proposal's fallback constrained operating point (4 CPU threads, compute-bound) is
   reported alongside the GPU.

## Provenance

Each training run appends to `results/experiment_log.csv` (git SHA, source-code hash taken at launch, config,
seed, device) and snapshots its source into `runs/<run>/code/`. Each benchmark JSON in `results/raw/` records
the code hash, GPU state before and after (clocks, temperature, co-resident processes), torch/CUDA versions,
per-frame latencies and the resolution chosen per frame.

## Controllers (Milestone 2)

All controllers are fit on train or val and selected on val, then scored once on test. Every variant tried is
kept in `results/processed/policy_sweep_*.csv` and plotted in F6.

| Controller | Result (test, seed 42) |
|---|---|
| **v1: depth-3 classification tree** on oracle labels (`controller_heuristic.py`) | CV accuracy 0.41 = majority baseline. Scene features barely correlate with per-image resolution gain (\|ρ\| ≤ 0.16). |
| **Probe cascade** (`probe_features.py`, `controller_fit.py --kind probe`) | 320-px detections do carry the signal (ρ = 0.39), but the probe costs 44% of a 640 pass on CPU. With 192/256/320-px probes, every cascade loses to fixed-640 or to random. |
| **v2: cost-aware policy tree**, light features, fit on train (`controller_fit.py --kind scene --fit-split train --features lum_mean lum_std`) | Rule: `lum_mean <= 142 → 480 px, else 640 px`. AP 0.405 vs 0.403 ± 0.002 for a cost-matched random mix. Online CPU p95 31.9 ms vs 31.1 ms for fixed-640. |

The oracle (ε=0.05) reaches AP 0.423 at a mean CPU latency of 18.6 ms, vs 0.415 at 26.6 ms for fixed-640. The
headroom is real; scene-level control captures about +0.3 AP of the oracle's +3.5.

Controllers are fit against a frozen, pinned CPU cost model (`results/processed/cpu_cost_model.json`).
Deployment latency is re-measured in a single session by `bench_session.sh`: GPU 1 idle, CPU pinned to
cores 104–107 on this shared host.

Slides are built by `python3 scripts/build_m2_slides.py`, which writes `../Project_Presentation_whisenaj_M2.pptx`
and leaves the original deck untouched.

## Progress notes

- **2026-09-30** — Data prep, grouped split, and 4-channel pipeline done. Stem, preprocessing and
  evaluator tests pass. Ten detectors trained: six seed-42 configurations, plus seeds 123/456 for B2 and MS.
  - The evaluator cross-check found a frame-size bug (5.8 pts); it is fixed.
  - Controllers v1, v2 and the probe cascade were evaluated.
  - Benchmarks were run in one pinned session.
  - The Milestone-2 deck is built.
  - Orin not yet brought up.
