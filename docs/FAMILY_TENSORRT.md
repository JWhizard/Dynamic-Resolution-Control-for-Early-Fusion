# Additional-family deployment on Jetson AGX Orin

The frozen six-class M3FD Faster R-CNN and RetinaNet checkpoints now have
full detection graphs exported to ONNX and built with native TensorRT 10.3.
They use strict FP32 with TF32 disabled. CPU-only PyTorch remains installed;
typed CUDA-runtime calls execute TensorRT on the GPU without replacing the
dashboard environment or system GPU libraries.

The CNN, classification/regression heads, proposal processing, ROI processing
and NMS execute inside TensorRT. Original Torchvision normalization, resizing,
padding and final restoration to native box coordinates execute on CPU.
Input order is B,G,R,IR; classes are Bus, Car, Lamp, Motorcycle, People, Truck.

Canonical engines on this Jetson:

```text
/media/imsel/4TBSSD/EMCH585_Artifacts/family_trt_20261116/retinanet_safe/engine.json
/media/imsel/4TBSSD/EMCH585_Artifacts/family_trt_20261116/fasterrcnn_safe_fixed/engine.json
```

Each engine.json records checkpoint/ONNX/engine hashes, build time and command.
Earlier export/build attempts are preserved separately and are not benchmark
engines. Rebuild engines on another target and validate again before measuring.

## Export and build

From the project root, use NEW output directories:

```bash
.venv/bin/python -m experiments.export_family_full \
  --weights /path/to/retinanet/best.pt --output /NEW/retinanet
.venv/bin/python -m experiments.build_family_trt --export /NEW/retinanet

.venv/bin/python -m experiments.export_family_full \
  --weights /path/to/fasterrcnn/best.pt --output /NEW/fasterrcnn_export
.venv/bin/python -m experiments.fix_roi_onnx \
  --export /NEW/fasterrcnn_export --output /NEW/fasterrcnn_fixed
.venv/bin/python -m experiments.build_family_trt --export /NEW/fasterrcnn_fixed
```

The export pads results to 300 rows with score-zero padding removed by the
adapter. A negative-score dummy removes Torchvision's empty-NMS conditional
without suppressing valid detections. The Faster R-CNN graph specifies known
seven-class bbox dimensions in two ambiguous empty-ROI reshapes. These helpers
are for these exact six-foreground-class models; no weights or thresholds change.

Input `images` is normalized/padded float32 NCHW; `image_size` is an int64
two-element tensor containing the actual resized, unpadded height/width.
Dynamic image bounds are 1×4×256×320 through1×4×640×640. Validation covers the
deployed 480/544/576/640 long-side choices and multiple input aspect ratios.

## Validate before timing

`family_reference.py` runs the original frozen PyTorch detector on the exact
saved BGRIR arrays. The existing references were generated on workstation CPU
with Torch 2.10/Torchvision0.25, retaining the original model preprocessing.
`inputs.npz` includes12 grouped M3FD validation pairs, 4 private recorded FLIR
pairs and one zero-image edge case. Keep the private input archive local.

```bash
bash scripts/gpu_python.sh -m experiments.validate_family_trt \
  --engine /path/to/engine.json \
  --reference /path/to/original_reference.json \
  --inputs /path/to/inputs.npz --labels /path/to/labels.json \
  --output /NEW/validation.json
```

The frozen primary gate requires class-aware exact detection counts at
confidence 0.10, IoU≥0.90 and score differences≤0.02; validation-subset AP
differences must be≤0.005 in fractional AP. Both new families passed 68 cases
each. This is sampled numerical agreement, not full-test AP certification.
Low-confidence IoU diagnostics and zero-area boxes are reported separately.

## Measure fixed and adaptive policies

```bash
taskset -c 8-11 bash scripts/gpu_python.sh -m experiments.run \
  --manifest captures/research_flir_pilot/manifest.json \
  --output /NEW/measurement.json --weights /path/to/validated/engine.json \
  --adapter experiments.family_trt_backend:create --device tensorrt \
  --policy sequence --thread-trace 4,4,4 --budget-ms 100 \
  --warmup 3 --idle-seconds 1 --max-frames 120
```

The study plan uses eight policies: fixed 480/544/576/640, sequence, deadline
EMA, AIMD and guarded. Three fresh-process repeats per model/policy are run in
randomized order. The orchestrator checks each engine's numerical gate and hash
before any measurement. It preserves existing output directories.

Measurements include decode/CPU transforms/GPU execution/CPU coordinate work,
controller overhead, service-budget misses, switching and sampled INA3221 rails.
CUDA work is synchronized before the detector returns. GPU-event time is the
execution span, which can include idle gaps during shape synchronization.
Setup and warmup are separate from the timed energy window.

The browser and acquisition services remain active. Named-rail sums include
background load; they are not isolated model power or wall-plug energy.
The 120-pair indoor clip is unannotated, repeated across cases and too short for
strong tail-latency or environmental-generalization claims. Lower cost does not
by itself establish preserved detection accuracy for an adaptive policy.

See `results/processed/family_trt_20261116/REPORT.md` for the completed matrix.
Native binding setup is documented in [GPU_SETUP.md](GPU_SETUP.md).
