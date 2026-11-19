# Jetson TensorRT deployment used on November 13, 2026

This installation uses JetPack/L4T R36.4.7, CUDA runtime 12.6 and the existing
system TensorRT 10.3.0.30. CPU PyTorch remains installed for exact preprocessing
and the existing dashboard. TensorRT executes the model on the Orin GPU through
typed CUDA-runtime calls; CUDA-enabled PyTorch is not required for this backend.

The matching NVIDIA Python binding package was downloaded and extracted locally:

```bash
mkdir -p /media/imsel/4TBSSD/EMCH585_Artifacts/dependencies
cd /media/imsel/4TBSSD/EMCH585_Artifacts/dependencies
apt-get download 'python3-libnvinfer=10.3.0.30-1+cuda12.5'
dpkg-deb -x python3-libnvinfer_10.3.0.30-1+cuda12.5_arm64.deb tensorrt-python
```

No system package was replaced and no sudo installation was needed. Verify the
installed system version before reproducing this on another JetPack release;
bindings must match the native TensorRT runtime.

From the project root:

```bash
bash scripts/gpu_python.sh -c 'import tensorrt; print(tensorrt.__version__)'
.venv/bin/python -m experiments.export_tensorrt --output /NEW/EMPTY/ENGINE/DIR
bash scripts/gpu_python.sh -m experiments.validate_tensorrt \
  --engine /NEW/EMPTY/ENGINE/DIR/engine.json \
  --manifest captures/research_flir_pilot/manifest.json --output /NEW/gate.json
```

The validated strict FP32 engine is at:
`/media/imsel/4TBSSD/EMCH585_Artifacts/project_trt_fp32_strict/engine.json`.
It disables TF32 and supports dynamic spatial shapes. Build logs, ONNX, engine,
checkpoint/engine/graph hashes and engine-build wall time are preserved there.
The earlier TF32-enabled engine is a separate build, not used for final results.

Example GPU benchmark:

```bash
bash scripts/gpu_python.sh -m experiments.run \
  --manifest captures/research_flir_pilot/manifest.json \
  --output /NEW/gpu_result.json \
  --weights /media/imsel/4TBSSD/EMCH585_Artifacts/project_trt_fp32_strict/engine.json \
  --adapter experiments.tensorrt_backend:create --device tensorrt \
  --policy sequence --budget-ms 50 --thread-trace 4,4,4
```

GPU work is synchronized before returning. CUDA-event GPU span is distinct from
host shape/transfer cost and CPU preprocessing/postprocessing. Native prediction
matching at confidence .10 passed 32 sampled frame/resolution cases, using
IoU >=.90, confidence difference <=.02 and equal class-matched counts. That gate
is not a substitute for full labeled AP equivalence or registration verification.

The CPU browser demonstration remains on port 5000; the GPU research runs are
separate artifacts. Study orchestration temporarily pauses dashboard inference
and restores its previous state in `finally`. The C++ acquisition service stays
available on port 8080.

NVIDIA documentation:
[TensorRT installation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/install-pip.html),
[Jetson power/performance](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html).
