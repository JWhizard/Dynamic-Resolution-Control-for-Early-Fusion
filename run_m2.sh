#!/usr/bin/env bash
# Milestone-2 pipeline after the six seed-42 runs finish:
#   caches -> E1/A1 -> oracle (B5) -> heuristic (H) -> matched random (B4) -> benchmarks -> figures
# Usage: ./run_m2.sh <gpu-for-inference> <idle-gpu-for-benchmarks>
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
GPU=${1:-1}; BGPU=${2:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID OPENCV_LOG_LEVEL=ERROR PYTHONUNBUFFERED=1
EPS=0.05

if [ -z "${SKIP_CACHE:-}" ]; then  # detection caches (native coordinates; independent of GT)
$PY predict_cache.py --run ms_fusion_seed42        --view fusion --device $GPU
$PY predict_cache.py --run b2_fusion_640_seed42    --view fusion --device $GPU
$PY predict_cache.py --run spec_fusion_480_seed42  --view fusion --device $GPU
$PY predict_cache.py --run spec_fusion_320_seed42  --view fusion --device $GPU
$PY predict_cache.py --run b0_rgb_640_seed42 --view rgb --device $GPU --res 640
$PY predict_cache.py --run b1_ir_640_seed42  --view ir  --device $GPU --res 640
fi

$PY evaluate.py --split test --runs ms_fusion_seed42 b2_fusion_640_seed42 spec_fusion_480_seed42 \
    spec_fusion_320_seed42 b0_rgb_640_seed42 b1_ir_640_seed42
$PY evaluate.py --split test --out results/processed/e1_fixed_test_seeds.csv --runs ms_fusion_seed42 ms_fusion_seed123 \
    ms_fusion_seed456 b2_fusion_640_seed42 b2_fusion_640_seed123 b2_fusion_640_seed456
$PY generate_oracle_labels.py --run ms_fusion_seed42
$PY scene_analyzer.py --device cuda:$GPU --extract
$PY controller_heuristic.py --run ms_fusion_seed42 --eps $EPS
$PY generate_oracle_labels.py --run ms_fusion_seed42 --matched-random results/processed/policy_heuristic_eps${EPS}_test.json

# controllers v2 (cost-sensitive) are fit against a frozen, pinned CPU cost model
for R in 192 256 320 480 640; do
  taskset -c 104-107 $PY bench_latency_energy.py --run ms_fusion_seed42 --policy fixed:$R --cpu-threads 4 --reps 3 > /dev/null
done
$PY scripts/freeze_cost_model.py   # controllers are fit against this frozen cost model
$PY predict_cache.py --run ms_fusion_seed42 --view fusion --device $GPU --res 320 480 640 --splits train
$PY predict_cache.py --run ms_fusion_seed42 --view fusion --device $GPU --res 192 256 --splits val test
$PY controller_fit.py --kind scene --fit-split train
$PY controller_fit.py --kind scene --fit-split train --features lum_mean lum_std   # H v2 (deployed)
$PY controller_fit.py --kind scene
for PR in 192 256 320; do $PY controller_fit.py --kind probe --probe-res $PR; done

# all latency / energy measurements in one session
PY=$PY ./bench_session.sh $BGPU

$PY analysis.py
