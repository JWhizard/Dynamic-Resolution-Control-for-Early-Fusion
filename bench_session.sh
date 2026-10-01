#!/usr/bin/env bash
# All latency/energy measurements in ONE session so policies are compared under the same machine state.
# GPU: an idle GPU (the harness records co-resident processes). CPU: 4 threads pinned to 4 dedicated cores
# (taskset) on a shared 112-thread host; the fixed resolutions are re-measured here, never reused from
# earlier sessions. Usage: ./bench_session.sh <gpu> [cores, default 104-107]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}; GPU=${1:-1}; CORES=${2:-104-107}
export CUDA_DEVICE_ORDER=PCI_BUS_ID OPENCV_LOG_LEVEL=ERROR PYTHONUNBUFFERED=1
P=results/processed
POLICIES=(
  fixed:320 fixed:480 fixed:640
  heuristic:$P/controller_scene_fittrain_lum_mean-lum_std.json  # H (v2): cost-aware tree on light features, fit on train
  file:$P/policy_random_costmatched_scene_fittrain_lum_mean-lum_std_s0_test.json  # its cost-matched random control
  heuristic:$P/controller_eps0.05.json                          # H (v1): classification tree (negative result)
  file:$P/policy_random_matched_s0_test.json                    # B4 for v1
  file:$P/policy_oracle_eps0.0_test.json file:$P/policy_oracle_eps0.02_test.json
  file:$P/policy_oracle_eps0.05_test.json file:$P/policy_oracle_eps0.1_test.json
)
$PY scene_analyzer.py --device cuda:$GPU --bench | tee $P/analyzer_bench_full.json
$PY scene_analyzer.py --device cuda:$GPU --bench --needed lum_mean | tee $P/analyzer_bench_light.json
for pol in "${POLICIES[@]}"; do
  $PY bench_latency_energy.py --run ms_fusion_seed42 --policy "$pol" --gpu "$GPU" > /dev/null
  taskset -c "$CORES" $PY bench_latency_energy.py --run ms_fusion_seed42 --policy "$pol" --cpu-threads 4 --reps 3 > /dev/null
  echo "done $pol"
done
$PY bench_latency_energy.py --run b0_rgb_640_seed42 --view rgb --policy fixed:640 --gpu "$GPU" > /dev/null
$PY bench_latency_energy.py --run b1_ir_640_seed42  --view ir  --policy fixed:640 --gpu "$GPU" > /dev/null
