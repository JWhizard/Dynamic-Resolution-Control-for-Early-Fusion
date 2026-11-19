#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${EMCH585_ARTIFACT_ROOT:-/media/imsel/4TBSSD/EMCH585_Artifacts}"
TRT_BINDINGS="${EMCH585_TRT_BINDINGS:-$ARTIFACT_ROOT/dependencies/tensorrt-python/usr/lib/python3.10/dist-packages}"
if [[ ! -d "$TRT_BINDINGS/tensorrt" ]]; then
  echo "Matching TensorRT Python bindings missing: $TRT_BINDINGS" >&2
  echo "See docs/GPU_SETUP.md for the isolated JetPack binding setup." >&2
  exit 1
fi
export PYTHONPATH="$TRT_BINDINGS${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv/bin/python" "$@"
