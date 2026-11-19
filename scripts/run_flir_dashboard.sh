#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv/bin/python" live_dashboard.py \
  --flir-server "${FLIR_SERVER:-http://127.0.0.1:8080}" \
  --host 0.0.0.0 --port "${FUSION_PORT:-5000}" --device cpu --cpu-threads 4 "$@"
