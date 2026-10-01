"""Freeze the CPU cost model used to fit controllers from pinned fixed-resolution benchmarks.

    taskset -c 104-107 python bench_latency_energy.py --run ms_fusion_seed42 --policy fixed:R --cpu-threads 4 --reps 3 [--tag T]
    python scripts/freeze_cost_model.py [--res 192 256 320 400 480 544 576 640] [--out cpu_cost_model.json]

For each resolution the untagged record is used if present, otherwise the first tagged one (e.g. _frontier).
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "raw"
ap = argparse.ArgumentParser()
ap.add_argument("--run", default="ms_fusion_seed42")
ap.add_argument("--res", type=int, nargs="+", default=[192, 256, 320, 480, 640])
ap.add_argument("--out", default="cpu_cost_model.json")
args = ap.parse_args()
cm = {}
for r in args.res:
    f = RAW / f"bench_{args.run}_fixed-{r}_test_cpu4.json"
    if not f.exists():
        f = sorted(RAW.glob(f"bench_{args.run}_fixed-{r}_test_cpu4_*.json"))[0]
    b = json.loads(f.read_text())
    assert b["cpu_threads"] == 4
    cm[str(r)] = {"mean_ms": b["latency_ms"]["mean"], "p95_ms": b["latency_ms"]["p95"], "timestamp": b["timestamp"],
                  "source": f.name}
out = {"description": "Frozen CPU cost model used to FIT controllers: 4 threads pinned to dedicated cores, "
                      "3 reps x 419 test frames. Deployment latency is re-measured separately by bench_session.sh.",
       "run": args.run, "platform": "_cpu4", "costs": cm}
(REPO / "results" / "processed" / args.out).write_text(json.dumps(out, indent=1))
print({k: round(v["mean_ms"], 2) for k, v in cm.items()})
