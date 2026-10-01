"""Batch-1 streaming latency / memory / energy harness for a resolution policy.

    python bench_latency_energy.py --run ms_fusion_seed42 --policy fixed:640 --gpu 1
    python bench_latency_energy.py --run ms_fusion_seed42 --policy heuristic:results/processed/controller_eps0.05.json --gpu 1
    python bench_latency_energy.py --run ms_fusion_seed42 --policy file:results/processed/policy_oracle_eps0.05_test.json --gpu 1

Per frame (frames pre-decoded into pinned host memory; decode reported separately):
    h2d -> analyze (scene features + controller decision; 0 for fixed/file policies)
    -> resize (GPU resize + letterbox) -> inference -> postprocess (conf filter,
    native-box rescale, D2H of detections; YOLO26 is NMS-free)
timed with CUDA events, plus an end-to-end wall clock. 50 warm-up frames are
discarded; each repetition streams the whole split. Energy is the NVML
total-energy counter delta over each repetition (idle power reported so dynamic
energy can be derived). One JSON record per run goes to results/raw/.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # CUDA index == NVML index

import numpy as np
import torch

from controller_heuristic import make_policy
from fusion.preprocess import load_frame, load_model, postprocess, to_input
from predict_cache import SRC, split_ids
from scene_analyzer import SceneAnalyzer
from train_early_fusion import code_hash, git_sha

REPO = Path(__file__).resolve().parent
STAGES = ("h2d", "analyze", "resize", "inference", "postprocess")


class NVML:
    """Minimal ctypes binding (nvidia-ml-py is not installed in this environment)."""

    def __init__(self, index: int):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        assert self.lib.nvmlInit_v2() == 0
        self.h = ctypes.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.h)) == 0

    def energy_mj(self) -> int:
        e = ctypes.c_ulonglong()
        assert self.lib.nvmlDeviceGetTotalEnergyConsumption(self.h, ctypes.byref(e)) == 0
        return e.value

    def power_w(self) -> float:
        p = ctypes.c_uint()
        assert self.lib.nvmlDeviceGetPowerUsage(self.h, ctypes.byref(p)) == 0
        return p.value / 1000


def gpu_state(index: int) -> dict:
    q = "name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used"
    row = subprocess.check_output(["nvidia-smi", f"--id={index}", f"--query-gpu={q}", "--format=csv,noheader"], text=True)
    procs = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True)
    apps = [p for p in procs.strip().splitlines() if p and not p.startswith(f"{os.getpid()},")]  # not ourselves
    return {"query": q, "values": row.strip(), "compute_apps": apps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--view", default="fusion", choices=["fusion", "rgb", "ir"])
    ap.add_argument("--weights", default="last.pt")
    ap.add_argument("--policy", required=True, help="fixed:R | heuristic:<json> | file:<json>")
    ap.add_argument("--split", default="test")
    ap.add_argument("--gpu", type=int, default=None, help="physical (PCI order) GPU index")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="run on CPU with N threads: a compute-bound constrained operating point (no Orin yet)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--idle-seconds", type=float, default=5.0)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="deployment threshold for the postprocess stage (mAP uses the conf=0.001 cache)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cpu = args.cpu_threads is not None
    assert cpu != (args.gpu is not None), "pass exactly one of --gpu / --cpu-threads"
    if cpu:
        torch.set_num_threads(args.cpu_threads)
        dev, state_before = torch.device("cpu"), {"cpu_threads": args.cpu_threads, "cpu": platform.processor()}
    else:
        dev = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(dev)
        state_before = gpu_state(args.gpu)
        if len(state_before["compute_apps"]) > 0:
            print("WARNING: other processes on this GPU:", state_before["compute_apps"])

    ids = split_ids(args.split)
    t0 = time.perf_counter()
    frames = [torch.from_numpy(load_frame(args.view, f"{SRC}/Vis/{i}.png", f"{SRC}/Ir/{i}.png")) for i in ids]
    if not cpu:
        frames = [f.pin_memory() for f in frames]
    decode_ms = (time.perf_counter() - t0) * 1e3 / len(ids)
    model = load_model(str(REPO / "runs" / args.run / "weights" / args.weights), dev)
    policy = make_policy(args.policy)
    analyzer = SceneAnalyzer(dev, getattr(policy, "features", None)) if policy.needs_features else None
    h, w = frames[0].shape[:2]
    nvml = None if cpu else NVML(args.gpu)
    sync = (lambda: None) if cpu else (lambda: torch.cuda.synchronize(dev))

    class CpuStamp:  # CUDA-event stand-in: on CPU, host wall clock is the device timeline
        def record(self):
            self.t = time.perf_counter()

        def elapsed_time(self, other):
            return (other.t - self.t) * 1e3

    def step(i: str, frame: torch.Tensor, ev):
        ev[0].record()
        g = frame.to(dev, non_blocking=True)
        ev[1].record()
        feats = analyzer(g) if analyzer is not None else None
        res = policy.choose(i, feats)
        ev[2].record()
        x, lb = to_input(g, res)
        ev[3].record()
        out = model(x)
        ev[4].record()
        dets = postprocess(out, lb, h, w, conf=args.conf).cpu()
        ev[5].record()
        return res, dets

    with torch.no_grad():
        ev = [CpuStamp() if cpu else torch.cuda.Event(enable_timing=True) for _ in range(6)]
        for k in range(args.warmup):
            step(ids[k % len(ids)], frames[k % len(ids)], ev)
        sync()

        idle_w = None
        if nvml:  # idle power (for dynamic-energy derivation)
            e0, t0 = nvml.energy_mj(), time.perf_counter()
            time.sleep(args.idle_seconds)
            idle_w = (nvml.energy_mj() - e0) / 1e3 / (time.perf_counter() - t0)
            torch.cuda.reset_peak_memory_stats(dev)

        e2e, stages, chosen, reps = [], {s: [] for s in STAGES}, {}, []
        for rep in range(args.reps):
            e_start, t_start = (nvml.energy_mj() if nvml else 0), time.perf_counter()
            for i, frame in zip(ids, frames):
                t = time.perf_counter()
                res, _ = step(i, frame, ev)
                sync()
                e2e.append((time.perf_counter() - t) * 1e3)
                for s, a, b in zip(STAGES, ev[:-1], ev[1:]):
                    stages[s].append(a.elapsed_time(b))
                chosen[i] = res
            secs = time.perf_counter() - t_start
            joules = (nvml.energy_mj() - e_start) / 1e3 if nvml else float("nan")
            reps.append({"seconds": secs, "joules": joules, "frames": len(ids),
                         "J_per_frame": joules / len(ids), "avg_power_W": joules / secs})

    lat = np.array(e2e)
    pct = lambda a, q: float(np.percentile(a, q))
    record = {
        "run": args.run, "view": args.view, "weights": args.weights, "policy": args.policy, "split": args.split,
        "tag": args.tag, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(), "code_hash": code_hash(), "device": "cpu" if cpu else "cuda", "gpu_index": args.gpu,
        "cpu_threads": args.cpu_threads,
        "gpu_state_before": state_before, "gpu_state_after": None if cpu else gpu_state(args.gpu),
        "torch": torch.__version__, "cuda": torch.version.cuda, "host": platform.node(),
        "batch_size": 1, "conf": args.conf, "warmup": args.warmup, "reps": args.reps, "precision": "fp32",
        "decode_ms_per_frame": decode_ms, "idle_power_W": idle_w,
        "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99), "mean": float(lat.mean()),
                       "iqr": [pct(lat, 25), pct(lat, 75)]},
        "stage_ms_mean": {s: float(np.mean(v)) for s, v in stages.items()},
        "stage_ms_p95": {s: pct(np.array(v), 95) for s, v in stages.items()},
        "stage_sum_over_e2e": float(sum(np.mean(v) for v in stages.values()) / lat.mean()),
        "throughput_fps": float(1e3 / lat.mean()),
        "peak_mem_MiB": None if cpu else torch.cuda.max_memory_allocated(dev) / 2**20,
        "energy": {"J_per_frame": float(np.mean([r["J_per_frame"] for r in reps])),
                   "J_per_frame_dynamic": (float(np.mean([(r["joules"] - idle_w * r["seconds"]) / r["frames"]
                                                          for r in reps])) if idle_w is not None else None),
                   "reps": reps},
        "resolution_mix": {str(r): int(sum(v == r for v in chosen.values())) for r in sorted(set(chosen.values()))},
        "chosen": chosen,
        "latency_ms_all": [round(v, 4) for v in e2e],
    }
    out = REPO / "results" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    where = f"_cpu{args.cpu_threads}" if cpu else ""
    name = f"bench_{args.run}_{args.policy.replace(':', '-').replace('/', '_')}_{args.split}{where}{args.tag}.json"
    (out / name).write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("policy", "latency_ms", "stage_ms_mean", "stage_sum_over_e2e",
                                             "peak_mem_MiB", "resolution_mix")}, indent=1))
    if idle_w is not None:
        print("energy J/frame", round(record["energy"]["J_per_frame"], 4), "idle W", round(idle_w, 1))


if __name__ == "__main__":
    main()
