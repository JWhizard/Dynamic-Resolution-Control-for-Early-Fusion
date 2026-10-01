"""Train one detector configuration (B0/B1/B2, resolution specialists, multi-scale).

    python train_early_fusion.py --config configs/b2_fusion_640.yaml --device 1
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # libtiff ExtraSamples warning on every 4-channel read

import yaml
from ultralytics import YOLO

from common.repro import set_global_seed
from fusion.trainer import make_trainer

REPO = Path(__file__).resolve().parent
LOG = REPO / "results" / "experiment_log.csv"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return "uncommitted"


def code_hash() -> str:
    """SHA-256 over the repo's Python sources and configs, so a result is traceable even before a commit."""
    h = hashlib.sha256()
    for f in sorted([*REPO.glob("*.py"), *REPO.glob("*/*.py"), *REPO.glob("configs/*.yaml")]):
        h.update(f.relative_to(REPO).as_posix().encode() + f.read_bytes())
    return h.hexdigest()[:12]


def log_run(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=None, help="overrides config seed")
    ap.add_argument("--epochs", type=int, default=None, help="overrides config (smoke tests)")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = args.seed if args.seed is not None else cfg["seed"]
    name = f"{cfg['name']}_seed{seed}{args.suffix}"
    set_global_seed(seed)

    train_args = dict(cfg["train"])
    if args.epochs is not None:
        train_args["epochs"] = args.epochs
    start = dt.datetime.now().isoformat(timespec="seconds")
    sha, chash = git_sha(), code_hash()  # at launch: code may be edited while training runs
    snap = REPO / "runs" / name / "code"
    shutil.rmtree(snap, ignore_errors=True)
    for d in ("common", "fusion", "configs"):
        shutil.copytree(REPO / d, snap / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in REPO.glob("*.py"):
        shutil.copy2(f, snap / f.name)
    model = YOLO(cfg["weights"])
    model.train(
        data=cfg["data"],
        trainer=make_trainer(cfg.get("ms_sizes")),
        project=str(REPO / "runs"),
        name=name,
        exist_ok=True,
        seed=seed,
        deterministic=True,
        device=args.device,
        **train_args,
    )
    log_run({
        "run_id": name, "git_sha": sha, "code_hash": chash, "config": args.config, "seed": seed,
        "device": f"cuda:{args.device}", "start": start,
        "end": dt.datetime.now().isoformat(timespec="seconds"), "outcome": "trained",
    })


if __name__ == "__main__":
    main()
