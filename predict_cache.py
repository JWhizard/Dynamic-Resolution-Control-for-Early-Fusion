"""Cache per-image detections for every (model, resolution, split).

Any resolution policy (fixed, oracle, heuristic, random) is then scored offline
by picking, per image, the cached detections at the resolution the policy chose.
This is exact: at batch size 1 an image's detections depend only on its own
input resolution.

    python predict_cache.py --run ms_fusion_seed42 --view fusion --device 1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import torch

from fusion.preprocess import RESOLUTIONS, detect, load_frame, load_model

REPO = Path(__file__).resolve().parent
SRC = Path("/mnt/12TB_Drive/whisenaj/Datasets/M3FD_Detection")
MANIFEST = REPO / "splits" / "m3fd_grouped_seed42.json"
CACHE = REPO / "results" / "cache"


def split_ids(split: str) -> list[str]:
    return json.loads(MANIFEST.read_text())["splits"][split]


def cache_path(run: str, res: int, split: str) -> Path:
    return CACHE / f"{run}_{res}_{split}.pkl"


def load_cache(run: str, res: int, split: str) -> dict[str, "np.ndarray"]:
    with cache_path(run, res, split).open("rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory name under runs/")
    ap.add_argument("--view", required=True, choices=["fusion", "rgb", "ir"])
    ap.add_argument("--res", type=int, nargs="+", default=list(RESOLUTIONS))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--weights", default="last.pt")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.device}")
    model = load_model(str(REPO / "runs" / args.run / "weights" / args.weights), dev)
    CACHE.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        ids = split_ids(split)
        frames = {i: torch.from_numpy(load_frame(args.view, f"{SRC}/Vis/{i}.png", f"{SRC}/Ir/{i}.png")) for i in ids}
        for res in args.res:
            out = {i: detect(model, f.to(dev), res).cpu().numpy() for i, f in frames.items()}
            with cache_path(args.run, res, split).open("wb") as fh:
                pickle.dump(out, fh)
            print(f"{args.run} {split} @{res}: {len(out)} images, {sum(len(v) for v in out.values())} dets")


if __name__ == "__main__":
    main()
