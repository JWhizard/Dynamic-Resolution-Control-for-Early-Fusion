"""pycocotools evaluation of any per-image resolution policy, from the prediction cache.

Conventions follow YOLO_Workspace's evaluate_official_coco.py: GT boxes in native
pixels of each frame (most are 1024x768, but 274 of 4200 are smaller, 400x280 to
880x520; sizes in splits/image_sizes.json), area = w*h (AP_small = area < 32^2 native),
category_id = class + 1, standard maxDets (1, 10, 100).

    python evaluate.py --runs b0_rgb_640_seed42 b2_fusion_640_seed42 ms_fusion_seed42 --split test
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from predict_cache import REPO, SRC, load_cache, split_ids
from fusion.preprocess import RESOLUTIONS

NAMES = ["Bus", "Car", "Lamp", "Motorcycle", "People", "Truck"]
PEOPLE = NAMES.index("People")
SIZES = REPO / "splits" / "image_sizes.json"
RECALL_CONF = 0.25
STAT_NAMES = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


@lru_cache(maxsize=None)
def ground_truth(split: str) -> tuple[COCO, tuple[str, ...]]:
    ids = tuple(split_ids(split))
    sizes = json.loads(SIZES.read_text())
    images, anns = [], []
    for k, i in enumerate(ids):
        W, H = sizes[i]
        images.append({"id": k, "file_name": f"{i}.png", "width": W, "height": H})
        for line in (SRC / "labels_yolo" / f"{i}.txt").read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, bw, bh = line.split()[:5]
            bw, bh = float(bw) * W, float(bh) * H
            anns.append({"id": len(anns) + 1, "image_id": k, "category_id": int(c) + 1, "iscrowd": 0,
                         "bbox": [float(cx) * W - bw / 2, float(cy) * H - bh / 2, bw, bh], "area": bw * bh})
    gt = COCO()
    gt.dataset = {"info": {}, "images": images, "annotations": anns,
                  "categories": [{"id": k + 1, "name": n} for k, n in enumerate(NAMES)]}
    _quiet(gt.createIndex)
    return gt, ids


def policy_detections(dets_by_res: dict[int, dict], policy: dict[str, int], ids, min_conf=0.0) -> list[dict]:
    out = []
    for k, i in enumerate(ids):
        d = dets_by_res[policy[i]][i]
        d = d[d[:, 4] >= min_conf]
        for x1, y1, x2, y2, s, c in d.tolist():
            out.append({"image_id": k, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    return out


def run_cocoeval(gt: COCO, dets: list[dict]) -> COCOeval:
    dt = _quiet(gt.loadRes, dets) if dets else COCO()
    ev = COCOeval(gt, dt, iouType="bbox")
    _quiet(ev.evaluate)
    _quiet(ev.accumulate)
    _quiet(ev.summarize)
    return ev


def evaluate_policy(dets_by_res: dict[int, dict], policy: dict[str, int], split: str) -> dict:
    """Scalar metrics for one policy. dets_by_res: {res: {img_id: (n,6) native dets}}."""
    gt, ids = ground_truth(split)
    ev = run_cocoeval(gt, policy_detections(dets_by_res, policy, ids))
    m = dict(zip(STAT_NAMES, map(float, ev.stats)))
    prec = ev.eval["precision"]  # [IoU, recall, class, area, maxDet]
    for c, n in enumerate(NAMES):
        v = prec[:, :, c, 0, -1]
        m[f"AP_{n}"] = float(v[v > -1].mean()) if (v > -1).any() else float("nan")
    # People recall at a deployment confidence threshold, IoU 0.5, all areas / small only
    ev25 = run_cocoeval(gt, policy_detections(dets_by_res, policy, ids, min_conf=RECALL_CONF))
    rec = ev25.eval["recall"]  # [IoU, class, area, maxDet]
    m["People_recall50"] = float(rec[0, PEOPLE, 0, -1])
    m["People_small_recall50"] = float(rec[0, PEOPLE, 1, -1])
    m["mean_res"] = float(np.mean([policy[i] for i in ids]))
    m["mean_pixels_rel640"] = float(np.mean([(policy[i] / 640) ** 2 for i in ids]))
    return m


def _ap_101(e: dict, rec_thrs: np.ndarray, max_det: int) -> float | None:
    """COCO 101-point AP (mean over IoU thresholds) for one evalImgs entry;
    mirrors COCOeval.accumulate. None when the entry has no non-ignored GT."""
    npig = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
    if npig == 0:
        return None
    scores = np.asarray(e["dtScores"][:max_det])
    order = np.argsort(-scores, kind="mergesort")
    dtm = np.asarray(e["dtMatches"])[:, :max_det][:, order]
    dtig = np.asarray(e["dtIgnore"])[:, :max_det][:, order].astype(bool)
    tps = np.cumsum(np.logical_and(dtm, ~dtig), axis=1, dtype=float)
    fps = np.cumsum(np.logical_and(dtm == 0, ~dtig), axis=1, dtype=float)
    aps = []
    for tp, fp in zip(tps, fps):
        rc = tp / npig
        pr = (tp / (fp + tp + np.spacing(1))).tolist()
        for j in range(len(pr) - 1, 0, -1):
            pr[j - 1] = max(pr[j - 1], pr[j])
        q = np.zeros(len(rec_thrs))
        inds = np.searchsorted(rc, rec_thrs, side="left")
        for ri, pi in enumerate(inds):
            if pi < len(pr):
                q[ri] = pr[pi]
        aps.append(q.mean())
    return float(np.mean(aps))


def per_image_ap(dets: dict, split: str) -> dict[str, float]:
    """Per-image mAP@50-95: mean over the classes present in that image's GT.

    Images with no GT get NaN. FPs on classes absent from an image's GT are not
    penalised (COCO AP is per class); a known limitation of per-image AP.
    """
    gt, ids = ground_truth(split)
    ev = COCOeval(gt, _quiet(gt.loadRes, policy_detections({0: dets}, dict.fromkeys(ids, 0), ids)), "bbox")
    _quiet(ev.evaluate)
    p = ev.params
    n_img, n_area, max_det = len(p.imgIds), len(p.areaRng), p.maxDets[-1]
    out = {}
    for k, i in enumerate(ids):
        aps = []
        for kc in range(len(p.catIds)):
            e = ev.evalImgs[kc * n_area * n_img + k]  # area index 0 = all
            ap = _ap_101(e, p.recThrs, max_det) if e is not None else None
            if ap is not None:
                aps.append(ap)
        out[i] = float(np.mean(aps)) if aps else float("nan")
    return out


def load_run(run: str, split: str, res=RESOLUTIONS) -> dict[int, dict]:
    return {r: load_cache(run, r, split) for r in res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        for r in RESOLUTIONS:
            try:
                dets = load_run(run, args.split, (r,))
            except FileNotFoundError:
                continue
            _, ids = ground_truth(args.split)
            rows.append({"run": run, "policy": f"fixed{r}", "split": args.split,
                         **evaluate_policy(dets, dict.fromkeys(ids, r), args.split)})
    df = pd.DataFrame(rows)
    out = Path(args.out or REPO / "results" / "processed" / f"e1_fixed_{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(df[["run", "policy", "AP", "AP50", "AP_small", "AP_People", "People_recall50"]].round(4))


if __name__ == "__main__":
    main()
