"""per_image_ap must equal a standalone COCOeval restricted to that image."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as E


def test_per_image_ap_matches_cocoeval(split="val", n=25, seed=0):
    gt, ids = E.ground_truth(split)
    rng = np.random.default_rng(seed)
    # synthetic detections: jittered GT boxes with random scores plus random false positives
    dets = {}
    for k, i in enumerate(ids):
        rows = []
        for a in gt.loadAnns(gt.getAnnIds(imgIds=[k])):
            x, y, w, h = a["bbox"]
            j = rng.normal(0, 0.1, 4) * [w, h, w, h]
            rows.append([x + j[0], y + j[1], x + w + j[2], y + h + j[3], rng.random(), a["category_id"] - 1])
        for _ in range(rng.integers(0, 5)):
            x, y = rng.random() * 900, rng.random() * 650
            rows.append([x, y, x + 60, y + 80, rng.random(), rng.integers(0, 6)])
        dets[i] = np.array(rows, dtype=np.float32).reshape(-1, 6)
    fast = E.per_image_ap(dets, split)
    dt = E._quiet(gt.loadRes, E.policy_detections({0: dets}, dict.fromkeys(ids, 0), ids))
    for k in rng.choice(len(ids), n, replace=False):
        ev = E.COCOeval(gt, dt, "bbox")
        ev.params.imgIds = [int(k)]
        E._quiet(ev.evaluate)
        E._quiet(ev.accumulate)
        v = ev.eval["precision"][:, :, :, 0, -1]
        ref = float(v[v > -1].mean()) if (v > -1).any() else float("nan")
        got = fast[ids[k]]
        assert (np.isnan(ref) and np.isnan(got)) or abs(ref - got) < 1e-9, (ids[k], ref, got)
    print("per-image AP matches COCOeval on", n, "images")


if __name__ == "__main__":
    test_per_image_ap_matches_cocoeval()
