"""Paired sequence-bootstrap of a saved test policy against the best-fixed hull.

On each resample of test sequences (with replacement), the hull is evaluated in expectation:
(1-w)*AP(r1) + w*AP(r2) for the hull's two fixed endpoints at the policy's mean cost. This avoids
the extra noise of comparing against one random realisation of the mix. Run per seed and averaged.

Fast path: pycocotools matching (COCOeval.evaluate) runs once per detection set; each resample's
AP is then accumulated from the stored per-image match records with duplicates allowed, using the
same 101-point interpolation as COCOeval.accumulate (validated against full re-evaluation in
tests/test_bootstrap_ap.py).

    python bootstrap_vs_hull.py results/processed/v3/policy_*.json [--reps 1000]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.cocoeval import COCOeval

import controller_v3 as V
from predict_cache import split_ids


class MatchStore:
    """Per-image, per-category COCO match records for one detection set (area = all, maxDet = 100)."""

    def __init__(self, gt, dets_by_res, policy, ids):
        dt = gt.loadRes(V.policy_detections(dets_by_res, policy, ids)) if policy else None
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(gt, dt, "bbox")
            ev.evaluate()
        p = ev.params
        self.rec_thrs, self.max_det = p.recThrs, p.maxDets[-1]
        n_img, n_area = len(p.imgIds), len(p.areaRng)
        self.cats = range(len(p.catIds))
        self.e = {(c, k): ev.evalImgs[c * n_area * n_img + k] for c in self.cats for k in range(n_img)}

    def ap(self, img_idx: np.ndarray) -> float:
        """COCO AP (IoU .50:.95) over a multiset of image indices."""
        aps = []
        for c in self.cats:
            es = [self.e[(c, k)] for k in img_idx if self.e[(c, k)] is not None]
            if not es:
                continue
            gt_ig = np.concatenate([np.asarray(e["gtIgnore"], bool) for e in es])
            npig = int((~gt_ig).sum())
            if npig == 0:
                continue
            scores = np.concatenate([np.asarray(e["dtScores"][: self.max_det]) for e in es])
            dtm = np.concatenate([np.asarray(e["dtMatches"])[:, : self.max_det] for e in es], axis=1)
            dtig = np.concatenate([np.asarray(e["dtIgnore"])[:, : self.max_det] for e in es], axis=1).astype(bool)
            order = np.argsort(-scores, kind="mergesort")
            dtm, dtig = dtm[:, order], dtig[:, order]
            tps = np.cumsum(np.logical_and(dtm, ~dtig), axis=1, dtype=float)
            fps = np.cumsum(np.logical_and(dtm == 0, ~dtig), axis=1, dtype=float)
            per_t = []
            for tp, fp in zip(tps, fps):
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
                q = np.zeros(len(self.rec_thrs))
                inds = np.searchsorted(rc, self.rec_thrs, side="left")
                ok = inds < len(pr)
                q[ok] = pr[inds[ok]]
                per_t.append(q.mean())
            aps.append(np.mean(per_t))
        return float(np.mean(aps)) if aps else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policies", nargs="+")
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    C = V.cost_model()
    te = split_ids("test")
    gt, ids = V.ground_truth("test")
    k_of = {i: k for k, i in enumerate(ids)}
    dets = {s: {r: V.load_cache(f"ms_fusion_seed{s}", r, "test") for r in V.FIXED} for s in V.SEEDS}
    hulls = {s: V.Hull(dets[s], "test", C) for s in V.SEEDS}
    fixed_store = {(s, r): MatchStore(gt, dets[s], dict.fromkeys(ids, r), ids) for s in V.SEEDS for r in V.FIXED}
    groups = sorted({V.GROUPS[i] for i in te})
    by_group = {g: np.array([k_of[i] for i in te if V.GROUPS[i] == g]) for g in groups}
    dst = V.PROC / "v3" / "bootstrap_vs_hull.json"
    out = json.loads(dst.read_text()) if dst.exists() else {}
    for path in args.policies:
        pol = {k: int(v) for k, v in json.loads(Path(path).read_text()).items()}
        cost = float(np.mean([C[r] for r in pol.values()]))
        stores = {s: MatchStore(gt, dets[s], pol, ids) for s in V.SEEDS}
        full = np.arange(len(ids))
        point = {s: stores[s].ap(full) - hulls[s].at(cost)[0] for s in V.SEEDS}
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(args.reps):
            idx = np.concatenate([by_group[g] for g in rng.choice(groups, len(groups), replace=True)])
            d = []
            for s in V.SEEDS:
                r1, r2, w = hulls[s].at(cost)[1]
                h = (1 - w) * fixed_store[(s, r1)].ap(idx) + (w * fixed_store[(s, r2)].ap(idx) if w > 0 else 0.0)
                d.append(stores[s].ap(idx) - h)
            diffs.append(float(np.mean(d)))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        name = Path(path).stem
        out[name] = {"cost_ms": cost, "gain_by_seed": {str(k): v for k, v in point.items()},
                     "gain_mean": float(np.mean(list(point.values()))), "ci95": [float(lo), float(hi)],
                     "p_gain_le_0": float(np.mean(np.array(diffs) <= 0)), "reps": args.reps}
        dst.write_text(json.dumps(out, indent=1))
        print(f"{name}: cost {cost:.2f} ms  gain {out[name]['gain_mean']:+.4f} "
              f"(seeds {', '.join(f'{v:+.4f}' for v in point.values())})  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(gain<=0) {out[name]['p_gain_le_0']:.3f}", flush=True)


if __name__ == "__main__":
    main()
