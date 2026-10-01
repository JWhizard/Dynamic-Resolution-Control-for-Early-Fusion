"""Controller v3: beat the best FIXED resolution, not just a random 320/480/640 mix.

Changes from v2 (see docs/controller_improvement_plan.md):
  * Actions {400, 480, 544, 576, 640}; the baseline is the best-fixed hull: every fixed size in
    {320, 400, 480, 544, 576, 640} and every random mix of two of them, at equal mean CPU cost.
  * Denoised targets on the fitting split (train):
      raw       per-image AP of seed 42
      seedavg   mean per-image AP over MS seeds 42/123/456
      crossfit  out-of-fold per-image AP from two half-train MS models (detector never saw the image)
    each optionally shrunk toward its sequence mean: q~ = a*q + (1-a)*mean_seq(q), a selected on val.
  * Selection on VAL by mean gain over each seed's val hull (3 seeds), test scored once on all 3 seeds,
    with a sequence-bootstrap CI of the paired difference vs a cost-matched mix of the hull endpoints.

    python controller_v3.py --target seedavg
    python controller_v3.py --target crossfit
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from controller_heuristic import Heuristic, fit_policy_tree, tree_rules
from evaluate import ground_truth, per_image_ap, policy_detections, run_cocoeval
from predict_cache import load_cache, split_ids

REPO = Path(__file__).resolve().parent
PROC = REPO / "results" / "processed"
QDIR = PROC / "q"
ACTIONS = [400, 480, 544, 576, 640]
FIXED = [320, 400, 480, 544, 576, 640]
SEEDS = (42, 123, 456)
MANIFEST = json.loads((REPO / "splits" / "m3fd_grouped_seed42.json").read_text())
GROUPS = MANIFEST["groups"]


def cost_model() -> dict[int, float]:
    cm = json.loads((PROC / "cpu_cost_model_v3.json").read_text())["costs"]
    return {int(k): v["mean_ms"] for k, v in cm.items()}


def q_table(run: str, split: str, res) -> pd.DataFrame:
    """Per-image AP (rows = image ids, columns = resolutions), cached per (run, split, res)."""
    QDIR.mkdir(parents=True, exist_ok=True)
    cols = {}
    for r in res:
        f = QDIR / f"{run}_{split}_{r}.csv"
        if not f.exists():
            pd.Series(per_image_ap(load_cache(run, r, split), split), name="q").rename_axis("id").to_csv(f)
        cols[r] = pd.read_csv(f, dtype={"id": str}).set_index("id")["q"]
    return pd.DataFrame(cols).fillna(0.0)


def crossfit_q(res) -> pd.DataFrame:
    halves = json.loads((REPO / "splits" / "crossfit_halves_seed42.json").read_text())
    qa = q_table("crossfit_ms_halfA_seed42", "train", res)  # model A scores half-B images
    qb = q_table("crossfit_ms_halfB_seed42", "train", res)
    return pd.concat([qa.loc[halves["B"]], qb.loc[halves["A"]]]).sort_index()


def shrink(q: pd.DataFrame, a: float) -> pd.DataFrame:
    g = q.groupby(lambda i: GROUPS[i]).transform("mean")
    return a * q + (1 - a) * g


def ap_only(dets_by_res, policy, split) -> float:
    gt, ids = ground_truth(split)
    return float(run_cocoeval(gt, policy_detections(dets_by_res, policy, ids)).stats[0])


class Hull:
    """Best-fixed hull for one seed and split: max over fixed sizes and two-size random mixes at a given cost."""

    def __init__(self, dets, split, C):
        ids = split_ids(split)
        self.C = C
        self.ap = {r: ap_only(dets, dict.fromkeys(ids, r), split) for r in FIXED}

    def at(self, c: float) -> tuple[float, tuple[int, int, float]]:
        # clamp: a policy's mean cost can exceed max(C) by float round-off (e.g. all-640 policies)
        c = min(max(c, min(self.C[r] for r in FIXED)), max(self.C[r] for r in FIXED))
        best = (-1.0, None)
        for r1 in FIXED:
            for r2 in FIXED:
                c1, c2 = self.C[r1], self.C[r2]
                if c1 - 1e-9 <= c <= c2 + 1e-9:
                    w = 0.0 if c2 == c1 else min(max((c - c1) / (c2 - c1), 0.0), 1.0)
                    a = self.ap[r1] + w * (self.ap[r2] - self.ap[r1])
                    if a > best[0]:
                        best = (a, (r1, r2, w))
        assert best[1] is not None, c
        return best


def apply(tree, X, names):
    h = Heuristic(tree)
    return [h.choose(None, dict(zip(names, row))) for row in X]


def hull_mix_policy(ids, mix, seed=0):
    r1, r2, w = mix
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(ids, int(round(w * len(ids))), replace=False).tolist())
    return {i: (r2 if i in pick else r1) for i in ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["raw", "seedavg", "crossfit"], default="seedavg")
    ap.add_argument("--features", nargs="+", default=["lum_mean_s", "lum_std_s"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    ap.add_argument("--boot", type=int, default=100)
    args = ap.parse_args()
    C = cost_model()
    cost = np.array([C[r] for r in ACTIONS])
    tag = f"{args.target}_{'-'.join(args.features)}"

    feats = pd.read_csv(PROC / "scene_features.csv", dtype={"id": str}).set_index("id")
    tr, va, te = (split_ids(s) for s in ("train", "val", "test"))

    if args.target == "raw":
        q_fit = q_table("ms_fusion_seed42", "train", ACTIONS)
    elif args.target == "seedavg":
        q_fit = sum(q_table(f"ms_fusion_seed{s}", "train", ACTIONS) for s in SEEDS) / len(SEEDS)
    else:
        q_fit = crossfit_q(ACTIONS)
    q_fit = q_fit.loc[tr]
    X_fit, X_val, X_test = (feats.loc[ids, args.features].to_numpy(float) for ids in (tr, va, te))

    dets = {sp: {s: {r: load_cache(f"ms_fusion_seed{s}", r, sp) for r in FIXED} for s in SEEDS} for sp in ("val", "test")}
    hulls = {sp: {s: Hull(dets[sp][s], sp, C) for s in SEEDS} for sp in ("val", "test")}

    # group-aware CV folds on the fitting split
    g = np.array([GROUPS[i] for i in tr])
    order = np.random.default_rng(42).permutation(np.unique(g))
    fold = np.array([{grp: k % 5 for k, grp in enumerate(order)}[x] for x in g])

    rows, store = [], {}
    lams = np.concatenate([[0.0], np.geomspace(2e-4, 2e-2, 14)])
    for a in args.alphas:
        Q = shrink(q_fit, a).to_numpy()
        for lam in lams:
            best = None
            for depth in (1, 2, 3):
                for ml in (40, 80, 160):
                    u = 0.0
                    for k in range(5):
                        t = fit_policy_tree(X_fit[fold != k], Q[fold != k], cost, args.features, ACTIONS, lam, depth, ml)
                        ch = np.array([ACTIONS.index(r) for r in apply(t, X_fit[fold == k], args.features)])
                        u += (Q[fold == k, ch] - lam * cost[ch]).sum()
                    if best is None or u > best[0]:
                        best = (u, depth, ml)
            tree = fit_policy_tree(X_fit, Q, cost, args.features, ACTIONS, lam, best[1], best[2])
            pv = dict(zip(va, apply(tree, X_val, args.features)))
            cv_ = float(np.mean([C[r] for r in pv.values()]))
            gains = [ap_only(dets["val"][s], pv, "val") - hulls["val"][s].at(cv_)[0] for s in SEEDS]
            rows.append({"alpha": a, "lam": lam, "depth": best[1], "min_leaf": best[2], "val_cost_ms": cv_,
                         "val_gain_vs_hull": float(np.mean(gains)),
                         **{f"val_n{r}": sum(v == r for v in pv.values()) for r in ACTIONS}})
            store[(a, float(lam))] = tree
    sweep = pd.DataFrame(rows)
    sel = sweep.loc[sweep.val_gain_vs_hull.idxmax()]
    tree = store[(sel.alpha, float(sel.lam))]

    # ---- test, scored once on all three seeds
    pt = dict(zip(te, apply(tree, X_test, args.features)))
    ct = float(np.mean([C[r] for r in pt.values()]))
    res = {}
    for s in SEEDS:
        hap, mix = hulls["test"][s].at(ct)
        res[s] = {"AP": ap_only(dets["test"][s], pt, "test"), "hull_AP": hap, "hull_mix": mix}
    gain = float(np.mean([v["AP"] - v["hull_AP"] for v in res.values()]))

    # paired sequence bootstrap: policy vs cost-matched mix of the hull endpoints (same frames, same seed)
    test_groups = sorted({GROUPS[i] for i in te})
    by_group = {grp: [i for i in te if GROUPS[i] == grp] for grp in test_groups}
    rng = np.random.default_rng(0)
    diffs = []
    gt, _ = ground_truth("test")
    for b in range(args.boot):
        draw = rng.choice(test_groups, len(test_groups), replace=True)
        ids_b = [i for grp in draw for i in by_group[grp]]
        d = []
        for s in SEEDS:
            hp = hull_mix_policy(te, res[s]["hull_mix"])
            d.append(_ap_on(gt, dets["test"][s], pt, ids_b) - _ap_on(gt, dets["test"][s], hp, ids_b))
        diffs.append(np.mean(d))
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    out = {"target": args.target, "features": args.features, "actions": ACTIONS, "fixed": FIXED, "cost_ms": C,
           "selected": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                        for k, v in sel.to_dict().items()},
           "tree": tree, "rules": tree_rules(tree), "test_cost_ms": ct,
           "test_mix": {str(r): sum(v == r for v in pt.values()) for r in ACTIONS},
           "test_by_seed": {str(s): {"AP": v["AP"], "hull_AP": v["hull_AP"], "hull_mix": list(v["hull_mix"])}
                            for s, v in res.items()},
           "test_gain_vs_hull_mean_over_seeds": gain,
           "bootstrap_gain_vs_hull_mix_95ci": [float(lo), float(hi)], "bootstrap_reps": args.boot,
           "fixed_test_AP_seed42": hulls["test"][42].ap}
    (PROC / "v3").mkdir(exist_ok=True)
    sweep.to_csv(PROC / "v3" / f"sweep_{tag}.csv", index=False)
    (PROC / "v3" / f"controller_{tag}.json").write_text(json.dumps(out, indent=1))
    (PROC / "v3" / f"policy_{tag}_test.json").write_text(json.dumps(pt))
    print(sweep.sort_values("val_gain_vs_hull", ascending=False).head(8).round(4).to_string(index=False))
    print(out["rules"])
    print(f"TEST: cost {ct:.2f} ms, mix {out['test_mix']}, gain vs best-fixed hull (mean of 3 seeds) {gain:+.4f}, "
          f"bootstrap 95% CI vs hull mix [{lo:+.4f}, {hi:+.4f}]")
    for s, v in res.items():
        print(f"  seed {s}: AP {v['AP']:.4f}  hull {v['hull_AP']:.4f}  ({v['hull_mix']})")


def _ap_on(gt, dets_by_res, policy, ids_b):
    """AP on a bootstrap resample (duplicated frames get fresh image ids)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    _, ids = ground_truth("test")
    k_of = {i: k for k, i in enumerate(ids)}
    images, anns, dts = [], [], []
    for n, i in enumerate(ids_b):
        img = dict(gt.imgs[k_of[i]])
        img["id"] = n
        images.append(img)
        for a in gt.imgToAnns[k_of[i]]:
            a = dict(a)
            a["id"], a["image_id"] = len(anns) + 1, n
            anns.append(a)
        d = dets_by_res[policy[i]][i]
        for x1, y1, x2, y2, sc, c in d.tolist():
            dts.append({"image_id": n, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": sc})
    g = COCO()
    g.dataset = {"info": {}, "images": images, "annotations": anns, "categories": gt.dataset["categories"]}
    with contextlib.redirect_stdout(io.StringIO()):
        g.createIndex()
        ev = COCOeval(g, g.loadRes(dts), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])


if __name__ == "__main__":
    main()
