"""Fit cost-sensitive resolution policies on VAL and score them on TEST.

    python controller_fit.py --kind probe --probe-res 192   # cascade: probe detections -> keep / re-run at 320/480/640
    python controller_fit.py --kind scene   # SceneAnalyzer features -> 320 / 480 / 640 (no probe)

Costs are measured per-frame latencies from bench_latency_energy.py on the chosen
platform (default: 4 CPU threads, the compute-bound edge proxy; on the GPU the
latency is flat in resolution, so no policy can save time there). For the
cascade, choosing R > 320 costs the probe plus the R pass.

For each lambda (AP-per-ms trade-off) the tree depth / min-leaf are chosen by
group-aware 5-fold CV on the fitting split's utility (val by default; --fit-split
train uses the 2940 train images), then refit on all of it. The operating
point reported as "selected" is chosen on val only: the cheapest lambda whose
val AP is within 2 points of fixed-640 (the proposal's success criterion).
A cost-matched random control (random mix of fixed resolutions with the same
expected cost, 10 draws) is scored alongside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from controller_heuristic import Heuristic, fit_policy_tree, tree_rules
from evaluate import evaluate_policy, load_run
from fusion.preprocess import RESOLUTIONS
from predict_cache import load_cache

REPO = Path(__file__).resolve().parent
PROC, RAW = REPO / "results" / "processed", REPO / "results" / "raw"
CLASSES = list(RESOLUTIONS)


def measured_cost(run: str, platform: str, res=RESOLUTIONS) -> np.ndarray:
    frozen = PROC / "cpu_cost_model.json"
    if platform == "_cpu4" and frozen.exists():  # fit against the frozen cost model, not the latest benchmark
        cm = json.loads(frozen.read_text())
        assert cm["run"] == run
        return np.array([cm["costs"][str(r)]["mean_ms"] for r in res])
    out = []
    for r in res:
        b = json.loads((RAW / f"bench_{run}_fixed-{r}_test{platform}.json").read_text())
        out.append(b["latency_ms"]["mean"])
    return np.array(out)


def features(kind: str, run: str, split: str, ids, probe_res: int = 320) -> tuple[np.ndarray, list[str]]:
    if kind == "probe":
        from probe_features import PROBE_FEATURES, probe_features

        cp = load_cache(run, probe_res, split)
        f = pd.DataFrame({i: probe_features(cp[i]) for i in ids}).T
        return f[PROBE_FEATURES].to_numpy(float), PROBE_FEATURES
    from scene_analyzer import FEATURES

    f = pd.read_csv(PROC / "scene_features.csv", dtype={"id": str}).set_index("id").loc[list(ids)]
    return f[FEATURES].to_numpy(float), FEATURES


def per_image_q(run: str, split: str, res) -> pd.DataFrame:
    from evaluate import per_image_ap

    base = PROC / f"per_image_ap_{run}_{split}.csv"
    if not base.exists():  # val/test come from generate_oracle_labels.py; train is built here on demand
        q = pd.DataFrame({r: per_image_ap(load_cache(run, r, split), split) for r in RESOLUTIONS})
        q.index.name = "id"
        q.to_csv(base)
    q = pd.read_csv(base, dtype={"id": str}).set_index("id")
    q.columns = [int(c) for c in q.columns]
    for r in res:
        if r not in q:
            extra = PROC / f"per_image_ap_{run}_{split}_{r}.csv"
            if not extra.exists():
                pd.Series(per_image_ap(load_cache(run, r, split), split), name=r).rename_axis("id").to_csv(extra)
            q[r] = pd.read_csv(extra, dtype={"id": str}).set_index("id").iloc[:, 0]
    return q[list(res)].fillna(0.0)


def cv_folds(ids, k=5, seed=42):
    groups = json.loads((REPO / "splits" / "m3fd_grouped_seed42.json").read_text())["groups"]
    g = np.array([groups[i] for i in ids])
    ug = np.random.default_rng(seed).permutation(np.unique(g))
    fold_of = {grp: n % k for n, grp in enumerate(ug)}
    return np.array([fold_of[x] for x in g])


def apply(tree, X, names):
    h = Heuristic(tree)
    return [h.choose(None, dict(zip(names, row))) for row in X]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["probe", "scene"], default="probe")
    ap.add_argument("--run", default="ms_fusion_seed42")
    ap.add_argument("--platform", default="_cpu4", help="bench filename suffix ('' = GPU)")
    ap.add_argument("--probe-res", type=int, default=320)
    ap.add_argument("--fit-split", choices=["val", "train"], default="val")
    ap.add_argument("--features", nargs="*", default=None,
                    help="restrict to these features (scene: e.g. lum_mean lum_std, the analyzer's light path, "
                         "0.11 ms GPU / 1.6 ms CPU vs 0.83 / 7.6 ms for the full feature set)")
    ap.add_argument("--max-loss", type=float, default=0.02, help="val AP loss allowed vs fixed-640 (selection)")
    args = ap.parse_args()

    global CLASSES
    base = measured_cost(args.run, args.platform)
    if args.kind == "probe":  # keep the probe's detections, or pay probe + a full pass at a larger resolution
        CLASSES = [args.probe_res] + [r for r in RESOLUTIONS if r > args.probe_res]
        cp = measured_cost(args.run, args.platform, (args.probe_res,))[0]
        cost = np.array([cp] + [cp + base[RESOLUTIONS.index(r)] for r in CLASSES[1:]])
    else:
        cost = base
    qv, qt = per_image_q(args.run, "val", CLASSES), per_image_q(args.run, "test", CLASSES)
    Xv, names = features(args.kind, args.run, "val", qv.index, args.probe_res)
    Xt, _ = features(args.kind, args.run, "test", qt.index, args.probe_res)
    if args.features:  # restrict columns (feature cost is part of the design, not an afterthought)
        keep = [names.index(f) for f in args.features]
        Xv, Xt, names = Xv[:, keep], Xt[:, keep], list(args.features)
    Qv = qv.to_numpy()
    # fitting data: val (default) or train (4x more images; the detector saw them, so absolute AP is
    # optimistic, but lambda is still selected on val and the result is scored once on test)
    if args.fit_split == "val":
        qf, Xf, Qf = qv, Xv, Qv
    else:
        qf = per_image_q(args.run, args.fit_split, CLASSES)
        Xf, all_names = features(args.kind, args.run, args.fit_split, qf.index, args.probe_res)
        if args.features:
            Xf = Xf[:, [all_names.index(f) for f in args.features]]
        Qf = qf.to_numpy()
    fold = cv_folds(list(qf.index))
    allres = sorted(set(CLASSES) | set(RESOLUTIONS))
    dets_v, dets_t = load_run(args.run, "val", allres), load_run(args.run, "test", allres)
    ap640_v = evaluate_policy(dets_v, dict.fromkeys(qv.index, 640), "val")["AP"]

    rows, trees = [], {}
    lams = np.concatenate([[0.0], np.geomspace(1e-4, 5e-2, 24)])
    for lam in lams:
        best = None
        for depth in (1, 2, 3):
            for ml in (20, 40, 80):
                u = 0.0
                for k in range(5):
                    tr, te = fold != k, fold == k
                    t = fit_policy_tree(Xf[tr], Qf[tr], cost, names, CLASSES, lam, depth, ml)
                    ch = np.array([CLASSES.index(r) for r in apply(t, Xf[te], names)])
                    u += (Qf[te, ch] - lam * cost[ch]).sum()
                if best is None or u > best[0]:
                    best = (u, depth, ml)
        _, depth, ml = best
        tree = fit_policy_tree(Xf, Qf, cost, names, CLASSES, lam, depth, ml)
        pol_v = dict(zip(qv.index, apply(tree, Xv, names)))
        pol_t = dict(zip(qt.index, apply(tree, Xt, names)))
        mv, mt = evaluate_policy(dets_v, pol_v, "val"), evaluate_policy(dets_t, pol_t, "test")
        cv_ = np.mean([cost[CLASSES.index(r)] for r in pol_v.values()])
        ct_ = np.mean([cost[CLASSES.index(r)] for r in pol_t.values()])
        mix = {r: sum(v == r for v in pol_t.values()) for r in CLASSES}  # keys: probe res + escalations
        rows.append({"kind": args.kind, "lam": lam, "depth": depth, "min_leaf": ml, "val_AP": mv["AP"],
                     "val_cost_ms": cv_, "test_AP": mt["AP"], "test_AP_small": mt["AP_small"],
                     "test_People_recall50": mt["People_recall50"], "test_cost_ms": ct_,
                     **{f"test_n{r}": n for r, n in mix.items()}})
        trees[float(lam)] = (tree, pol_t)
    df = pd.DataFrame(rows)

    ok = df[df.val_AP >= ap640_v - args.max_loss]
    sel = ok.loc[ok.val_cost_ms.idxmin()] if len(ok) else df.loc[df.val_AP.idxmax()]
    tree, pol_t = trees[float(sel.lam)]
    df["selected"] = df.lam == sel.lam
    tag = (f"{args.kind}{args.probe_res if args.kind == 'probe' else ''}{'_fittrain' if args.fit_split == 'train' else ''}"
           f"{'_' + '-'.join(args.features) if args.features else ''}")
    df.to_csv(PROC / f"policy_sweep_{tag}{args.platform}.csv", index=False)

    # cost-matched random control: random mix of fixed resolutions with the selected policy's expected cost
    fixed_t = {r: evaluate_policy(dets_t, dict.fromkeys(qt.index, r), "test")["AP"] for r in RESOLUTIONS}
    target = sel.test_cost_ms
    R = list(RESOLUTIONS)
    lo = max([r for r in R if base[R.index(r)] <= target], default=320)
    hi = min([r for r in R if base[R.index(r)] >= target], default=640)
    p_hi = 0.0 if hi == lo else (target - base[R.index(lo)]) / (base[R.index(hi)] - base[R.index(lo)])
    ids_t = list(qt.index)
    rnd = []
    for s in range(10):
        rng = np.random.default_rng(s)
        n_hi = int(round(p_hi * len(ids_t)))
        pick = set(rng.choice(ids_t, n_hi, replace=False).tolist())
        pol = {i: (hi if i in pick else lo) for i in ids_t}
        (PROC / f"policy_random_costmatched_{tag}_s{s}_test.json").write_text(json.dumps(pol))
        rnd.append(evaluate_policy(dets_t, pol, "test")["AP"])

    out = {"kind": args.kind, "fit_split": args.fit_split, "features": names, "probe_res": args.probe_res if args.kind == "probe" else None, "choices": CLASSES,
           "run": args.run, "platform": args.platform or "gpu", "cost_ms_by_choice": cost.tolist(),
           "fixed_cost_ms": base.tolist(), "selection": f"cheapest val policy with val AP >= fixed640 - {args.max_loss}",
           "val_AP_fixed640": ap640_v, "lam": float(sel.lam), "depth": int(sel.depth), "min_leaf": int(sel.min_leaf),
           "tree": tree, "rules": tree_rules(tree),
           "test": {k: float(sel[k]) for k in ("test_AP", "test_AP_small", "test_People_recall50", "test_cost_ms")},
           "test_fixed_AP": fixed_t,
           "random_costmatched": {"mix": f"{lo}/{hi} with p({hi})={p_hi:.3f}", "AP_mean": float(np.mean(rnd)),
                                  "AP_std": float(np.std(rnd, ddof=1))}}
    (PROC / f"controller_{tag}.json").write_text(json.dumps(out, indent=1))
    (PROC / f"policy_{tag}_selected_test.json").write_text(json.dumps(pol_t))
    with pd.option_context("display.width", 200):
        print(df[["lam", "depth", "min_leaf", "val_AP", "val_cost_ms", "test_AP", "test_cost_ms",
                  *[f"test_n{r}" for r in CLASSES], "selected"]].round(4).to_string(index=False))
    print(out["rules"])
    print("fixed test AP", {k: round(v, 4) for k, v in fixed_t.items()}, "fixed cost ms", base.round(2).tolist())
    print("selected:", out["test"], "| cost-matched random AP %.4f +- %.4f (%s)" % (
        out["random_costmatched"]["AP_mean"], out["random_costmatched"]["AP_std"], out["random_costmatched"]["mix"]))


if __name__ == "__main__":
    main()
