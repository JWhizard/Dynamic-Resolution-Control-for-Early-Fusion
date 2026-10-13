"""Fit and evaluate an amortised controller over preceding-frame detections.

Training uses out-of-fold detector outputs and targets.  At runtime there is no probe pass:
features come from detections already produced for the preceding frame, a decision is made once
per window, and that resolution is held for the next ``interval`` frames.

    python fit_sequence_controller.py --interval 16

The selected artifact is written to ``results/processed/sequence/controller.json`` and can be
loaded by ``bench_latency_energy.py --policy sequence:<path>``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import controller_v3 as V
from controller_heuristic import fit_policy_tree, tree_rules
from controller_sequence import SequenceController, feature_vector
from predict_cache import load_cache, split_ids

REPO = Path(__file__).resolve().parent
OUT = V.PROC / "sequence"
ACTIONS = [480, 544, 576, 640]
FEATURES = [
    "n_conf25", "n_conf10", "n_small_conf10", "n_people_conf10",
    "n_small_people_conf10", "frac_small_conf10", "median_area_conf25",
    "min_area_conf25", "mean_conf_top10", "n_uncertain", "max_conf",
]


def ordered_groups(ids: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        out[V.GROUPS[i]].append(i)
    return {g: sorted(v, key=int) for g, v in out.items()}


def crossfit_detections(resolutions) -> dict[int, dict[str, np.ndarray]]:
    halves = json.loads((REPO / "splits" / "crossfit_halves_seed42.json").read_text())
    half_a = set(halves["A"])
    out = {}
    for r in resolutions:
        # Model A scores half B and model B scores half A: neither saw its scored frame.
        a = load_cache("crossfit_ms_halfA_seed42", r, "train")
        b = load_cache("crossfit_ms_halfB_seed42", r, "train")
        out[r] = {i: (b[i] if i in half_a else a[i]) for i in half_a | set(halves["B"])}
    return out


def training_windows(interval: int) -> tuple[np.ndarray, np.ndarray]:
    """Previous-frame detection features -> mean quality over the following window.

    Each observation is repeated for every possible preceding resolution so the tree is robust
    to its own earlier choices rather than being trained only on fixed-576 feedback.
    """
    ids = split_ids("train")
    q = V.crossfit_q(ACTIONS).loc[ids]
    dets = crossfit_detections(ACTIONS)
    xs, ys = [], []
    for seq in ordered_groups(ids).values():
        for boundary in range(interval - 1, len(seq) - 1, interval):
            future = seq[boundary + 1:boundary + 1 + interval]
            target = q.loc[future].mean(0).to_numpy(float)
            for source_res in ACTIONS:
                xs.append(feature_vector(dets[source_res][seq[boundary]]))
                ys.append(target)
    if not xs:
        raise RuntimeError(f"no training sequence is longer than interval={interval}")
    return np.stack(xs), np.stack(ys)


def simulate(tree: dict, split: str, run: str, interval: int, default_res: int,
             safety_small_people: bool, feedback_dets=None) -> dict[str, int]:
    ids = split_ids(split)
    dets = feedback_dets or {r: load_cache(run, r, split) for r in ACTIONS}
    policy = SequenceController(tree, interval, default_res, ACTIONS, V.GROUPS, safety_small_people)
    chosen = {}
    # Preserve the manifest's streaming order exactly. A group can recur after another group;
    # runtime resets in that case, so regrouping frames offline would produce a different trace.
    for i in ids:
        r = policy.choose(i, None)
        chosen[i] = r
        policy.update(i, dets[r][i])
    return chosen


def score(tree: dict, split: str, interval: int, default_res: int, safety: bool,
          dets_by_seed, hulls, costs, score_cache=None) -> tuple[float, list[dict]]:
    score_cache = {} if score_cache is None else score_cache
    ids = split_ids(split)
    rows = []
    for seed in V.SEEDS:
        p = simulate(tree, split, f"ms_fusion_seed{seed}", interval, default_res, safety,
                     dets_by_seed[seed])
        signature = tuple(p[i] for i in ids)
        key = (split, seed, signature)
        if key in score_cache:
            ap, mean_cost, hull_ap, mix = score_cache[key]
        else:
            mean_cost = float(np.mean([costs[r] for r in p.values()]))
            ap = V.ap_only(dets_by_seed[seed], p, split)
            hull_ap, mix = hulls[seed].at(mean_cost)
            score_cache[key] = (ap, mean_cost, hull_ap, mix)
        rows.append({"seed": seed, "AP": ap, "cost_ms": mean_cost, "hull_AP": hull_ap,
                     "gain_vs_hull": ap - hull_ap, "hull_mix": list(mix),
                     "mix": {str(r): sum(v == r for v in p.values()) for r in ACTIONS}, "policy": p})
    return float(np.mean([x["gain_vs_hull"] for x in rows])), rows


def linear_hull_at(c: float, quality: dict[int, float], costs: dict[int, float]) -> float:
    """Best interpolation of fixed operating points for a cheap validation surrogate."""
    best = -np.inf
    for r1 in ACTIONS:
        for r2 in ACTIONS:
            c1, c2 = costs[r1], costs[r2]
            if c1 - 1e-9 <= c <= c2 + 1e-9:
                w = 0.0 if c1 == c2 else (c - c1) / (c2 - c1)
                best = max(best, quality[r1] + w * (quality[r2] - quality[r1]))
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=16)
    ap.add_argument("--default-res", type=int, default=576, choices=ACTIONS)
    ap.add_argument("--no-safety-small-people", action="store_true")
    args = ap.parse_args()
    safety = not args.no_safety_small_people
    costs = V.cost_model()
    cost = np.asarray([costs[r] for r in ACTIONS])
    X, Q = training_windows(args.interval)

    dets = {
        split: {s: {r: load_cache(f"ms_fusion_seed{s}", r, split) for r in V.FIXED} for s in V.SEEDS}
        for split in ("val", "test")
    }
    hulls = {
        split: {s: V.Hull(dets[split][s], split, costs) for s in V.SEEDS}
        for split in ("val", "test")
    }

    q_val = sum(V.q_table(f"ms_fusion_seed{s}", "val", ACTIONS) for s in V.SEEDS) / len(V.SEEDS)
    fixed_q = {r: float(q_val[r].mean()) for r in ACTIONS}
    val_ids = split_ids("val")
    candidates, trees, seen_policies = [], {}, set()
    candidate_id = 0
    for lam in np.concatenate([[0.0], np.geomspace(2e-4, 2e-2, 14)]):
        for depth in (1, 2, 3):
            for min_leaf in (4, 8, 16, 32):
                if 2 * min_leaf > len(X):
                    continue
                tree = fit_policy_tree(X, Q, cost, FEATURES, ACTIONS, float(lam), depth, min_leaf)
                policies = [simulate(tree, "val", f"ms_fusion_seed{s}", args.interval,
                                     args.default_res, safety, dets["val"][s]) for s in V.SEEDS]
                signature = tuple(tuple(p[i] for i in val_ids) for p in policies)
                if signature in seen_policies:
                    continue
                seen_policies.add(signature)
                surrogate_gains, policy_costs = [], []
                for p in policies:
                    c = float(np.mean([costs[p[i]] for i in val_ids]))
                    q = float(np.mean([q_val.loc[i, p[i]] for i in val_ids]))
                    surrogate_gains.append(q - linear_hull_at(c, fixed_q, costs))
                    policy_costs.append(c)
                candidates.append({"lam": lam, "depth": depth, "min_leaf": min_leaf,
                                   "surrogate_gain_vs_hull": float(np.mean(surrogate_gains)),
                                   "val_cost_ms": float(np.mean(policy_costs)),
                                   "candidate_id": candidate_id})
                trees[candidate_id] = tree
                candidate_id += 1

    # Exact COCO AP is expensive. Rank all distinct policies with per-image AP, then apply the
    # actual validation criterion to a compact shortlist. Test remains untouched until selection.
    shortlist = sorted(candidates, key=lambda x: x["surrogate_gain_vs_hull"], reverse=True)[:8]
    val_score_cache = {}
    exact = {}
    for row in shortlist:
        gain, _ = score(trees[row["candidate_id"]], "val", args.interval, args.default_res, safety,
                        dets["val"], hulls["val"], costs, val_score_cache)
        exact[row["candidate_id"]] = gain
        row["val_gain_vs_hull"] = gain
    sweep = pd.DataFrame(candidates)
    sweep["val_gain_vs_hull"] = sweep["candidate_id"].map(exact)
    sweep = sweep.sort_values(["val_gain_vs_hull", "surrogate_gain_vs_hull"],
                              ascending=False, na_position="last")
    selected = sweep.iloc[0]
    tree = trees[int(selected.candidate_id)]
    test_gain, test_rows = score(tree, "test", args.interval, args.default_res, safety,
                                 dets["test"], hulls["test"], costs)

    artifact = {
        "kind": "sequence_previous_detections", "interval": args.interval,
        "default_res": args.default_res, "actions": ACTIONS,
        "features": FEATURES, "safety_small_people": safety,
        "reset_on_manifest_group": True, "fit_target": "crossfit_window_mean_AP",
        "fit_samples": len(X), "selected": {k: float(v) for k, v in selected.items()},
        "tree": tree, "rules": tree_rules(tree),
        "test_gain_vs_hull_mean_over_seeds": test_gain,
        "test_by_seed": [{k: v for k, v in row.items() if k != "policy"} for row in test_rows],
        "limitations": ("M3FD held-out test contains only one manifest group longer than the decision "
                        f"interval ({args.interval}); sequence-policy evidence is therefore preliminary."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(OUT / "sweep.csv", index=False)
    (OUT / "controller.json").write_text(json.dumps(artifact, indent=1))
    for row in test_rows:
        (OUT / f"policy_seed{row['seed']}_test.json").write_text(json.dumps(row["policy"]))
    print(sweep.head(8).round(5).to_string(index=False))
    print(artifact["rules"])
    print(f"TEST mean gain vs best-fixed hull: {test_gain:+.4f}")
    for row in test_rows:
        print(f"  seed {row['seed']}: AP {row['AP']:.4f}, cost {row['cost_ms']:.2f} ms, "
              f"gain {row['gain_vs_hull']:+.4f}, mix {row['mix']}")


if __name__ == "__main__":
    main()
