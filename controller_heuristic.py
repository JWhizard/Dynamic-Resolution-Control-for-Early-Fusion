"""Resolution policies behind one interface, and the depth<=3 heuristic controller.

Every policy implements `choose(frame_id, features) -> resolution`, so the
benchmark harness never needs to know which one is loaded:
    fixed:R            constant resolution (B2 / B3)
    file:<json>        precomputed {frame_id: R} (oracle B5, matched random B4)
    heuristic:<json>   decision tree over SceneAnalyzer features (H)

The tree is a small self-contained CART (Gini, exhaustive thresholds), so it has
no sklearn dependency and exports as readable if/else rules for the slides.

    python controller_heuristic.py --run ms_fusion_seed42 --eps 0.05
      fits on VAL features vs VAL oracle labels (5-fold CV over min-leaf size),
      writes results/processed/controller_eps0.05.json and the test-split policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fusion.preprocess import RESOLUTIONS

REPO = Path(__file__).resolve().parent
PROC = REPO / "results" / "processed"


# ----------------------------------------------------------------------------- policies
class Policy:
    needs_features = False
    needs_probe = False

    def choose(self, frame_id: str, features: dict | None) -> int:
        raise NotImplementedError


@dataclass
class Fixed(Policy):
    res: int

    def choose(self, frame_id, features):
        return self.res


@dataclass
class FromFile(Policy):
    table: dict

    def choose(self, frame_id, features):
        return self.table[frame_id]


def tree_features(node: dict) -> set[str]:
    if "leaf" in node:
        return set()
    return {node["feature"]} | tree_features(node["left"]) | tree_features(node["right"])


class Heuristic(Policy):
    needs_features = True

    def __init__(self, tree: dict):
        self.tree = tree
        self.features = tree_features(tree)

    def choose(self, frame_id, features):
        node = self.tree
        while "leaf" not in node:
            node = node["left"] if features[node["feature"]] <= node["threshold"] else node["right"]
        return node["leaf"]


class Cascade(Policy):
    """Probe cascade: the 320-px detections are always computed; the tree over their
    statistics (probe_features.py) decides whether to keep them or re-run at 480/640."""

    needs_probe = True

    def __init__(self, tree: dict):
        self.tree = tree

    def choose(self, frame_id, features):
        return Heuristic.choose(self, frame_id, features)


def make_policy(spec: str) -> Policy:
    kind, _, arg = spec.partition(":")
    if kind == "fixed":
        return Fixed(int(arg))
    if kind == "file":
        return FromFile({k: int(v) for k, v in json.loads(Path(arg).read_text()).items()})
    if kind == "heuristic":
        return Heuristic(json.loads(Path(arg).read_text())["tree"])
    if kind == "cascade":
        return Cascade(json.loads(Path(arg).read_text())["tree"])
    raise ValueError(spec)


# ----------------------------------------------------------------------------- CART
def _gini(y: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=k) / len(y)
    return 1.0 - float((p ** 2).sum())


def fit_tree(X: np.ndarray, y: np.ndarray, names: list[str], classes: list[int],
             max_depth: int = 3, min_leaf: int = 20) -> dict:
    """y holds class indices into `classes`. Ties in the leaf vote go to the smaller resolution."""
    k = len(classes)

    def build(idx: np.ndarray, depth: int) -> dict:
        counts = np.bincount(y[idx], minlength=k)
        leaf = {"leaf": classes[int(np.argmax(counts))], "n": int(len(idx)), "counts": counts.tolist()}
        if depth == max_depth or len(idx) < 2 * min_leaf or counts.max() == len(idx):
            return leaf
        best, parent = None, _gini(y[idx], k)
        for f in range(X.shape[1]):
            xs = X[idx, f]
            order = np.argsort(xs, kind="mergesort")
            xs_sorted, ys_sorted = xs[order], y[idx][order]
            for cut in range(min_leaf, len(idx) - min_leaf + 1):
                if xs_sorted[cut - 1] == xs_sorted[cut]:
                    continue
                l, r = ys_sorted[:cut], ys_sorted[cut:]
                g = (len(l) * _gini(l, k) + len(r) * _gini(r, k)) / len(idx)
                if best is None or g < best[0]:
                    best = (g, f, (xs_sorted[cut - 1] + xs_sorted[cut]) / 2)
        if best is None or best[0] >= parent - 1e-9:
            return leaf
        _, f, thr = best
        mask = X[idx, f] <= thr
        return {"feature": names[f], "threshold": float(thr), "n": int(len(idx)),
                "left": build(idx[mask], depth + 1), "right": build(idx[~mask], depth + 1)}

    return build(np.arange(len(y)), 0)


def fit_policy_tree(X: np.ndarray, Q: np.ndarray, cost: np.ndarray, names: list[str], classes: list[int],
                    lam: float, max_depth: int = 3, min_leaf: int = 20) -> dict:
    """Cost-sensitive policy tree. Q[i, r] = per-image quality at resolution r,
    cost[r] = measured per-frame cost of choosing r. A node's value is
    max_r sum_i (Q[i, r] - lam * cost[r]); splits maximise the children's summed
    value. Fitting on per-image quality directly avoids the noisy argmax labels
    of plain classification (which scored at the majority-class baseline)."""
    U = Q - lam * cost[None, :]

    def value(idx):
        tot = U[idx].sum(0)
        return float(tot.max()), int(tot.argmax())

    def build(idx, depth):
        v, r = value(idx)
        leaf = {"leaf": classes[r], "n": int(len(idx)), "mean_q": Q[idx].mean(0).round(4).tolist()}
        if depth == max_depth or len(idx) < 2 * min_leaf:
            return leaf
        best = None
        for f in range(X.shape[1]):
            order = np.argsort(X[idx, f], kind="mergesort")
            xs, us = X[idx, f][order], U[idx][order]
            cl = np.cumsum(us, 0)
            tot = cl[-1]
            cuts = np.arange(min_leaf, len(idx) - min_leaf + 1)
            if len(cuts) == 0:
                continue
            valid = xs[cuts - 1] != xs[np.minimum(cuts, len(xs) - 1)]
            gain = cl[cuts - 1].max(1) + (tot[None] - cl[cuts - 1]).max(1)
            gain[~valid] = -np.inf
            k = int(np.argmax(gain))
            if best is None or gain[k] > best[0]:
                best = (gain[k], f, (xs[cuts[k] - 1] + xs[cuts[k]]) / 2)
        if best is None or best[0] <= v + 1e-9:
            return leaf
        _, f, thr = best
        m = X[idx, f] <= thr
        return {"feature": names[f], "threshold": float(thr), "n": int(len(idx)),
                "left": build(idx[m], depth + 1), "right": build(idx[~m], depth + 1)}

    return build(np.arange(len(Q)), 0)


def tree_rules(node: dict, indent: str = "") -> str:
    if "leaf" in node:
        extra = f"oracle counts {node['counts']}" if "counts" in node else f"mean AP@320/480/640 {node['mean_q']}"
        return f"{indent}-> {node['leaf']} px  (n={node['n']}, {extra})\n"
    return (f"{indent}if {node['feature']} <= {node['threshold']:.4g}:\n" + tree_rules(node["left"], indent + "    ")
            + f"{indent}else:\n" + tree_rules(node["right"], indent + "    "))


# ----------------------------------------------------------------------------- fitting
def main():
    import pandas as pd

    from scene_analyzer import FEATURES

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ms_fusion_seed42")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--min-leaf", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    feats = pd.read_csv(PROC / "scene_features.csv", dtype={"id": str}).set_index("id")
    labels = json.loads((PROC / f"oracle_{args.run}_eps{args.eps}_val.json").read_text())
    ids = sorted(labels)
    X = feats.loc[ids, FEATURES].to_numpy(float)
    classes = list(RESOLUTIONS)
    y = np.array([classes.index(labels[i]) for i in ids])

    # 5-fold CV (folds over frame groups, so near-duplicates never straddle folds)
    groups = json.loads((REPO / "splits" / "m3fd_grouped_seed42.json").read_text())["groups"]
    g = np.array([groups[i] for i in ids])
    rng = np.random.default_rng(args.seed)
    ug = rng.permutation(np.unique(g))
    fold_of = {grp: k % 5 for k, grp in enumerate(ug)}
    fold = np.array([fold_of[x] for x in g])
    cv = {}
    for ml in args.min_leaf:
        acc = []
        for k in range(5):
            tr, te = fold != k, fold == k
            t = fit_tree(X[tr], y[tr], FEATURES, classes, args.max_depth, ml)
            h = Heuristic(t)
            pred = [h.choose(None, dict(zip(FEATURES, row))) for row in X[te]]
            acc.append(np.mean(np.array(pred) == np.array([classes[v] for v in y[te]])))
        cv[ml] = float(np.mean(acc))
    best_ml = max(cv, key=cv.get)
    tree = fit_tree(X, y, FEATURES, classes, args.max_depth, best_ml)
    rules = tree_rules(tree)
    majority = float(np.bincount(y).max() / len(y))
    out = {"run": args.run, "eps": args.eps, "fit_split": "val", "max_depth": args.max_depth,
           "min_leaf": best_ml, "cv_accuracy": cv, "majority_baseline": majority,
           "label_mix_val": {str(c): int((y == k).sum()) for k, c in enumerate(classes)},
           "tree": tree, "rules": rules}
    path = PROC / f"controller_eps{args.eps}.json"
    path.write_text(json.dumps(out, indent=1))

    h = Heuristic(tree)
    test_ids = feats.index[feats.split == "test"]
    policy = {i: h.choose(i, feats.loc[i, FEATURES].to_dict()) for i in test_ids}
    (PROC / f"policy_heuristic_eps{args.eps}_test.json").write_text(json.dumps(policy, indent=0))
    print(rules)
    print("CV accuracy by min_leaf", {k: round(v, 3) for k, v in cv.items()}, "majority", round(majority, 3))
    print("test mix", {r: sum(v == r for v in policy.values()) for r in classes})


if __name__ == "__main__":
    main()
