"""Per-image oracle (B5) and matched random control (B4) from the prediction cache.

oracle_eps(i) = smallest R in {320,480,640} with q(i,R) >= max_R q(i,R) - eps,
where q is per-image mAP@50-95 (evaluate.per_image_ap). Images without GT are
assigned 320 (nothing to lose). Labels are computed on VAL (controller fitting)
and TEST (upper bound for the report); never on train, where the detector's
memorisation would make low resolution look artificially safe.

B4 draws resolutions at random with exactly the heuristic's test mix
(10 draws, seeds 0..9), so "adaptation" is separated from "smaller average input".

    python generate_oracle_labels.py --run ms_fusion_seed42
    python generate_oracle_labels.py --run ms_fusion_seed42 --matched-random results/processed/policy_heuristic_eps0.05_test.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import evaluate_policy, load_run, per_image_ap
from fusion.preprocess import RESOLUTIONS

REPO = Path(__file__).resolve().parent
PROC = REPO / "results" / "processed"
EPS = (0.0, 0.02, 0.05, 0.1)


def oracle(q: pd.DataFrame, eps: float) -> dict[str, int]:
    out = {}
    for i, row in q.iterrows():
        if row.isna().all():
            out[i] = RESOLUTIONS[0]
            continue
        best = row.max()
        out[i] = next(r for r in RESOLUTIONS if row[r] >= best - eps - 1e-12)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ms_fusion_seed42")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--matched-random", default=None, help="policy json whose test mix B4 should match")
    ap.add_argument("--draws", type=int, default=10)
    args = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)

    if args.matched_random:
        target = json.loads(Path(args.matched_random).read_text())
        mix = Counter(target.values())
        ids = sorted(target)
        dets = load_run(args.run, "test")
        rows = []
        for s in range(args.draws):
            pool = np.array([r for r, n in sorted(mix.items()) for _ in range(n)])
            np.random.default_rng(s).shuffle(pool)
            pol = dict(zip(ids, map(int, pool)))
            (PROC / f"policy_random_matched_s{s}_test.json").write_text(json.dumps(pol, indent=0))
            rows.append({"draw": s, **evaluate_policy(dets, pol, "test")})
        df = pd.DataFrame(rows)
        df.to_csv(PROC / f"b4_matched_random_{args.run}_test.csv", index=False)
        print("B4 matched random mix", dict(mix), "AP %.4f +- %.4f" % (df.AP.mean(), df.AP.std()))
        return

    rows = []
    for split in args.splits:
        dets = load_run(args.run, split)
        q = pd.DataFrame({r: per_image_ap(dets[r], split) for r in RESOLUTIONS})
        q.index.name = "id"
        q.to_csv(PROC / f"per_image_ap_{args.run}_{split}.csv")
        fixed = {r: evaluate_policy(dets, dict.fromkeys(q.index, r), split) for r in RESOLUTIONS}
        for eps in EPS:
            pol = oracle(q, eps)
            (PROC / f"oracle_{args.run}_eps{eps}_{split}.json").write_text(json.dumps(pol, indent=0))
            if split == "test":
                (PROC / f"policy_oracle_eps{eps}_{split}.json").write_text(json.dumps(pol, indent=0))
            m = evaluate_policy(dets, pol, split)
            mix = Counter(pol.values())
            rows.append({"split": split, "policy": f"oracle_eps{eps}", **{f"n{r}": mix[r] for r in RESOLUTIONS}, **m})
        for r in RESOLUTIONS:
            rows.append({"split": split, "policy": f"fixed{r}", **{f"n{x}": (len(q) if x == r else 0) for x in RESOLUTIONS},
                         **fixed[r]})
        # how often does lower resolution *help*? (AdaScale's observation)
        both = q.dropna()
        print(f"[{split}] images where 320 beats 640: {(both[320] > both[640]).mean():.1%}, "
              f"480 beats 640: {(both[480] > both[640]).mean():.1%}, no-GT images: {q.isna().all(1).sum()}")
    df = pd.DataFrame(rows)
    df.to_csv(PROC / f"oracle_summary_{args.run}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(df[["split", "policy", "n320", "n480", "n640", "AP", "AP50", "AP_small", "People_recall50",
                  "mean_pixels_rel640"]].round(4))


if __name__ == "__main__":
    main()
