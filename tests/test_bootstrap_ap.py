"""Fast bootstrap AP (stored match records) must equal a full pycocotools re-evaluation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller_v3 as V
from bootstrap_vs_hull import MatchStore


def test_fast_ap_matches_pycocotools():
    gt, ids = V.ground_truth("test")
    dets = {r: V.load_cache("ms_fusion_seed42", r, "test") for r in (480, 640)}
    pol = {i: (480 if k % 3 else 640) for k, i in enumerate(ids)}
    store = MatchStore(gt, dets, pol, ids)
    full = store.ap(np.arange(len(ids)))
    ref = V.ap_only(dets, pol, "test")
    assert abs(full - ref) < 1e-9, (full, ref)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(ids), len(ids), replace=True)  # with duplicates, like a bootstrap resample
    fast = store.ap(idx)
    slow = V._ap_on(gt, dets, pol, [ids[k] for k in idx])
    assert abs(fast - slow) < 1e-9, (fast, slow)
    print(f"full {full:.6f} == {ref:.6f}; resample {fast:.6f} == {slow:.6f}")


if __name__ == "__main__":
    test_fast_ap_matches_pycocotools()
