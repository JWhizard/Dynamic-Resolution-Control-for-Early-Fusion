"""Convert M3FD to four-channel (B,G,R,IR) TIFF and build a sequence-grouped split.

M3FD frames are sampled from driving/surveillance sequences, so neighbouring
frames are near-duplicates (median nearest-neighbour thumbnail correlation ~0.96).
A random split therefore leaks scenes from train into test. Frames are grouped by
connected components of a thumbnail-correlation graph, and whole groups are
assigned to train/val/test (stratified by dominant class).

Outputs (under --out):
    fusion/images/{split}/*.tiff   4-channel uint8, channel order B,G,R,IR
    rgb/images/{split}/*.png       symlinks to Vis
    ir/images/{split}/*.png        symlinks to Ir
    */labels/{split}/*.txt         symlinks to labels_yolo
    {fusion,rgb,ir}.yaml           Ultralytics dataset configs
and in this repo:
    splits/m3fd_grouped_seed42.json (+ .sha256), splits/group_contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

NAMES = ["Bus", "Car", "Lamp", "Motorcycle", "People", "Truck"]
SPLITS = ("train", "val", "test")
REPO = Path(__file__).resolve().parent


def thumb(path: str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    t = cv2.resize(g, (32, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (t - t.mean()) / (t.std() + 1e-6)


def group_frames(src: Path, ids: list[str], thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (group label per frame, frame x frame thumbnail correlation)."""
    x = np.stack([np.concatenate([thumb(f"{src}/Vis/{i}.png"), thumb(f"{src}/Ir/{i}.png")]) for i in ids])
    corr = (x @ x.T) / x.shape[1]
    np.fill_diagonal(corr, 0)
    _, labels = connected_components(csr_matrix(corr > thresh), directed=False)
    return labels, corr


def dominant_class(label_file: Path) -> int:
    cls = [int(l.split()[0]) for l in label_file.read_text().splitlines() if l.strip()]
    return Counter(cls).most_common(1)[0][0] if cls else -1


def split_groups(groups: np.ndarray, dom: list[int], fracs, seed: int) -> dict[int, str]:
    """Greedy stratified group split: within each stratum (group's dominant class),
    visit groups largest-first (random tie order) and give each to the split
    furthest below its target frame count."""
    rng = np.random.default_rng(seed)
    sizes = np.bincount(groups)
    gdom = {}
    for g in range(len(sizes)):
        members = np.flatnonzero(groups == g)
        gdom[g] = Counter(dom[m] for m in members).most_common(1)[0][0]
    assign = {}
    for stratum in sorted(set(gdom.values())):
        gs = [g for g in gdom if gdom[g] == stratum]
        rng.shuffle(gs)
        gs.sort(key=lambda g: -sizes[g])
        total = sizes[gs].sum()
        filled = dict.fromkeys(SPLITS, 0)
        for g in gs:
            deficit = {s: f * total - filled[s] for s, f in zip(SPLITS, fracs)}
            s = max(deficit, key=deficit.get)
            assign[g] = s
            filled[s] += sizes[g]
    return assign


def leakage(corr: np.ndarray, split: dict[str, str], ids: list[str], thresh: float = 0.95) -> float:
    """Fraction of test frames whose most similar train frame exceeds `thresh`."""
    idx = {i: k for k, i in enumerate(ids)}
    tr = [idx[i] for i in ids if split[i] == "train"]
    te = [idx[i] for i in ids if split[i] == "test"]
    return float((corr[np.ix_(te, tr)].max(1) > thresh).mean())


def random_split(ids, dom, fracs, seed):
    """The proposal's original stratified random split, used only to measure leakage."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in sorted(set(dom)):
        m = [i for i, d in zip(ids, dom) if d == c]
        rng.shuffle(m)
        a, b = int(fracs[0] * len(m)), int((fracs[0] + fracs[1]) * len(m))
        out.update({i: ("train" if k < a else "val" if k < b else "test") for k, i in enumerate(m)})
    return out


def contact_sheet(src: Path, ids, groups, path: Path, n_groups=8, per=6, seed=0):
    rng = np.random.default_rng(seed)
    big = [g for g in np.argsort(-np.bincount(groups)) if (groups == g).sum() >= per][:40]
    rows = []
    for g in rng.choice(big, size=min(n_groups, len(big)), replace=False):
        members = np.flatnonzero(groups == g)
        pick = np.sort(rng.choice(members, per, replace=False))
        row = [cv2.resize(cv2.imread(f"{src}/Vis/{ids[m]}.png"), (160, 120)) for m in pick]
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_fourchannel(src: Path, i: str, dst: Path) -> None:
    vis = cv2.imread(f"{src}/Vis/{i}.png", cv2.IMREAD_COLOR)  # B,G,R
    ir = cv2.imread(f"{src}/Ir/{i}.png", cv2.IMREAD_GRAYSCALE)
    assert vis.shape[:2] == ir.shape, i
    cv2.imwrite(str(dst), np.dstack([vis, ir]))


def check_roundtrip(src: Path, i: str, tiff: Path) -> None:
    from ultralytics.utils.patches import imread

    got = imread(str(tiff))
    vis = cv2.imread(f"{src}/Vis/{i}.png", cv2.IMREAD_COLOR)
    ir = cv2.imread(f"{src}/Ir/{i}.png", cv2.IMREAD_GRAYSCALE)
    assert got.shape == (*ir.shape, 4), got.shape
    assert np.array_equal(got[..., :3], vis) and np.array_equal(got[..., 3], ir), "TIFF channel order mismatch"


def link(target: Path, name: Path) -> None:
    if name.is_symlink() or name.exists():
        name.unlink()
    name.symlink_to(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/12TB_Drive/whisenaj/Datasets/M3FD_Detection")
    ap.add_argument("--out", default="/mnt/12TB_Drive/whisenaj/Datasets/M3FD_4ch")
    ap.add_argument("--thresh", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fracs", type=float, nargs=3, default=(0.7, 0.2, 0.1))
    ap.add_argument("--skip-images", action="store_true", help="only (re)build split and links")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)

    ids = sorted(p.stem for p in (src / "Vis").glob("*.png"))
    dom = [dominant_class(src / "labels_yolo" / f"{i}.txt") for i in ids]
    groups, corr = group_frames(src, ids, args.thresh)
    assign = split_groups(groups, dom, args.fracs, args.seed)
    split = {i: assign[g] for i, g in zip(ids, groups)}

    for a in SPLITS:  # leakage check: no group spans two splits
        for b in SPLITS:
            if a < b:
                ga = {g for i, g in zip(ids, groups) if split[i] == a}
                gb = {g for i, g in zip(ids, groups) if split[i] == b}
                assert not ga & gb, f"group leakage between {a} and {b}"

    counts = Counter(split.values())
    stats = {
        "n_groups": int(groups.max() + 1),
        "largest_group": int(np.bincount(groups).max()),
        "counts": {s: counts[s] for s in SPLITS},
        "test_near_duplicate_rate_grouped": leakage(corr, split, ids),
        "test_near_duplicate_rate_random": leakage(corr, random_split(ids, dom, args.fracs, args.seed), ids),
    }
    manifest = {
        "description": "M3FD sequence-grouped stratified split",
        "grouping": f"connected components of 32x24 Vis+IR thumbnail correlation > {args.thresh}",
        "seed": args.seed,
        "fractions": list(args.fracs),
        "stats": stats,
        "splits": {s: sorted(i for i in ids if split[i] == s) for s in SPLITS},
        "groups": {i: int(g) for i, g in zip(ids, groups)},
    }
    (REPO / "splits").mkdir(exist_ok=True)
    mpath = REPO / "splits" / f"m3fd_grouped_seed{args.seed}.json"
    mpath.write_text(json.dumps(manifest, indent=1))
    (mpath.with_suffix(".sha256")).write_text(hashlib.sha256(mpath.read_bytes()).hexdigest() + "\n")
    contact_sheet(src, ids, groups, REPO / "splits" / "group_contact_sheet.jpg")
    print(json.dumps(stats, indent=1))

    for view in ("fusion", "rgb", "ir"):
        for s in SPLITS:
            (out / view / "images" / s).mkdir(parents=True, exist_ok=True)
            (out / view / "labels" / s).mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(ids):
        s = split[i]
        tiff = out / "fusion" / "images" / s / f"{i}.tiff"
        if not args.skip_images:
            write_fourchannel(src, i, tiff)
            if k < 5:
                check_roundtrip(src, i, tiff)
        link(src / "Vis" / f"{i}.png", out / "rgb" / "images" / s / f"{i}.png")
        link(src / "Ir" / f"{i}.png", out / "ir" / "images" / s / f"{i}.png")
        for view in ("fusion", "rgb", "ir"):
            link(src / "labels_yolo" / f"{i}.txt", out / view / "labels" / s / f"{i}.txt")
    # remove stale files from a previous split assignment
    for view in ("fusion", "rgb", "ir"):
        for s in SPLITS:
            for d in ("images", "labels"):
                for f in (out / view / d / s).iterdir():
                    if split.get(f.stem) != s:
                        f.unlink()

    for view, ch in (("fusion", 4), ("rgb", 3), ("ir", 3)):
        lines = [f"path: {out / view}"] + [f"{s}: images/{s}" for s in SPLITS] + [f"channels: {ch}", "names:"]
        lines += [f"  {k}: {n}" for k, n in enumerate(NAMES)]
        (out / f"{view}.yaml").write_text("\n".join(lines) + "\n")
    print("done:", out)


if __name__ == "__main__":
    main()
