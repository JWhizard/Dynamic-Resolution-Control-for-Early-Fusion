"""Milestone-2 figures and tables from results/ (no training or GPU needed).

    python analysis.py            -> results/figures/F1..F4 (.png/.pdf), results/processed/T1_*.csv, A1_*.csv

Latency for every fixed operating point is the MS model measured at that
resolution; the specialists share its architecture and input shapes, so their
compute is identical and their accuracy is plotted at the same latency.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import evaluate_policy, load_run
from fusion.preprocess import RESOLUTIONS

REPO = Path(__file__).resolve().parent
PROC, RAW, FIG = REPO / "results" / "processed", REPO / "results" / "raw", REPO / "results" / "figures"
MS = "ms_fusion_seed42"
SPEC = {320: "spec_fusion_320_seed42", 480: "spec_fusion_480_seed42", 640: "b2_fusion_640_seed42"}
EPS = 0.05

# reference palette (dataviz skill, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # adjacent-validated slots 1-5
ORDINAL = {320: "#86b6ef", 480: "#2a78d6", 640: "#104281"}  # blue 250 / 450 / 650
GRAY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 8, "text.color": INK,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


PLATFORMS = {"": "GPU: RTX 6000 Ada", "_cpu4": "CPU: 4 threads (Xeon w9-3495X)"}


def bench(policy: str, run: str = MS, platform: str = "") -> dict | None:
    p = RAW / f"bench_{run}_{policy.replace(':', '-').replace('/', '_')}_test{platform}.json"
    return json.loads(p.read_text()) if p.exists() else None


def policy_table(platform: str = "") -> pd.DataFrame:
    """One row per operating point: accuracy (test, offline from cache) + measured cost."""
    dets = load_run(MS, "test")
    ids = sorted(next(iter(dets.values())))
    rows = []

    def add(name, kind, policy_json, bench_policy, **extra):
        pol = json.loads(Path(policy_json).read_text()) if isinstance(policy_json, (str, Path)) else policy_json
        b = bench(bench_policy, platform=platform)
        rows.append({"name": name, "kind": kind, "platform": PLATFORMS[platform], **evaluate_policy(dets, pol, "test"), **extra,
                     **({"p50_ms": b["latency_ms"]["p50"], "p95_ms": b["latency_ms"]["p95"],
                         "p99_ms": b["latency_ms"]["p99"], "mean_ms": b["latency_ms"]["mean"],
                         "J_per_frame": b["energy"]["J_per_frame"], "peak_mem_MiB": b["peak_mem_MiB"],
                         "stage_sum_over_e2e": b["stage_sum_over_e2e"],
                         **{f"stage_{k}": v for k, v in b["stage_ms_mean"].items()}} if b else {})})

    for r in RESOLUTIONS:
        add(f"MS fixed-{r}", "fixed", dict.fromkeys(ids, r), f"fixed:{r}")
    v1 = PROC / f"policy_heuristic_eps{EPS}_test.json"
    if v1.exists():
        add("H v1: classification tree", "h_v1", v1, f"heuristic:results/processed/controller_eps{EPS}.json")
    b4 = PROC / "policy_random_matched_s0_test.json"
    if b4.exists():
        add("B4 for v1: matched random", "rand_v1", b4, f"file:results/processed/{b4.name}")
    v2 = PROC / "controller_scene_fittrain_lum_mean-lum_std.json"
    if v2.exists():
        add("H v2: cost-aware tree", "h_v2", PROC / "policy_scene_fittrain_lum_mean-lum_std_selected_test.json", f"heuristic:{v2.relative_to(REPO)}")
        add("Cost-matched random (v2)", "rand_v2", PROC / "policy_random_costmatched_scene_fittrain_lum_mean-lum_std_s0_test.json",
            "file:results/processed/policy_random_costmatched_scene_fittrain_lum_mean-lum_std_s0_test.json")
    for eps in (0.0, 0.02, 0.05, 0.1):
        p = PROC / f"policy_oracle_eps{eps}_test.json"
        if p.exists():
            add(f"Oracle (eps={eps})", "oracle", p, f"file:results/processed/policy_oracle_eps{eps}_test.json")
    df = pd.DataFrame(rows)
    b4csv = PROC / f"b4_matched_random_{MS}_test.csv"
    if b4csv.exists():
        d = pd.read_csv(b4csv)
        df.loc[df.kind == "rand_v1", ["AP_draws_mean", "AP_draws_std"]] = [d.AP.mean(), d.AP.std()]
    if v2.exists():
        r = json.loads(v2.read_text())["random_costmatched"]
        df.loc[df.kind == "rand_v2", ["AP_draws_mean", "AP_draws_std"]] = [r["AP_mean"], r["AP_std"]]
    return df


def f1_tradeoff(tables: dict[str, pd.DataFrame], e1: pd.DataFrame):
    fig, axes = plt.subplots(1, len(tables), figsize=(6.4 * len(tables), 4.6), sharey=True, squeeze=False)
    for ax, (plat, df) in zip(axes[0], tables.items()):
        _f1_panel(ax, df, e1)
        ax.set_title(PLATFORMS[plat], fontsize=11)
    axes[0][0].set_ylabel("mAP@50-95 (test)")
    axes[0][-1].legend(loc="lower right", fontsize=9)
    fig.suptitle("F1  Accuracy vs. p95 latency (batch 1, seed 42)", fontweight="bold", x=0.02, ha="left", y=1.05)
    save(fig, "F1_accuracy_latency")


def _f1_panel(ax, df: pd.DataFrame, e1: pd.DataFrame):
    fx = df[df.kind == "fixed"].sort_values("p95_ms")
    ax.plot(fx.p95_ms, fx.AP, "-o", color=BLUE, label="Multi-scale model, fixed resolution", zorder=3)
    for _, r in fx.iterrows():
        ax.annotate(r["name"].split("-")[-1] + " px", (r.p95_ms, r.AP), textcoords="offset points", xytext=(6, -12),
                    color=INK2, fontsize=9)
    lat = dict(zip(fx.name.str.split("-").str[-1].astype(int), fx.p95_ms))
    sp = e1[(e1.run.isin(SPEC.values())) & (e1.policy == e1.run.map({v: f"fixed{k}" for k, v in SPEC.items()}))]
    if len(sp):
        xs = [lat[int(p[5:])] for p in sp.policy]
        ax.plot(xs, sp.AP, "s--", color=GRAY, mfc="white", label="Resolution specialists (A1 reference)", zorder=2)
    orc = df[df.kind == "oracle"].sort_values("p95_ms")
    if len(orc):
        ax.plot(orc.p95_ms, orc.AP, "-^", color=AQUA, label="Per-image oracle (B5), eps sweep", zorder=3)
    for kind, mk, label in (("h_v1", "D", "H v1: classification tree"), ("h_v2", "*", "H v2: cost-aware tree")):
        h = df[df.kind == kind]
        if len(h):
            ax.plot(h.p95_ms, h.AP, mk, color=ORANGE, ms=13 if mk == "*" else 9, mfc=ORANGE if mk == "*" else "white",
                    mec=ORANGE, mew=2, label=label, zorder=5)
    for kind, label in (("rand_v1", "B4 matched random (v1), 10 draws"), ("rand_v2", "Cost-matched random (v2), 10 draws")):
        rnd = df[df.kind == kind]
        if len(rnd):
            ax.errorbar(rnd.p95_ms, rnd.AP_draws_mean, yerr=rnd.AP_draws_std, fmt="x" if kind == "rand_v2" else "+",
                        color=INK2, ms=9, mew=2, label=label, zorder=4)
    ax.set_xlabel("p95 end-to-end latency, ms/frame")


def f2_resolution_by_scene():
    feats = pd.read_csv(PROC / "scene_features.csv", dtype={"id": str}).set_index("id")
    pols = {"Oracle (eps=%s)" % EPS: PROC / f"policy_oracle_eps{EPS}_test.json",
            "H v2: cost-aware tree": PROC / "policy_scene_fittrain_lum_mean-lum_std_selected_test.json"}
    pols = {k: v for k, v in pols.items() if v.exists()}
    fig, axes = plt.subplots(1, len(pols), figsize=(4.2 * len(pols), 3.6), sharey=True, squeeze=False)
    scenes = ["day", "night", "adverse"]
    for ax, (name, path) in zip(axes[0], pols.items()):
        pol = pd.Series(json.loads(path.read_text()), name="res")
        d = feats.join(pol, how="inner")
        share = pd.crosstab(d.scene, d.res, normalize="index").reindex(scenes).fillna(0)
        n = d.scene.value_counts().reindex(scenes).fillna(0).astype(int)
        bottom = np.zeros(len(scenes))
        for r in RESOLUTIONS:
            v = share.get(r, pd.Series(0, index=scenes)).to_numpy()
            ax.bar(range(len(scenes)), v, bottom=bottom, color=ORDINAL[r], edgecolor="white", linewidth=2,
                   width=0.6, label=f"{r} px")
            bottom += v
        ax.set_xticks(range(len(scenes)), [f"{s}\n(n={n[s]})" for s in scenes])
        ax.set_title(name, fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("share of test frames")
    axes[0][-1].legend(title="chosen resolution", loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    fig.suptitle("F2  Which resolution each scene type gets", fontweight="bold", x=0.02, ha="left", y=1.05)
    save(fig, "F2_resolution_by_scene")


def f3_stages(df: pd.DataFrame, plat: str = ""):
    d = df[df.kind.isin(["fixed", "h_v1", "h_v2", "rand_v2"]) | (df.name == f"Oracle (eps={EPS})")].dropna(
        subset=["stage_inference"])
    stages = ["h2d", "analyze", "resize", "inference", "postprocess"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    left = np.zeros(len(d))
    for s, c in zip(stages, STAGE_COLORS):
        v = d[f"stage_{s}"].to_numpy()
        ax.barh(range(len(d)), v, left=left, color=c, edgecolor="white", linewidth=2, height=0.6, label=s)
        left += v
    for k, (tot, p95) in enumerate(zip(left, d.p95_ms)):
        ax.text(tot + 0.1, k, f"{tot:.2f} ms  (p95 {p95:.2f})", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(d)), d.name)
    ax.invert_yaxis()
    ax.set_xlabel("mean per-stage latency, ms/frame (" + ("host timer" if plat else "CUDA events") + ")")
    ax.set_xlim(0, left.max() * 1.45)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax.set_title(f"F3  Where the time goes: {PLATFORMS[plat]}", pad=28)
    save(fig, f"F3_stage_latency{plat}")


def f4_safety(e1: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True)
    ms = e1[e1.run == MS].assign(res=lambda d: d.policy.str[5:].astype(int)).sort_values("res")
    sp = e1[e1.run.isin(SPEC.values())].assign(res=lambda d: d.policy.str[5:].astype(int))
    sp = sp[sp.run == sp.res.map(SPEC)].sort_values("res")
    for ax, col, title in zip(axes, ["AP_small", "People_recall50"],
                              ["Small-object AP (area < 32² px native)", "People recall (IoU 0.5, conf 0.25)"]):
        ax.plot(ms.res, ms[col], "-o", color=BLUE, label="Multi-scale model")
        if len(sp):
            ax.plot(sp.res, sp[col], "s--", color=GRAY, mfc="white", label="Specialists")
        for x, y in zip(ms.res, ms[col]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9,
                        color=INK2)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(ms[col].max(), sp[col].max() if len(sp) else 0) * 1.2)
        ax.set_xticks(list(RESOLUTIONS))
        ax.set_xlabel("input resolution (long side, px)")
    axes[0].legend(fontsize=9)
    fig.suptitle("F4  Safety: what lower resolution costs small objects and pedestrians", fontweight="bold", x=0.02,
                 ha="left", y=1.05)
    save(fig, "F4_small_object_safety")


def f5_examples(n: int = 2):
    """Test frames where 640 px finds the most small pedestrians that 320 px misses (conf >= 0.25, IoU 0.5)."""
    import cv2
    from ultralytics.utils.metrics import box_iou
    import torch

    from evaluate import SRC, ground_truth

    gt, ids = ground_truth("test")
    dets = load_run(MS, "test")
    people = NAMES_PEOPLE = 4

    def hits(i, k, r):
        g = [a["bbox"] for a in gt.loadAnns(gt.getAnnIds(imgIds=[k], catIds=[people + 1])) if a["area"] < 32 ** 2]
        d = dets[r][i]
        d = d[(d[:, 4] >= 0.25) & (d[:, 5] == people)]
        if not g or not len(d):
            return 0, len(g)
        g = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in g])
        return int((box_iou(g, torch.from_numpy(d[:, :4])).max(1).values >= 0.5).sum()), len(g)

    scored = []
    for k, i in enumerate(ids):
        h640, ng = hits(i, k, 640)
        h320, _ = hits(i, k, 320)
        scored.append((h640 - h320, ng, k, i))
    picks = sorted(scored, reverse=True)[:n]
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.1 * n), squeeze=False)
    for row, (_, ng, k, i) in zip(axes, picks):
        img = cv2.cvtColor(cv2.imread(str(SRC / "Vis" / f"{i}.png")), cv2.COLOR_BGR2RGB)
        for ax, r in zip(row, (320, 640)):
            ax.imshow(img)
            for a in gt.loadAnns(gt.getAnnIds(imgIds=[k])):
                x, y, w, h = a["bbox"]
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="white", lw=1.2, ls="--"))
            d = dets[r][i]
            for x1, y1, x2, y2, c, cl in d[d[:, 4] >= 0.25]:
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                           ec=ORANGE if int(cl) == people else BLUE, lw=1.6))
            got, _ = hits(i, k, r)
            ax.set_title(f"frame {i} @ {r} px: {got}/{ng} small pedestrians found", fontsize=10)
            ax.axis("off")
    fig.suptitle("Detections at conf ≥ 0.25 (orange = People, blue = other; dashed white = ground truth)",
                 fontweight="bold", x=0.02, ha="left", y=1.0)
    save(fig, "F5_detection_examples")


def f6_design_space():
    """Every controller variant tried, on the modelled CPU cost (mean ms/frame, 4 threads): test AP of each
    lambda in each sweep, against the fixed-resolution frontier and its random-mix interpolation."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ctl = json.loads((PROC / "controller_scene_fittrain_lum_mean-lum_std.json").read_text())
    base = ctl["fixed_cost_ms"]
    fx = [ctl["test_fixed_AP"][str(r)] for r in RESOLUTIONS]
    ax.plot(base, fx, "-o", color=BLUE, label="Fixed resolution (random mixes lie on these segments)", zorder=3)
    for c, a, r in zip(base, fx, RESOLUTIONS):
        ax.annotate(f"{r} px", (c, a), textcoords="offset points", xytext=(6, -12), fontsize=9, color=INK2)
    o = pd.read_csv(PROC / f"oracle_summary_{MS}.csv")
    o = o[(o.split == "test") & o.policy.str.startswith("oracle")]
    oc = (o.n320 * base[0] + o.n480 * base[1] + o.n640 * base[2]) / (o.n320 + o.n480 + o.n640)
    ax.plot(oc, o.AP, "-^", color=AQUA, label="Per-image oracle (upper bound)", zorder=3)
    sweeps = [("policy_sweep_scene_cpu4.csv", "Scene tree, fit on val", GRAY, ":"),
              ("policy_sweep_probe320_cpu4.csv", "Probe cascade (320 px probe)", INK2, "--"),
              ("policy_sweep_probe192_cpu4.csv", "Probe cascade (192 px probe)", INK2, "-."),
              ("policy_sweep_scene_fittrain_cpu4.csv", "Scene tree (all features), fit on train", GRAY, "--"), ("policy_sweep_scene_fittrain_lum_mean-lum_std_cpu4.csv", "Luminance tree, fit on train (H v2)", ORANGE, "-")]
    for f, label, col, ls in sweeps:
        if (PROC / f).exists():
            d = pd.read_csv(PROC / f).drop_duplicates(["test_cost_ms", "test_AP"]).sort_values("test_cost_ms")
            ax.plot(d.test_cost_ms, d.test_AP, ls, color=col, lw=1.6, label=label, zorder=2)
            sel = d[d.selected]
            ax.plot(sel.test_cost_ms, sel.test_AP, "o", color=col, ms=8, mec="white", mew=1.5, zorder=4)
    ax.set_xlabel("expected CPU cost, ms/frame (4 threads; measured per-resolution latencies)")
    ax.set_ylabel("mAP@50-95 (test)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("F6  Controller design space (dots = operating point selected on val)")
    save(fig, "F6_controller_design_space")


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    e1 = pd.read_csv(PROC / "e1_fixed_test.csv")
    tables = {p: policy_table(p) for p in PLATFORMS}
    tables = {p: t for p, t in tables.items() if "p95_ms" in t}
    df = pd.concat(tables.values(), ignore_index=True)
    cols = ["name", "AP", "AP50", "AP_small", "AP_People", "People_recall50", "mean_pixels_rel640", "p50_ms",
            "p95_ms", "p99_ms", "J_per_frame", "peak_mem_MiB", "AP_draws_mean", "AP_draws_std"]
    df.to_csv(PROC / "T1_policies_test.csv", index=False)
    print(df[[c for c in cols if c in df]].round(4).to_string(index=False))

    a1 = e1[e1.run.isin([MS, *SPEC.values()])].pivot_table(index="policy", columns="run", values="AP")
    a1.to_csv(PROC / "A1_multiscale_vs_specialists_test.csv")
    print("\nA1 (AP by eval resolution):\n", a1.round(4))
    base = e1[e1.policy == "fixed640"].set_index("run")[["AP", "AP50", "AP_small", "AP_People", "People_recall50"]]
    base.to_csv(PROC / "T0_baselines_640_test.csv")
    print("\nBaselines @640:\n", base.round(4))

    seeds = PROC / "e1_fixed_test_seeds.csv"
    if seeds.exists():
        d = pd.read_csv(seeds).assign(model=lambda d: d.run.str.replace(r"_seed\d+", "", regex=True))
        t2 = d.groupby(["model", "policy"])[["AP", "AP50", "AP_small", "People_recall50"]].agg(["mean", "std", "count"])
        t2.to_csv(PROC / "T2_multiseed_fixed_test.csv")
        print("\nT2 (3 seeds, mean/std):\n", t2.round(4).to_string())

    f1_tradeoff(tables, e1)
    f2_resolution_by_scene()
    for plat, t in tables.items():
        f3_stages(t, plat)
    f4_safety(e1)
    f5_examples()
    if (PROC / "controller_scene_fittrain_lum_mean-lum_std.json").exists():
        f6_design_space()


if __name__ == "__main__":
    main()
