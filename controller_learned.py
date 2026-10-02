"""Learned resolution controller (RQ3): a tiny CNN on the 4-channel thumbnail predicts per-image utility.

The thumbnail is a strided subsample of the native B,G,R,IR frame resized to 96x128, so it costs
almost nothing to form at run time. Unlike whole-frame statistics, a CNN can respond to small warm IR
blobs and dense far-field structure, which decide how much a frame gains from resolution.

Targets on train (choose with --target): "crossfit" (out-of-fold per-image AP from two half-train
detectors) or "seedavg" (mean per-image AP of 3 MS seeds), optionally shrunk toward the sequence mean.
Decision: argmax_r  q_hat(r) - lam * cost(r); lam selected on VAL by mean gain over each seed's
best-fixed hull; test scored once on 3 seeds with a sequence-bootstrap CI (machinery from controller_v3).

    python controller_learned.py --target crossfit --model cnn
    python controller_learned.py --target crossfit --model knn
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import controller_v3 as V
from fusion.preprocess import load_frame
from predict_cache import SRC, split_ids

REPO = Path(__file__).resolve().parent
THUMBS = REPO / "results" / "cache" / "thumbs_96x128.npz"
TH, TW = 96, 128


def thumbnail(frame: torch.Tensor, th: int = TH, tw: int = TW) -> torch.Tensor:
    """(H,W,4) uint8 -> (4,th,tw) float in [0,1]: strided subsample, then a small resize."""
    step = max(1, round(max(frame.shape[:2]) / tw))
    x = frame[::step, ::step].permute(2, 0, 1)[None].float().div_(255)
    return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)[0]


def controller_overhead_ms(model: str, width: int, size: int, ensemble: int, n_train: int, reps: int = 300) -> float:
    """Measured per-frame controller cost on 4 CPU threads (thumbnail + inference), mean ms."""
    import time

    from scene_analyzer import load_fusion_frame

    torch.set_num_threads(4)
    th, tw = (size, size * 4 // 3)
    f = torch.from_numpy(load_fusion_frame("00909"))
    if model == "cnn":
        nets = [TinyNet(len(V.ACTIONS), width).eval() for _ in range(ensemble)]
        run = lambda: [n(thumbnail(f, th, tw)[None]) for n in nets]
    else:
        bank = torch.randn(n_train, 4 * 12 * 16)
        run = lambda: torch.cdist(F.adaptive_avg_pool2d(thumbnail(f, th, tw)[None], (12, 16)).flatten(1), bank).topk(50, largest=False)
    with torch.no_grad():
        for _ in range(30):
            run()
        t = []
        for _ in range(reps):
            a = time.perf_counter()
            run()
            t.append((time.perf_counter() - a) * 1e3)
    return float(np.mean(t))


def all_thumbnails() -> dict[str, np.ndarray]:
    if THUMBS.exists():
        z = np.load(THUMBS)
        return dict(zip(z["ids"], z["x"]))
    ids = [i for s in ("train", "val", "test") for i in split_ids(s)]
    x = np.stack([(thumbnail(torch.from_numpy(load_frame("fusion", f"{SRC}/Vis/{i}.png", f"{SRC}/Ir/{i}.png")))
                   .mul(255).round().byte().numpy()) for i in ids])
    THUMBS.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(THUMBS, ids=np.array(ids), x=x)
    return dict(zip(ids, x))


class TinyNet(nn.Module):
    def __init__(self, n_out: int, w: int = 24):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(block(4, w), block(w, 2 * w), block(2 * w, 2 * w), block(2 * w, 4 * w))
        self.head = nn.Linear(4 * w, n_out)

    def forward(self, x):
        return self.head(self.f(x).mean((2, 3)))


def fit_cnn(X, Y, Xv, Yv, seed=0, epochs=80, device="cpu", width=24):
    """Regress per-action utility (centred per image so the net learns the *differences*)."""
    torch.manual_seed(seed)
    net = TinyNet(Y.shape[1], width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.from_numpy(X).float().div(255).to(device)
    Yt = torch.from_numpy(Y - Y.mean(1, keepdims=True)).float().to(device)
    Xvt = torch.from_numpy(Xv).float().div(255).to(device)
    Yvt = torch.from_numpy(Yv - Yv.mean(1, keepdims=True)).float().to(device)
    best, state = 1e9, None
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        net.train()
        for idx in torch.randperm(len(Xt), generator=g).split(64):
            xb = Xt[idx]
            if torch.rand(1, generator=g).item() < 0.5:
                xb = xb.flip(-1)
            loss = F.mse_loss(net(xb), Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = F.mse_loss(net(Xvt), Yvt).item()
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    net.eval()
    return net, best


def predict(net, X, device="cpu"):
    with torch.no_grad():
        return net(torch.from_numpy(X).float().div(255).to(device)).cpu().numpy()


def knn_predict(Xtr, Ytr, Xq, k=50):
    """Mean utility of the k nearest train thumbnails (z-scored 12x16 pooled pixels)."""
    def emb(X):
        t = torch.from_numpy(X).float()
        return F.adaptive_avg_pool2d(t, (12, 16)).flatten(1).numpy()
    a, b = emb(Xtr), emb(Xq)
    mu, sd = a.mean(0), a.std(0) + 1e-6
    a, b = (a - mu) / sd, (b - mu) / sd
    d = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    nn_idx = np.argsort(d, axis=1)[:, :k]
    Yc = Ytr - Ytr.mean(1, keepdims=True)
    return Yc[nn_idx].mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["crossfit", "seedavg"], default="crossfit")
    ap.add_argument("--model", choices=["cnn", "knn"], default="cnn")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5, 0.0])
    ap.add_argument("--ensemble", type=int, default=5, help="CNN seeds averaged")
    ap.add_argument("--boot", type=int, default=0, help="legacy random-mix bootstrap; use bootstrap_vs_hull.py")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed-offset", type=int, default=0, help="CNN init seeds = offset..offset+ensemble-1")
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--size", type=int, default=96, choices=[96, 48], help="thumbnail height (width = 4/3 x)")
    ap.add_argument("--no-overhead", action="store_true", help="ignore controller cost (legacy comparison)")
    args = ap.parse_args()
    torch.set_num_threads(8)
    C = V.cost_model()
    A = V.ACTIONS
    cost = np.array([C[r] for r in A])
    tr, va, te = (split_ids(s) for s in ("train", "val", "test"))
    th = all_thumbnails()
    Xtr, Xva, Xte = (np.stack([th[i] for i in ids]) for ids in (tr, va, te))
    if args.size == 48:  # 2x average-pool of the stored 96x128 thumbnails
        Xtr, Xva, Xte = (F.avg_pool2d(torch.from_numpy(x).float(), 2).round().byte().numpy() for x in (Xtr, Xva, Xte))
    ens = args.ensemble if args.model == "cnn" else 1
    overhead = 0.0 if args.no_overhead else controller_overhead_ms(args.model, args.width, args.size, ens, len(tr))
    print(f"controller overhead (CPU, 4 threads): {overhead:.3f} ms/frame", flush=True)

    q_tr = (V.crossfit_q(A) if args.target == "crossfit"
            else sum(V.q_table(f"ms_fusion_seed{s}", "train", A) for s in V.SEEDS) / len(V.SEEDS)).loc[tr]
    q_va = (sum(V.q_table(f"ms_fusion_seed{s}", "val", A) for s in V.SEEDS) / len(V.SEEDS)).loc[va]

    dets = {sp: {s: {r: V.load_cache(f"ms_fusion_seed{s}", r, sp) for r in V.FIXED} for s in V.SEEDS}
            for sp in ("val", "test")}
    hulls = {sp: {s: V.Hull(dets[sp][s], sp, C) for s in V.SEEDS} for sp in ("val", "test")}

    rows, preds, saved_nets = [], {}, {}
    for a in args.alphas:
        Y = V.shrink(q_tr, a).to_numpy()
        if args.model == "cnn":
            nets = [fit_cnn(Xtr, Y, Xva, q_va.to_numpy(), seed=s, device=args.device, width=args.width)[0]
                    for s in range(args.seed_offset, args.seed_offset + args.ensemble)]
            saved_nets[a] = [{k: v.cpu() for k, v in n.state_dict().items()} for n in nets]
            pv = np.mean([predict(n, Xva, args.device) for n in nets], 0)
            pt = np.mean([predict(n, Xte, args.device) for n in nets], 0)
        else:
            pv, pt = knn_predict(Xtr, Y, Xva), knn_predict(Xtr, Y, Xte)
        preds[a] = (pv, pt)
        corr = np.mean([np.corrcoef(pv[:, j] - pv[:, -1], (q_va.to_numpy()[:, j] - q_va.to_numpy()[:, -1]))[0, 1]
                        for j in range(len(A) - 1)])
        for lam in np.concatenate([[0.0], np.geomspace(1e-4, 2e-2, 20)]):
            ch = np.argmax(pv - lam * cost[None], 1)
            pol = {i: A[c] for i, c in zip(va, ch)}
            cv_ = float(np.mean(cost[ch]))
            gain = np.mean([V.ap_only(dets["val"][s], pol, "val") - hulls["val"][s].at(cv_ + overhead)[0]
                            for s in V.SEEDS])
            rows.append({"alpha": a, "lam": lam, "val_cost_ms": cv_, "val_gain_vs_hull": float(gain),
                         "val_pred_corr": float(corr), **{f"val_n{r}": int((ch == k).sum()) for k, r in enumerate(A)}})
    import pandas as pd
    sweep = pd.DataFrame(rows)
    sel = sweep.loc[sweep.val_gain_vs_hull.idxmax()]
    pt = preds[sel.alpha][1]
    ch = np.argmax(pt - sel.lam * cost[None], 1)
    pol_t = {i: A[c] for i, c in zip(te, ch)}
    ct = float(np.mean(cost[ch])) + overhead  # total CPU cost includes the controller itself
    res = {}
    for s in V.SEEDS:
        hap, mix = hulls["test"][s].at(ct)
        res[s] = {"AP": V.ap_only(dets["test"][s], pol_t, "test"), "hull_AP": hap, "hull_mix": list(mix)}
    gain = float(np.mean([v["AP"] - v["hull_AP"] for v in res.values()]))

    test_groups = sorted({V.GROUPS[i] for i in te})
    by_group = {grp: [i for i in te if V.GROUPS[i] == grp] for grp in test_groups}
    rng = np.random.default_rng(0)
    gt, _ = V.ground_truth("test")
    diffs = []
    for _ in range(args.boot):
        draw = rng.choice(test_groups, len(test_groups), replace=True)
        ids_b = [i for grp in draw for i in by_group[grp]]
        diffs.append(np.mean([V._ap_on(gt, dets["test"][s], pol_t, ids_b)
                              - V._ap_on(gt, dets["test"][s], V.hull_mix_policy(te, res[s]["hull_mix"]), ids_b)
                              for s in V.SEEDS]))
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if diffs else (float("nan"), float("nan"))
    tag = (f"learned_{args.model}_{args.target}" + (f"_off{args.seed_offset}" if args.seed_offset else "")
           + (f"_w{args.width}_s{args.size}_e{ens}" if (args.width, args.size, ens) != (24, 96, 5) or args.model == "knn" else "")
           + ("" if not args.no_overhead else "_nooverhead"))
    if saved_nets:
        torch.save({"width": args.width, "size": args.size, "actions": A, "lam": float(sel.lam),
                    "nets": saved_nets[sel.alpha]}, V.PROC / "v3" / f"{tag}_nets.pt")
    out = {"model": args.model, "target": args.target, "actions": A, "selected": {k: float(v) for k, v in sel.to_dict().items()},
           "test_cost_ms": ct, "controller_overhead_ms": overhead, "test_mix": {str(r): int((ch == k).sum()) for k, r in enumerate(A)},
           "test_by_seed": {str(s): v for s, v in res.items()}, "test_gain_vs_hull_mean_over_seeds": gain,
           "bootstrap_gain_vs_hull_mix_95ci": [float(lo), float(hi)],
           "params": sum(p.numel() for p in TinyNet(len(A), args.width).parameters()) if args.model == "cnn" else None}
    (V.PROC / "v3").mkdir(exist_ok=True)
    sweep.to_csv(V.PROC / "v3" / f"sweep_{tag}.csv", index=False)
    (V.PROC / "v3" / f"controller_{tag}.json").write_text(json.dumps(out, indent=1))
    (V.PROC / "v3" / f"policy_{tag}_test.json").write_text(json.dumps(pol_t))
    print(sweep.sort_values("val_gain_vs_hull", ascending=False).head(6).round(4).to_string(index=False))
    print(f"TEST {tag}: cost {ct:.2f} ms, mix {out['test_mix']}, gain vs best-fixed hull {gain:+.4f}, "
          f"bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
    for s, v in res.items():
        print(f"  seed {s}: AP {v['AP']:.4f}  hull {v['hull_AP']:.4f}")


if __name__ == "__main__":
    main()
