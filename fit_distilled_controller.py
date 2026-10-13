"""Train a distilled, cost-aware resolution controller without touching test until gates pass."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import controller_v3 as V
from controller_distilled import ACTIONS, StudentNet, thumbnail
from controller_learned import TinyNet, all_thumbnails
from fusion.preprocess import load_frame
from evaluate import evaluate_policy
from predict_cache import SRC, split_ids

REPO = Path(__file__).resolve().parent
OUT = V.PROC / "distilled"
SEED = 42


def resize_thumbs(x: np.ndarray, h: int, w: int) -> np.ndarray:
    return F.interpolate(torch.from_numpy(x).float(), (h, w), mode="bilinear", align_corners=False).byte().numpy()


def all_student_thumbnails(ids):
    path = REPO / "results" / "cache" / "thumbs_direct_32x48.npz"
    if path.exists():
        z = np.load(path)
        cached = dict(zip(z["ids"], z["x"]))
        if all(i in cached for i in ids):
            return cached
    x = []
    for i in ids:
        frame = torch.from_numpy(load_frame("fusion", f"{SRC}/Vis/{i}.png", f"{SRC}/Ir/{i}.png"))
        x.append(thumbnail(frame)[0].mul(255).round().byte().numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.asarray(ids), x=np.stack(x))
    return dict(zip(ids, x))


def pairwise_loss(pred: torch.Tensor, target: torch.Tensor, deadband: float = 0.005) -> torch.Tensor:
    losses = []
    for a in range(target.shape[1]):
        for b in range(a + 1, target.shape[1]):
            d = target[:, a] - target[:, b]
            keep = d.abs() >= deadband
            if keep.any():
                losses.append(F.softplus(-d[keep].sign() * (pred[keep, a] - pred[keep, b])).mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0.0


def group_shrunk_targets(q, ids):
    """Choose empirical-Bayes strength by within-sequence leave-one-out prediction."""
    delta = q.to_numpy(float) - q[576].to_numpy(float)[:, None]
    groups = np.array([V.GROUPS[i] for i in ids])
    global_mean = delta.mean(0)
    candidates = (2.0, 4.0, 8.0, 16.0)
    errors = {}
    for k in candidates:
        err = []
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            if len(idx) < 2:
                continue
            total = delta[idx].sum(0)
            pred = (total[None] - delta[idx] + k * global_mean) / (len(idx) - 1 + k)
            err.append(((pred - delta[idx]) ** 2).mean())
        errors[k] = float(np.mean(err))
    strength = min(errors, key=errors.get)
    out = np.empty_like(delta)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = (delta[idx].sum(0) + strength * global_mean) / (len(idx) + strength)
    return out, strength, errors


def small_people_labels(ids):
    sizes = json.loads((REPO / "splits" / "image_sizes.json").read_text())
    y = []
    for i in ids:
        W, H = sizes[i]
        found = False
        for line in (SRC / "labels_yolo" / f"{i}.txt").read_text().splitlines():
            c, _, _, w, h = line.split()[:5]
            if int(c) == 4 and float(w) * W * float(h) * H < 32 ** 2:
                found = True
                break
        y.append(float(found))
    return np.asarray(y, dtype=np.float32)


def group_bootstrap_indices(ids, rng):
    groups = np.array([V.GROUPS[i] for i in ids])
    unique = np.unique(groups)
    draw = rng.choice(unique, len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == g) for g in draw])


def fit_teacher(X, Y, Xv, Yv, seed, epochs, device):
    torch.manual_seed(seed)
    net = TinyNet(len(ACTIONS), 24).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Yt = torch.from_numpy(X).float().div(255).to(device), torch.from_numpy(Y).float().to(device)
    Xv, Yv = torch.from_numpy(Xv).float().div(255).to(device), torch.from_numpy(Yv).float().to(device)
    best, state, patience = np.inf, None, 0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        idx = torch.from_numpy(rng.permutation(len(Xt))).to(device)
        for b in idx.split(64):
            pred = net(Xt[b])
            loss = F.huber_loss(pred, Yt[b]) + 0.30 * pairwise_loss(pred, Yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            pred = net(Xv)
            val = float(F.huber_loss(pred, Yv) + 0.30 * pairwise_loss(pred, Yv))
        if val < best:
            best, patience = val, 0
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                break
    net.load_state_dict(state); return net.eval(), best


def predict_teacher(net, X, device):
    with torch.inference_mode():
        return net(torch.from_numpy(X).float().div(255).to(device)).cpu().numpy()


def fit_student(X, true_y, teacher_y, safety_y, Xv, Yv, safety_v, epochs, device):
    torch.manual_seed(SEED)
    net = StudentNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tensors = [torch.from_numpy(a).float().to(device) for a in (X / 255, true_y, teacher_y, safety_y)]
    Xv = torch.from_numpy(Xv).float().div(255).to(device)
    Yv = torch.from_numpy(Yv).float().to(device)
    Sv = torch.from_numpy(safety_v).float().to(device)
    best, state, patience = np.inf, None, 0
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(X), generator=g).split(64):
            b = b.to(device)
            pred, risk = net(tensors[0][b])
            loss = (0.50 * F.huber_loss(pred, tensors[1][b]) +
                    0.35 * F.huber_loss(pred, tensors[2][b]) +
                    0.15 * pairwise_loss(pred, tensors[1][b]) +
                    0.10 * F.binary_cross_entropy_with_logits(risk, tensors[3][b]))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step(); net.eval()
        with torch.no_grad():
            pred, risk = net(Xv)
            val = float(F.huber_loss(pred, Yv) + 0.1 * F.binary_cross_entropy_with_logits(risk, Sv))
        if val < best:
            best, patience = val, 0
            state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    net.load_state_dict(state); return net.eval(), best


def predict_student(net, X, device="cpu"):
    with torch.inference_mode():
        u, r = net(torch.from_numpy(X).float().div(255).to(device))
    return u.cpu().numpy(), torch.sigmoid(r).cpu().numpy()


def overhead_ms(model, frame, reps=500):
    torch.set_num_threads(4)
    with torch.inference_mode():
        for _ in range(50): model(thumbnail(frame))
        values = []
        for _ in range(reps):
            t = time.perf_counter(); model(thumbnail(frame)); values.append((time.perf_counter() - t) * 1e3)
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95))}


def policy_from(pred, risk, ids, costs, lam, margin, safety):
    score = pred - lam * costs[None]
    order = np.argsort(score, axis=1)
    ch = order[:, -1]
    uncertain = score[np.arange(len(ids)), order[:, -1]] - score[np.arange(len(ids)), order[:, -2]] < margin
    ch[(uncertain) | (risk >= safety)] = list(ACTIONS).index(576)
    return {i: ACTIONS[c] for i, c in zip(ids, ch)}


def hull_surrogate(cost, fixed_q, costs):
    best = -np.inf
    for r1 in V.FIXED:
        for r2 in V.FIXED:
            c1, c2 = costs[r1], costs[r2]
            if c1 - 1e-9 <= cost <= c2 + 1e-9:
                w = 0 if c1 == c2 else (cost - c1) / (c2 - c1)
                best = max(best, fixed_q[r1] + w * (fixed_q[r2] - fixed_q[r1]))
    return best


def select_operating_point(pred, risk, overhead, ids, q_val, dets, hulls, costs, safety_grid=True,
                           score_cache=None):
    score_cache = {} if score_cache is None else score_cache
    fixed_q = {r: float(q_val[r].mean()) for r in V.FIXED}
    candidates, seen = [], set()
    safety_values = (0.5, 0.7, 0.9, 1.1) if safety_grid else (1.1,)
    for lam in np.concatenate([[0.0], np.geomspace(1e-4, 2e-2, 16)]):
        for margin in (0.0, 0.002, 0.005, 0.01):
            for safety in safety_values:
                p = policy_from(pred, risk, ids, np.array([costs[r] for r in ACTIONS]), lam, margin, safety)
                sig = tuple(p[i] for i in ids)
                if sig in seen: continue
                seen.add(sig)
                c = float(np.mean([costs[r] for r in sig])) + overhead
                quality = float(np.mean([q_val.loc[i, p[i]] for i in ids]))
                candidates.append({"lam": float(lam), "margin": margin, "safety": safety, "cost": c,
                                   "surrogate": quality - hull_surrogate(c, fixed_q, costs), "policy": p})
    shortlist = sorted(candidates, key=lambda x: x["surrogate"], reverse=True)[:4]
    for row in shortlist:
        gains = []
        signature = tuple(row["policy"][i] for i in ids)
        for s in V.SEEDS:
            key = (s, signature)
            if key not in score_cache:
                score_cache[key] = V.ap_only(dets[s], row["policy"], "val")
            gains.append(score_cache[key] - hulls[s].at(row["cost"])[0])
        row["gain"] = float(np.mean(gains))
    return max(shortlist, key=lambda x: x["gain"]), shortlist


def export_models(net, calibration, out):
    net = copy.deepcopy(net).cpu().eval()
    example = torch.from_numpy(calibration[:1]).float().div(255)
    fp = torch.jit.trace(net, example)
    fp = torch.jit.freeze(fp)
    fp.save(str(out / "student_fp32.pt"))
    quantized = None
    try:
        from torch.ao.quantization import QConfigMapping, get_default_qconfig
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
        mapping = QConfigMapping().set_global(get_default_qconfig("x86"))
        prepared = prepare_fx(copy.deepcopy(net), mapping, (example,))
        with torch.inference_mode():
            for b in torch.from_numpy(calibration[:256]).float().div(255).split(32): prepared(b)
        quantized = convert_fx(prepared)
        torch.jit.trace(quantized, example).save(str(out / "student_int8.pt"))
    except Exception as exc:
        print("INT8 export unavailable:", exc)
    return fp, quantized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--teachers", type=int, default=10)
    ap.add_argument("--teacher-epochs", type=int, default=60)
    ap.add_argument("--student-epochs", type=int, default=120)
    ap.add_argument("--force-test", action="store_true", help="evaluate test despite failed gates (not recommended)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    costs = V.cost_model()
    tr, va = split_ids("train"), split_ids("val")
    thumbs = all_thumbnails()
    student_thumbs = all_student_thumbnails(tr + va + split_ids("test"))
    Xtr96, Xva96 = np.stack([thumbs[i] for i in tr]), np.stack([thumbs[i] for i in va])
    Xtr32, Xva32 = np.stack([student_thumbs[i] for i in tr]), np.stack([student_thumbs[i] for i in va])
    qtr = V.crossfit_q(ACTIONS).loc[tr]
    qva = sum(V.q_table(f"ms_fusion_seed{s}", "val", V.FIXED) for s in V.SEEDS) / len(V.SEEDS)
    target, shrink, shrink_cv = group_shrunk_targets(qtr, tr)
    val_target = qva[list(ACTIONS)].to_numpy() - qva[576].to_numpy()[:, None]
    safety_tr, safety_va = small_people_labels(tr), small_people_labels(va)

    teacher_train, teacher_val, teacher_losses = [], [], []
    rng = np.random.default_rng(SEED)
    for n in range(args.teachers):
        idx = group_bootstrap_indices(tr, rng)
        net, loss = fit_teacher(Xtr96[idx], target[idx], Xva96, val_target,
                                SEED + n, args.teacher_epochs, device)
        teacher_train.append(predict_teacher(net, Xtr96, device))
        teacher_val.append(predict_teacher(net, Xva96, device))
        teacher_losses.append(loss)
        print(f"teacher {n + 1}/{args.teachers}: val loss {loss:.5f}", flush=True)
    ttr, tva = np.mean(teacher_train, 0), np.mean(teacher_val, 0)
    student, student_loss = fit_student(Xtr32, target, ttr, safety_tr, Xva32, val_target,
                                        safety_va, args.student_epochs, device)
    student = student.cpu().eval()
    OUT.mkdir(exist_ok=True)
    fp, int8 = export_models(student, Xtr32, OUT)
    pred_fp, risk_fp = predict_student(fp, Xva32)

    sample_frame = torch.from_numpy(load_frame("fusion", f"{SRC}/Vis/00909.png", f"{SRC}/Ir/00909.png"))
    latency_fp = overhead_ms(fp, sample_frame)
    dets_val = {s: {r: V.load_cache(f"ms_fusion_seed{s}", r, "val") for r in V.FIXED} for s in V.SEEDS}
    hulls_val = {s: V.Hull(dets_val[s], "val", costs) for s in V.SEEDS}
    score_cache = {}
    teacher_sel, _ = select_operating_point(tva, np.zeros(len(va)), 0.0, va, qva, dets_val, hulls_val,
                                            costs, False, score_cache)
    student_free, _ = select_operating_point(pred_fp, risk_fp, 0.0, va, qva, dets_val, hulls_val,
                                             costs, True, score_cache)
    selected_model, selected_name, selected_pred, selected_risk, latency = fp, "student_fp32.pt", pred_fp, risk_fp, latency_fp
    fp_sel, _ = select_operating_point(pred_fp, risk_fp, latency_fp["mean"], va, qva, dets_val, hulls_val,
                                       costs, True, score_cache)
    quant_info = None
    if int8 is not None:
        pred_q, risk_q = predict_student(int8, Xva32)
        latency_q = overhead_ms(int8, sample_frame)
        corr = float(np.corrcoef(pred_fp.ravel(), pred_q.ravel())[0, 1])
        qsel, _ = select_operating_point(pred_q, risk_q, latency_q["mean"], va, qva, dets_val, hulls_val,
                                         costs, True, score_cache)
        quant_info = {"correlation": corr, "latency": latency_q, "val_gain": qsel["gain"],
                      "fp32_latency": latency_fp, "fp32_val_gain": fp_sel["gain"]}
        if (corr >= 0.99 and qsel["gain"] >= fp_sel["gain"] - 0.001 and
                latency_q["mean"] < latency_fp["mean"] and latency_q["p95"] < latency_fp["p95"]):
            selected_model, selected_name, selected_pred, selected_risk, latency = int8, "student_int8.pt", pred_q, risk_q, latency_q
    deployed, _ = select_operating_point(selected_pred, selected_risk, latency["mean"], va, qva,
                                         dets_val, hulls_val, costs, True, score_cache)
    safety_deltas = {"AP_small": [], "People_recall50": []}
    for s in V.SEEDS:
        _, mix = hulls_val[s].at(deployed["cost"])
        baseline = V.hull_mix_policy(va, mix, seed=0)
        pm = evaluate_policy(dets_val[s], deployed["policy"], "val")
        bm = evaluate_policy(dets_val[s], baseline, "val")
        for metric in safety_deltas:
            safety_deltas[metric].append(pm[metric] - bm[metric])
    safety_mean = {k: float(np.mean(v)) for k, v in safety_deltas.items()}
    gates = {
        "teacher_free_gain_ge_1AP": teacher_sel["gain"] >= 0.010,
        "student_free_gain_ge_0.8AP": student_free["gain"] >= 0.008,
        "mean_overhead_le_0.12ms": latency["mean"] <= 0.12,
        "p95_overhead_le_0.20ms": latency["p95"] <= 0.20,
        "deployed_val_gain_ge_0.5AP": deployed["gain"] >= 0.005,
        "small_AP_drop_le_1AP": safety_mean["AP_small"] >= -0.010,
        "People_recall_drop_le_1pt": safety_mean["People_recall50"] >= -0.010,
    }
    artifact = {
        "kind": "distilled_thumbnail", "model_file": selected_name, "actions": list(ACTIONS),
        "cost_ms": [costs[r] for r in ACTIONS], "lambda": deployed["lam"],
        "uncertainty_margin": deployed["margin"], "safety_threshold": deployed["safety"],
        "fallback_resolution": 576, "refresh_interval": 16, "scene_change_threshold": 0.12,
        "target": "crossfit_empirical_bayes_delta_vs_576", "shrinkage_strength": shrink,
        "shrinkage_cv_mse": shrink_cv, "teacher_count": args.teachers,
        "teacher_val_loss": teacher_losses, "student_val_loss": student_loss,
        "controller_latency_ms": latency, "quantization": quant_info,
        "validation": {"teacher_free_gain": teacher_sel["gain"], "student_free_gain": student_free["gain"],
                       "deployed_gain": deployed["gain"], "deployed_cost_ms": deployed["cost"],
                       "safety_delta": safety_mean},
        "gates": gates, "validation_pass": all(gates.values()),
        "test_evaluated": False,
    }
    (OUT / "controller.json").write_text(json.dumps(artifact, indent=1))
    (OUT / "policy_val.json").write_text(json.dumps(deployed["policy"]))
    print(json.dumps({"validation": artifact["validation"], "latency": latency, "gates": gates}, indent=1))
    if not all(gates.values()) and not args.force_test:
        print("TEST SEALED: one or more validation/latency gates failed")
        return

    te = split_ids("test")
    Xte = np.stack([student_thumbs[i] for i in te])
    pred_te, risk_te = predict_student(selected_model, Xte)
    policy = policy_from(pred_te, risk_te, te, np.array([costs[r] for r in ACTIONS]),
                         deployed["lam"], deployed["margin"], deployed["safety"])
    test_cost = float(np.mean([costs[r] for r in policy.values()])) + latency["mean"]
    dets_test = {s: {r: V.load_cache(f"ms_fusion_seed{s}", r, "test") for r in V.FIXED} for s in V.SEEDS}
    test = {}
    for s in V.SEEDS:
        hull = V.Hull(dets_test[s], "test", costs)
        ap_value = V.ap_only(dets_test[s], policy, "test")
        test[str(s)] = {"AP": ap_value, "hull_AP": hull.at(test_cost)[0],
                        "gain": ap_value - hull.at(test_cost)[0]}
    artifact["test_evaluated"] = True
    artifact["test_cost_ms"] = test_cost
    artifact["test_by_seed"] = test
    artifact["test_gain_mean"] = float(np.mean([x["gain"] for x in test.values()]))
    (OUT / "policy_test.json").write_text(json.dumps(policy))
    (OUT / "controller.json").write_text(json.dumps(artifact, indent=1))
    print("test gain mean", artifact["test_gain_mean"])


if __name__ == "__main__":
    main()
