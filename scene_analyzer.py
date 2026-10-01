"""Scene Analyzer: cheap per-frame statistics that predict resolution sensitivity.

Input: native uint8 HWC frame on the GPU, channels B,G,R,IR (usually 768x1024x4;
274 of 4200 M3FD frames are smaller, 400x280 to 880x520).
Features (proposal component 1, with two corrections):
  * lighting / contrast on a 64x48 thumbnail: RGB luminance mean and std,
    IR p95-p5 spread, RGB dark-channel mean (haze), Sobel edge energy and
    Laplacian variance per modality;
  * spectral detail on four NATIVE-resolution 64x64 patches: share of FFT energy
    above the Nyquist limit of 320-px (and 480-px) sampling. A 64-px thumbnail
    cannot contain those frequencies, so the proposal's thumbnail version was
    ill-defined;
  * IR object-size proxy from connected components of a thresholded 128x96 IR
    saliency map (small objects vanish at 64x48). This is the only CPU step; the
    12 KB device-to-host copy is included in the measured cost.

    python scene_analyzer.py --bench      # microbenchmark (target < 1 ms/frame)
    python scene_analyzer.py --extract    # features for all frames -> results/processed/scene_features.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
PATCH = 64
FEATURES = [
    "lum_mean", "lum_std", "ir_spread", "dark_channel",
    "edge_rgb", "edge_ir", "lapvar_rgb", "lapvar_ir",
    "hf320_rgb", "hf320_ir", "hf480_rgb", "hf480_ir",
    "ir_blobs", "ir_blob_median_area", "ir_blob_min_area",
]

_SOBEL = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]])
_LAP = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]])


LIGHT = {"lum_mean", "lum_std"}
FAST = ["lum_mean_s", "lum_std_s"]  # strided-subsample luminance: ~free; used by controller v3


def fast_luminance(frame: torch.Tensor) -> dict[str, float]:
    """Mean / std of luminance over a strided 64-sample-wide grid of the native frame (HWC, B,G,R,...).
    Reads ~3k of ~786k pixels; on a real sensor the auto-exposure statistics give the same for free."""
    step = max(1, round(max(frame.shape[:2]) / 64))
    x = frame[::step, ::step, :3].float()
    lum = x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299
    m, sd = torch.stack([lum.mean(), lum.std()]).tolist()
    return {"lum_mean_s": m, "lum_std_s": sd}


class SceneAnalyzer:
    """`needed` restricts the work to the features a controller actually reads. When only
    luminance statistics are needed (the deployed rule uses lum_mean alone) the analyzer
    takes a light path: one area-downsample of B,G,R to 64x48 and two reductions, the
    same values as the full path up to float rounding (area pooling and the luminance
    weights are both linear, so they commute)."""

    def __init__(self, device: torch.device | str = "cuda", needed=None):
        self.device = torch.device(device)
        self.needed = set(needed) if needed else set(FEATURES)
        self._w = torch.tensor([0.114, 0.587, 0.299], device=self.device).view(1, 3, 1, 1)
        k = torch.stack([_SOBEL, _SOBEL.T, _LAP])[:, None].to(self.device)
        self.kernels = k  # (3,1,3,3)
        f = torch.fft.fftfreq(PATCH, device=self.device)
        self.radius = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)  # cycles / native pixel
        self._masks = {}

    def hf_masks(self, width: int) -> dict:
        """Resampling a frame of long side `width` to R px keeps frequencies below 0.5 * R / width."""
        if width not in self._masks:
            self._masks[width] = {r: self.radius > 0.5 * r / width for r in (320, 480)}
        return self._masks[width]

    def _patches(self, gray: torch.Tensor) -> torch.Tensor:
        """Four native-resolution patches along the road band (rows ~55% of height)."""
        h, w = gray.shape[-2:]
        y = int(0.55 * h) - PATCH // 2
        xs = [int(w * f) - PATCH // 2 for f in (0.125, 0.375, 0.625, 0.875)]
        return torch.stack([gray[..., y:y + PATCH, x:x + PATCH] for x in xs])  # (4,C,64,64)

    @torch.no_grad()
    def __call__(self, frame: torch.Tensor) -> dict[str, float]:
        if self.needed <= set(FAST):
            return fast_luminance(frame)
        if self.needed <= LIGHT:
            return self._light(frame)
        return self._full(frame)

    def _light(self, frame: torch.Tensor) -> dict[str, float]:
        x = frame[..., :3].permute(2, 0, 1)[None].float()
        lum = (F.interpolate(x, size=(48, 64), mode="area") * self._w).sum(1)
        m, sd = torch.stack([lum.mean(), lum.std()]).tolist()
        return {"lum_mean": m, "lum_std": sd}

    def _full(self, frame: torch.Tensor) -> dict[str, float]:
        x = frame.permute(2, 0, 1)[None].float()  # (1,4,H,W) B,G,R,IR in 0..255
        lum_native = 0.114 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.299 * x[:, 2:3]
        two = torch.cat([lum_native, x[:, 3:4]], 1)  # (1,2,H,W) luminance, IR

        # thumbnail statistics
        th = F.interpolate(torch.cat([x, lum_native], 1), size=(48, 64), mode="area")[0]  # B,G,R,IR,Y
        lum, ir = th[4], th[3]
        ir_sorted = ir.flatten().sort().values
        n = ir_sorted.numel()
        ir_spread = ir_sorted[int(0.95 * (n - 1))] - ir_sorted[int(0.05 * (n - 1))]
        dark = th[:3].min(0).values.mean()
        g2 = th[[4, 3]][:, None]  # (2,1,48,64)
        resp = F.conv2d(g2, self.kernels)  # (2,3,46,62)
        edge = torch.sqrt(resp[:, 0] ** 2 + resp[:, 1] ** 2).mean((1, 2))
        lapvar = resp[:, 2].var((1, 2))

        # native-resolution spectral detail
        p = self._patches(two[0])  # (4,2,64,64)
        p = p - p.mean((-2, -1), keepdim=True)
        power = torch.fft.fft2(p).abs() ** 2
        total = power.sum((-2, -1)).sum(0) + 1e-6  # (2,)
        hf = {r: power[..., m].sum(-1).sum(0) / total for r, m in self.hf_masks(max(x.shape[-2:])).items()}

        # IR saliency map; one device-to-host copy carries it and every GPU statistic
        ir_small = F.interpolate(x[:, 3:4], size=(96, 128), mode="area")[0, 0]
        sal = (ir_small > ir_small.mean() + 2 * ir_small.std()).float().flatten()
        stats_gpu = torch.stack([
            lum.mean(), lum.std(), ir_spread, dark,
            edge[0], edge[1], lapvar[0], lapvar[1],
            hf[320][0], hf[320][1], hf[480][0], hf[480][1],
        ])
        host = torch.cat([stats_gpu, sal]).cpu().numpy()
        vals = host[:12].tolist()
        sal = host[12:].reshape(96, 128).astype(np.uint8)
        n, _, cc, _ = cv2.connectedComponentsWithStats(sal, connectivity=8)
        areas = cc[1:, cv2.CC_STAT_AREA].astype(np.float32) if n > 1 else np.zeros(1, np.float32)  # 128x96 px
        return dict(zip(FEATURES, vals + [float(n - 1), float(np.median(areas)), float(areas.min())]))


# Proxy for the day / night / adverse split (M3FD ships no scene tags). Thresholds
# were read off contact sheets at feature quantiles (splits/scene_lum_quantiles.jpg,
# splits/scene_haze_quantiles.jpg; result spot-checked in splits/scene_proxy_check.jpg):
# mean luminance < 90 is night or dusk; haze = dark-channel mean - luminance std
# > 115 isolates smoke and fog (the next band down is ordinary overcast).
SCENE_THRESH = {"night_lum": 90.0, "haze": 115.0}


def scene_type(f: dict) -> str:
    if f["lum_mean"] < SCENE_THRESH["night_lum"]:
        return "night"
    if f["dark_channel"] - f["lum_std"] > SCENE_THRESH["haze"]:
        return "adverse"
    return "day"


def load_fusion_frame(i: str) -> np.ndarray:
    from fusion.preprocess import load_frame

    src = "/mnt/12TB_Drive/whisenaj/Datasets/M3FD_Detection"
    return load_frame("fusion", f"{src}/Vis/{i}.png", f"{src}/Ir/{i}.png")


def bench(an: SceneAnalyzer, frames: list[torch.Tensor], warmup=50, reps=3) -> dict:
    for f in frames[:warmup]:
        an(f)
    times = []
    for _ in range(reps):
        for f in frames:
            torch.cuda.synchronize(an.device)
            t = time.perf_counter()
            an(f)
            torch.cuda.synchronize(an.device)
            times.append((time.perf_counter() - t) * 1e3)
    t = np.array(times)
    return {"mean_ms": t.mean(), "p50_ms": np.percentile(t, 50), "p95_ms": np.percentile(t, 95),
            "p99_ms": np.percentile(t, 99), "n": len(t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--needed", nargs="*", default=None, help="restrict --bench to these features")
    args = ap.parse_args()
    an = SceneAnalyzer(args.device, args.needed if args.bench else None)
    manifest = json.loads((REPO / "splits" / "m3fd_grouped_seed42.json").read_text())
    if args.bench:
        ids = manifest["splits"]["test"]
        frames = [torch.from_numpy(load_fusion_frame(i)).to(args.device) for i in ids]
        print(json.dumps({k: round(float(v), 4) for k, v in bench(an, frames).items()}))
    if args.extract:
        import pandas as pd

        rows = []
        for s, ids in manifest["splits"].items():
            for i in ids:
                g = torch.from_numpy(load_fusion_frame(i)).to(args.device)
                f = an(g)
                rows.append({"id": i, "split": s, **f, **fast_luminance(g), "scene": scene_type(f)})
        df = pd.DataFrame(rows)
        out = REPO / "results" / "processed" / "scene_features.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(df.groupby("split").scene.value_counts().unstack(fill_value=0))
        print(df[FEATURES].describe().T[["mean", "25%", "50%", "75%"]].round(3))


if __name__ == "__main__":
    main()
