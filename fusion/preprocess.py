"""The single GPU inference path shared by evaluation (predict_cache.py) and the
latency/energy harness (bench_latency_energy.py), so accuracy and cost numbers
come from identical code.

Frame -> model input:
    native uint8 HWC (768x1024xC) --H2D--> resize (bilinear, antialiased) to
    long side = R --> letterbox pad to a stride-32 multiple (640x480, 480x384,
    320x256 for 4:3 input) --> float [0,1] NCHW.
Model output (YOLO26 end-to-end head, NMS-free): (1, 300, 6) = x1,y1,x2,y2,conf,cls
in letterboxed pixels --> confidence filter --> native-resolution boxes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

STRIDE = 32
PAD_VALUE = 114 / 255
RESOLUTIONS = (320, 480, 640)


def load_frame(view: str, vis_path: str, ir_path: str) -> np.ndarray:
    """Native HWC uint8 frame in the channel order each model was trained on.

    fusion: B,G,R,IR (TIFF order; Ultralytics does not flip 4-channel input)
    rgb:    R,G,B    (Ultralytics flips 3-channel BGR -> RGB)
    ir:     IR,IR,IR
    """
    if view == "fusion":
        vis = cv2.imread(vis_path, cv2.IMREAD_COLOR)
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        return np.ascontiguousarray(np.dstack([vis, ir]))
    if view == "rgb":
        return np.ascontiguousarray(cv2.imread(vis_path, cv2.IMREAD_COLOR)[..., ::-1])
    if view == "ir":
        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        return np.ascontiguousarray(np.repeat(ir[..., None], 3, axis=2))
    raise ValueError(view)


@dataclass
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int
    shape: tuple[int, int]  # model input (h, w)


def letterbox_params(h: int, w: int, res: int) -> Letterbox:
    s = res / max(h, w)
    nh, nw = round(h * s), round(w * s)
    ph, pw = math.ceil(nh / STRIDE) * STRIDE, math.ceil(nw / STRIDE) * STRIDE
    return Letterbox(s, (pw - nw) // 2, (ph - nh) // 2, (ph, pw))


def to_input(frame_gpu: torch.Tensor, res: int) -> tuple[torch.Tensor, Letterbox]:
    """frame_gpu: uint8 HWC on device. Returns (1,C,h,w) float input and its letterbox."""
    h, w = frame_gpu.shape[:2]
    lb = letterbox_params(h, w, res)
    x = frame_gpu.permute(2, 0, 1).unsqueeze(0).float().div_(255)
    nh, nw = round(h * lb.scale), round(w * lb.scale)
    if (nh, nw) != (h, w):
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    ph, pw = lb.shape
    if (ph, pw) != (nh, nw):
        x = F.pad(x, (lb.pad_x, pw - nw - lb.pad_x, lb.pad_y, ph - nh - lb.pad_y), value=PAD_VALUE)
    return x, lb


def postprocess(out, lb: Letterbox, h: int, w: int, conf: float = 0.001) -> torch.Tensor:
    """(1,300,6) end-to-end output -> (n,6) x1,y1,x2,y2,conf,cls in native pixels."""
    y = out[0] if isinstance(out, (tuple, list)) else out
    d = y[0]
    d = d[d[:, 4] > conf].clone()
    d[:, [0, 2]] = ((d[:, [0, 2]] - lb.pad_x) / lb.scale).clamp_(0, w)
    d[:, [1, 3]] = ((d[:, [1, 3]] - lb.pad_y) / lb.scale).clamp_(0, h)
    return d


def load_model(weights: str, device: str | torch.device = "cuda", half: bool = False) -> torch.nn.Module:
    from ultralytics import YOLO

    m = YOLO(weights).model.float().fuse(verbose=False).to(device).eval()
    return m.half() if half else m


@torch.no_grad()
def detect(model, frame_gpu: torch.Tensor, res: int, conf: float = 0.001) -> torch.Tensor:
    x, lb = to_input(frame_gpu, res)
    if next(model.parameters()).dtype == torch.float16:
        x = x.half()
    return postprocess(model(x), lb, *frame_gpu.shape[:2], conf=conf).float()
