"""Distilled four-channel thumbnail controller for per-image and temporal policies."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIONS = (400, 480, 544, 576)


def thumbnail(frame: torch.Tensor, height: int = 32, width: int = 48) -> torch.Tensor:
    """Cheap direct sensor-grid sample: HWC uint8 -> NCHW float.

    Direct sampling avoids paying for a resize before deciding whether the detector itself should
    resize. The exact same function builds the cached training thumbnails.
    """
    if frame.device.type != "cpu":
        frame = frame.cpu()
    sh, sw = max(1, frame.shape[0] // height), max(1, frame.shape[1] // width)
    x = frame[::sh, ::sw][:height, :width].permute(2, 0, 1)[None]
    return x.float().div_(255.0)


class DSBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, 3, 2, 1, groups=in_ch, bias=False)
        self.depth_bn = nn.BatchNorm2d(in_ch)
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.point_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.depth_bn(self.depth(x)), inplace=True)
        return F.relu(self.point_bn(self.point(x)), inplace=True)


class StudentNet(nn.Module):
    """Under-3k-parameter utility regressor with an auxiliary small-person head."""

    def __init__(self, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.stem = nn.Conv2d(4, 8, 3, 4, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(8)
        self.block1 = DSBlock(8, 12)
        self.utility = nn.Linear(12, n_actions)
        self.small_person = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)), inplace=True)
        x = self.block1(x).mean((2, 3))
        return self.utility(x), self.small_person(x).squeeze(1)


class DistilledController:
    needs_features = False
    needs_probe = False

    def __init__(self, model, config: dict, temporal: bool = False):
        self.model = model.eval()
        self.actions = tuple(map(int, config["actions"]))
        self.cost_ms = torch.tensor(config["cost_ms"], dtype=torch.float32)
        self.lam = float(config["lambda"])
        self.margin = float(config["uncertainty_margin"])
        self.safety_threshold = float(config["safety_threshold"])
        self.fallback = int(config.get("fallback_resolution", 576))
        self.temporal = temporal
        self.refresh_interval = int(config.get("refresh_interval", 16))
        self.scene_threshold = float(config.get("scene_change_threshold", 0.12))
        self.reset()

    @classmethod
    def from_json(cls, path: str | Path, temporal: bool = False):
        path = Path(path)
        cfg = json.loads(path.read_text())
        model_path = path.parent / cfg["model_file"]
        return cls(torch.jit.load(str(model_path), map_location="cpu"), cfg, temporal)

    def reset(self):
        self.cached_resolution = self.fallback
        self.frames_since_refresh = self.refresh_interval
        self.previous_scene = None

    @staticmethod
    def _scene_signature(frame: torch.Tensor) -> torch.Tensor:
        if frame.device.type != "cpu":
            frame = frame.cpu()
        gray = frame[..., :3].float().mean(2)[None, None].div_(255.0)
        return F.interpolate(gray, (16, 24), mode="bilinear", align_corners=False)[0, 0]

    def _infer(self, frame: torch.Tensor) -> int:
        with torch.inference_mode():
            utility, risk = self.model(thumbnail(frame))
        score = utility[0] - self.lam * self.cost_ms
        top = torch.topk(score, min(2, len(self.actions)))
        uncertain = len(self.actions) > 1 and float(top.values[0] - top.values[1]) < self.margin
        unsafe = float(torch.sigmoid(risk[0])) >= self.safety_threshold
        return self.fallback if uncertain or unsafe else self.actions[int(top.indices[0])]

    def choose(self, frame_id: str, features=None, frame=None) -> int:
        if frame is None:
            raise ValueError("distilled controller requires the current frame")
        if not self.temporal:
            return self._infer(frame)
        signature = self._scene_signature(frame)
        changed = (self.previous_scene is None or
                   float((signature - self.previous_scene).abs().mean()) >= self.scene_threshold)
        if changed or self.frames_since_refresh >= self.refresh_interval:
            self.cached_resolution = self._infer(frame)
            self.frames_since_refresh = 0
        self.previous_scene = signature
        self.frames_since_refresh += 1
        return self.cached_resolution

    def update(self, frame_id: str, detections) -> None:
        pass
