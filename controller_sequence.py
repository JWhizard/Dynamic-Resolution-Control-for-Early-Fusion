"""Amortised sequence controller using detections that the detector already produced.

The controller makes no probe pass.  It summarizes the preceding frame's native-coordinate
detections, chooses a resolution once every ``interval`` frames, and holds that choice for the
remainder of the window.  Sequence changes reset to a conservative fixed resolution.

The JSON artifact is fitted by ``fit_sequence_controller.py`` and can be benchmarked with::

    python bench_latency_energy.py --run ms_fusion_seed42 \
      --policy sequence:results/processed/sequence/controller.json --cpu-threads 4
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probe_features import PROBE_FEATURES, probe_features

REPO = Path(__file__).resolve().parent
MANIFEST = REPO / "splits" / "m3fd_grouped_seed42.json"


def choose_tree(tree: dict, features: dict[str, float]) -> int:
    """Evaluate the small exported CART without an sklearn runtime dependency."""
    node = tree
    while "leaf" not in node:
        node = node["left"] if features[node["feature"]] <= node["threshold"] else node["right"]
    return int(node["leaf"])


class SequenceController:
    """Stateful resolution policy driven by the previous frame's detections.

    A decision is made for the first frame after every completed ``interval``-frame window.
    The first frame in a new sequence uses ``default_res`` because no past observation exists.
    """

    needs_features = False
    needs_probe = False

    def __init__(self, tree: dict, interval: int = 16, default_res: int = 576,
                 actions=(480, 544, 576, 640), groups: dict[str, str] | None = None,
                 safety_small_people: bool = True):
        if interval < 1:
            raise ValueError("interval must be >= 1")
        self.tree = tree
        self.interval = int(interval)
        self.default_res = int(default_res)
        self.actions = tuple(map(int, actions))
        if self.default_res not in self.actions:
            raise ValueError("default_res must be one of actions")
        self.groups = groups or {}
        self.safety_small_people = bool(safety_small_people)
        self.reset()

    @classmethod
    def from_json(cls, path: str | Path) -> "SequenceController":
        cfg = json.loads(Path(path).read_text())
        groups = None
        if cfg.get("reset_on_manifest_group", True) and MANIFEST.exists():
            groups = json.loads(MANIFEST.read_text())["groups"]
        return cls(cfg["tree"], cfg.get("interval", 16), cfg.get("default_res", 576),
                   cfg.get("actions", [480, 544, 576, 640]), groups,
                   cfg.get("safety_small_people", True))

    def reset(self) -> None:
        self.current_res = self.default_res
        self.current_group = None
        self.frames_in_window = 0
        self.previous_features: dict[str, float] | None = None

    def _start_group(self, group: str) -> None:
        self.current_group = group
        self.current_res = self.default_res
        self.frames_in_window = 0
        self.previous_features = None

    def choose(self, frame_id: str, features=None, frame=None) -> int:
        group = self.groups.get(frame_id, self.current_group if self.current_group is not None else "stream")
        if group != self.current_group:
            self._start_group(group)
        if self.frames_in_window >= self.interval and self.previous_features is not None:
            self.current_res = choose_tree(self.tree, self.previous_features)
            if self.current_res not in self.actions:
                raise ValueError(f"tree selected unsupported resolution {self.current_res}")
            self.frames_in_window = 0
        self.frames_in_window += 1
        return self.current_res

    def update(self, frame_id: str, detections) -> None:
        if hasattr(detections, "detach"):
            detections = detections.detach().cpu().numpy()
        d = np.asarray(detections, dtype=float).reshape(-1, 6)
        self.previous_features = probe_features(d)
        # A small pedestrian is the safety-critical failure mode. Escalate the *next*
        # frame immediately rather than waiting for the next scheduled decision.
        if self.safety_small_people and self.previous_features["n_small_people_conf10"] > 0:
            self.current_res = max(self.actions)
            self.frames_in_window = 0


def feature_vector(detections) -> np.ndarray:
    """Stable feature ordering shared by fitting and runtime."""
    f = probe_features(np.asarray(detections, dtype=float).reshape(-1, 6))
    return np.asarray([f[n] for n in PROBE_FEATURES], dtype=float)
