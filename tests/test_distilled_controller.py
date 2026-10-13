import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller_distilled import DistilledController, StudentNet, thumbnail


class FakeModel(torch.nn.Module):
    def __init__(self, utility, risk=-10.0):
        super().__init__()
        self.register_buffer("u", torch.tensor([utility], dtype=torch.float32))
        self.risk = risk

    def forward(self, x):
        return self.u.expand(len(x), -1), torch.full((len(x),), self.risk)


def config(**updates):
    x = {"actions": [400, 480, 544, 576], "cost_ms": [16.46, 19.31, 21.80, 23.54],
         "lambda": 0.0, "uncertainty_margin": 0.01, "safety_threshold": 0.8,
         "fallback_resolution": 576, "refresh_interval": 2, "scene_change_threshold": 1.0}
    x.update(updates)
    return x


def frame(v=0):
    return torch.full((64, 96, 4), v, dtype=torch.uint8)


def test_student_size_and_thumbnail_shape():
    assert sum(p.numel() for p in StudentNet().parameters()) < 3000
    assert thumbnail(frame()).shape == (1, 4, 32, 48)


def test_selects_best_action():
    p = DistilledController(FakeModel([0.0, 0.1, 0.4, 0.2]), config())
    assert p.choose("a", frame=frame()) == 544


def test_uncertainty_and_safety_fall_back():
    uncertain = DistilledController(FakeModel([0.0, 0.1, 0.2, 0.195]), config())
    unsafe = DistilledController(FakeModel([0.0, 0.1, 0.4, 0.2], risk=10.0), config())
    assert uncertain.choose("a", frame=frame()) == 576
    assert unsafe.choose("a", frame=frame()) == 576


def test_temporal_mode_reuses_decision_and_reset():
    p = DistilledController(FakeModel([0.0, 0.1, 0.4, 0.2]), config(), temporal=True)
    assert p.choose("a", frame=frame()) == 544
    p.model = FakeModel([0.0, 0.1, 0.2, 0.5])
    assert p.choose("b", frame=frame()) == 544
    assert p.choose("c", frame=frame()) == 576
    p.reset()
    assert p.choose("d", frame=frame()) == 576
