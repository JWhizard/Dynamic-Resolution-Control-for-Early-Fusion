import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller_heuristic import make_policy
from controller_sequence import SequenceController


TREE = {
    "feature": "n_conf25", "threshold": 0.5,
    "left": {"leaf": 480}, "right": {"leaf": 640},
}


def detections(n=0, small_person=False):
    rows = [[10, 10, 100, 100, 0.9, 1] for _ in range(n)]
    if small_person:
        rows.append([0, 0, 10, 10, 0.5, 4])
    return np.asarray(rows, dtype=float).reshape(-1, 6)


def test_holds_decision_for_interval_and_uses_previous_detection():
    p = SequenceController(TREE, interval=2, default_res=576, groups={"a": "g", "b": "g", "c": "g"},
                           safety_small_people=False)
    assert p.choose("a", None) == 576
    p.update("a", detections(2))
    assert p.choose("b", None) == 576
    p.update("b", detections(2))
    assert p.choose("c", None) == 640


def test_group_change_resets_to_safe_default():
    p = SequenceController(TREE, interval=1, default_res=576, groups={"a": "g1", "b": "g1", "c": "g2"},
                           safety_small_people=False)
    assert p.choose("a", None) == 576
    p.update("a", detections())
    assert p.choose("b", None) == 480
    assert p.choose("c", None) == 576


def test_small_person_escalates_next_frame():
    p = SequenceController(TREE, interval=16, default_res=576, groups={"a": "g", "b": "g"})
    assert p.choose("a", None) == 576
    p.update("a", detections(small_person=True))
    assert p.choose("b", None) == 640


def test_policy_factory_loads_sequence_json(tmp_path):
    path = tmp_path / "controller.json"
    path.write_text(json.dumps({"tree": TREE, "interval": 8, "default_res": 576,
                                "actions": [480, 576, 640], "reset_on_manifest_group": False}))
    p = make_policy(f"sequence:{path}")
    assert isinstance(p, SequenceController)
    assert p.interval == 8
