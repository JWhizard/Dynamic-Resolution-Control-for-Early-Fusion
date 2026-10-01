"""4-channel model on (B,G,R,IR=0) must reproduce the COCO model on (R,G,B)."""
import sys
from copy import deepcopy
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion.stem import inflate_stem

W = "/home/whisenaj/anaconda3/envs/YOLO_Workspace/YOLO_Workspace/YOLO_Workspace/yolo26n.pt"


def test_stem_equivalence():
    m3 = YOLO(W).model.float().eval()
    m4 = DetectionModel(deepcopy(m3.yaml), ch=4, nc=m3.yaml["nc"], verbose=False)
    m4.load(m3, verbose=False)
    assert inflate_stem(m4, m3)
    m4.eval()
    rgb = torch.rand(1, 3, 256, 320)
    bgr_ir0 = torch.cat([rgb.flip(1), torch.zeros(1, 1, 256, 320)], 1)
    with torch.no_grad():
        a, b = m3(rgb), m4(bgr_ir0)
    a, b = (a[0] if isinstance(a, (tuple, list)) else a), (b[0] if isinstance(b, (tuple, list)) else b)
    err = (a - b).abs().max().item()
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-3), err
    with torch.no_grad():  # negative control: un-permuted channel order must differ
        c = m4(torch.cat([rgb, torch.zeros(1, 1, 256, 320)], 1))
    c = c[0] if isinstance(c, (tuple, list)) else c
    assert (a - c).abs().max().item() > 100 * err
    assert not inflate_stem(m4, m4), "must no-op on a 4-channel source (resume)"
    print("stem equivalence max abs err", err)


if __name__ == "__main__":
    test_stem_equivalence()
