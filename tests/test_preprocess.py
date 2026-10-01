"""Shared GPU path vs Ultralytics predict: shapes, box format, channel order."""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion.preprocess import detect, letterbox_params, load_frame, load_model

RUN = Path(__file__).resolve().parents[1] / "runs" / "b2_fusion_640_seed42" / "weights" / "last.pt"
SRC = "/mnt/12TB_Drive/whisenaj/Datasets/M3FD_Detection"
TIFF = "/mnt/12TB_Drive/whisenaj/Datasets/M3FD_4ch/fusion/images/{}/{}.tiff"


def test_shapes():
    assert [letterbox_params(768, 1024, r).shape for r in (320, 480, 640)] == [(256, 320), (384, 480), (480, 640)]


def test_matches_ultralytics(ids=("00909", "00010", "01500", "03000"), dev="cuda:0"):
    import json
    split = json.load(open(Path(__file__).resolve().parents[1] / "splits" / "m3fd_grouped_seed42.json"))["splits"]
    where = {i: s for s, v in split.items() for i in v}
    model, ref = load_model(str(RUN), dev), YOLO(str(RUN))
    for i in ids:
        frame = torch.from_numpy(load_frame("fusion", f"{SRC}/Vis/{i}.png", f"{SRC}/Ir/{i}.png")).to(dev)
        ours = detect(model, frame, 640, conf=0.25).cpu()
        r = ref.predict(TIFF.format(where[i], i), imgsz=640, conf=0.25, device=dev, verbose=False)[0].boxes
        theirs = torch.cat([r.xyxy, r.conf[:, None], r.cls[:, None]], 1).cpu()
        if len(theirs) == 0:
            assert len(ours) <= 2, (i, len(ours))
            continue
        iou = box_iou(theirs[:, :4], ours[:, :4]).max(1).values
        print(i, "ultralytics", len(theirs), "ours", len(ours), "median best IoU", round(float(iou.median()), 3))
        assert abs(len(ours) - len(theirs)) <= max(2, 0.15 * len(theirs))
        assert float(iou.median()) > 0.9


if __name__ == "__main__":
    test_shapes()
    test_matches_ultralytics()
    print("preprocess ok")
