"""Ultralytics DetectionTrainer for 4-channel early fusion and discrete multi-scale training."""

from __future__ import annotations

import math
import random

import numpy as np
import torch.nn.functional as F
from ultralytics.data.augment import RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model

from .stem import inflate_stem


class ColorOnlyHSV:
    """Apply RandomHSV to the B,G,R channels of a 4-channel image.

    Stock RandomHSV returns early for any non-3-channel image, which would give
    the RGB-only baseline colour augmentation that the fusion model never sees.
    """

    def __init__(self, hsv: RandomHSV):
        self.hsv = hsv

    def __call__(self, labels):
        img = labels["img"]
        if img.shape[-1] != 4:
            return self.hsv(labels)
        bgr = np.ascontiguousarray(img[..., :3])
        self.hsv({"img": bgr})
        img[..., :3] = bgr
        return labels


class FusionDataset(YOLODataset):
    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        for k, t in enumerate(transforms.transforms):
            if isinstance(t, RandomHSV):
                transforms.transforms[k] = ColorOnlyHSV(t)
        return transforms


class FusionTrainer(DetectionTrainer):
    """Adds (1) 4-channel stem inflation, (2) colour augmentation on 4-channel
    inputs, and (3) per-batch resolution drawn from `ms_sizes` (None = off).

    Ultralytics' own `multi_scale` samples a continuous range; the controller
    only ever serves {320, 480, 640}, so training draws from that exact set.
    """

    ms_sizes: tuple[int, ...] | None = None
    warm_start: str | None = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        source = weights
        if weights is not None and len(getattr(model.model[-1], "cv3", ())) == 4:
            # A P2 checkpoint may have a widened classification branch that is not
            # represented in Ultralytics' architecture YAML. The stock loader first
            # rebuilds the narrow YAML model and silently drops those tensors. Match
            # the saved branch width, then reload so a P2 fine-tune preserves its head.
            from .p2warm import align_cls_width

            width = align_cls_width(model, weights)
            if width is not None:
                model.load(weights, verbose=False)
                print(f"{colorstr('P2 resume:')} restored widened c3={width} classification branches")
        if weights is None and self.warm_start:
            # Architecture came from a YAML, so Ultralytics built it randomly. Transfer
            # what a trained P3-P5 checkpoint can supply (see fusion/p2warm.py).
            import torch

            from .p2warm import warm_start_p2

            ckpt = torch.load(self.warm_start, weights_only=False)
            source = ckpt["model"].float()
            n, total = warm_start_p2(model, source)
            print(f"{colorstr('warm start:')} {n}/{total} tensors from {self.warm_start}")
        if source is not None and self.data["channels"] == 4 and inflate_stem(model, source):
            print(f"{colorstr('stem:')} inflated COCO RGB stem to B,G,R,IR")
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        cfg = self.args
        return FusionDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=cfg,
            rect=cfg.rect or mode == "val",
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        if self.ms_sizes:
            imgs = batch["img"]
            sf = random.choice(self.ms_sizes) / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                batch["img"] = F.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
        return batch


def make_trainer(ms_sizes=None, warm_start=None):
    return type("FusionTrainerMS" if ms_sizes else "FusionTrainer", (FusionTrainer,),
                {"ms_sizes": ms_sizes, "warm_start": warm_start})
