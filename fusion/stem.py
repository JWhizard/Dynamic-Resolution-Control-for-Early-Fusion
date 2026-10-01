"""Inflate a COCO-pretrained 3-channel YOLO stem to 4 input channels (B,G,R,IR).

Adapted from `adapt_first_conv_to_four_channels` (YOLO_Workspace early-fusion
trainer, init_mode="rgb_copy_ir_mean"). Two differences:
  * Ultralytics' 4-channel data path does not flip BGR->RGB, so the network sees
    the TIFF channels as stored (B,G,R,IR). The pretrained filters were learned
    on R,G,B and are permuted accordingly.
  * `DetectionModel.load` in Ultralytics 8.4.x already copies input channels 0-2
    verbatim (wrong order) and leaves channel 3 randomly initialised; this
    function overwrites all four.
"""

from __future__ import annotations

import torch

FIRST_CONV = "model.0.conv.weight"


@torch.no_grad()
def inflate_stem(model4: torch.nn.Module, source: torch.nn.Module | dict) -> bool:
    """Write permuted RGB filters and an RGB-mean IR filter into `model4`'s stem.

    Returns False (and does nothing) when the source stem is not 3-channel,
    e.g. when resuming from a 4-channel checkpoint.
    """
    src = source["model"] if isinstance(source, dict) else source
    w_rgb = src.float().state_dict()[FIRST_CONV]
    if w_rgb.shape[1] != 3:
        return False
    conv = model4.model[0].conv
    assert conv.weight.shape[1] == 4 and conv.weight.shape[2:] == w_rgb.shape[2:], conv.weight.shape
    w = w_rgb.to(conv.weight)
    conv.weight[:, 0] = w[:, 2]  # B
    conv.weight[:, 1] = w[:, 1]  # G
    conv.weight[:, 2] = w[:, 0]  # R
    conv.weight[:, 3] = w.mean(dim=1)  # IR
    return True
