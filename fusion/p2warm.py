"""Warm-start a P2 (stride 4/8/16/32) model from a trained P3-P5 checkpoint.

The P2 head is the stock head with one level inserted, so most of the network is
structurally identical and can be transferred by name once the indices are shifted:

    base (yolo26.yaml)              P2 (yolo26-p2.yaml)
    0-10   backbone            ->   0-10   identical
    11-13  top-down to P4      ->   11-13  identical
    14-16  top-down to P3      ->   14-16  identical
    --                              17-19  NEW: top-down to P2/4
    --                              20-22  NEW: bottom-up P2 -> P3
    17-19  bottom-up to P4     ->   23-25  structurally identical
    20-22  bottom-up to P5     ->   26-28  structurally identical
    23     Detect(P3,P4,P5)    ->   29     Detect(P2,P3,P4,P5), levels shifted by +1

Only two C3k2 blocks, one Conv and the new stride-4 Detect level start from scratch.

The Detect head's internal width is derived from its finest input (`c3 = max(ch[0], nc)`),
and the P2 head's finest input is narrower than the base head's, so the classification
branch does not transfer on shape. `DetectionModel.load` drops those automatically via
`intersect_dicts`; the box branch and everything upstream still transfer.
"""

from __future__ import annotations

import torch

# Detect submodules whose first index is the pyramid level.
_LEVELWISE = ("cv2", "cv3", "one2one_cv2", "one2one_cv3")

BASE_DETECT = 23
P2_DETECT = 29


def remap_key(key: str) -> str | None:
    """Rename one base-model state-dict key to its P2 equivalent, or None to drop it."""
    parts = key.split(".")
    if parts[0] != "model" or not parts[1].isdigit():
        return key
    i, rest = int(parts[1]), parts[2:]

    if i <= 16:
        j = i
    elif 17 <= i <= 22:
        j = i + 6
    elif i == BASE_DETECT:
        j = P2_DETECT
        # shift the pyramid level by one: base level k feeds P2 level k+1
        if rest and rest[0] in _LEVELWISE and len(rest) > 1 and rest[1].isdigit():
            rest = [rest[0], str(int(rest[1]) + 1), *rest[2:]]
    else:
        return None
    return ".".join(["model", str(j), *rest])


def remap_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        nk = remap_key(k)
        if nk is not None:
            out[nk] = v
    return out


def align_cls_width(model, source) -> int | None:
    """Rebuild the target's classification branch at the source's mid-width.

    `Detect` derives that width from its *finest* input (`c3 = max(ch[0], nc)`), and the
    P2 head's finest level is narrower than the base head's (32 vs 64). Left alone, the
    classification weights cannot transfer on shape and the classifier retrains from
    scratch -- which costs accuracy uniformly across every object size, including sizes
    the new P2 level does not touch. Widening first makes the transfer possible; it is
    near-free at 400 px (1.30x vs 1.27x of the base model).

    Returns the new width, or None if no change was needed.
    """
    import torch.nn as nn
    from ultralytics.nn.modules.conv import Conv, DWConv

    tgt, src = model.model[-1], source.model[-1]
    c3_src = src.cv3[0][0][1].conv.out_channels
    c3_tgt = tgt.cv3[0][0][1].conv.out_channels
    if c3_src == c3_tgt:
        return None

    ch = [b[0][0].conv.in_channels for b in tgt.cv3]
    make = lambda: nn.ModuleList(
        nn.Sequential(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3_src, 1)),
            nn.Sequential(DWConv(c3_src, c3_src, 3), Conv(c3_src, c3_src, 1)),
            nn.Conv2d(c3_src, tgt.nc, 1),
        )
        for x in ch
    )
    tgt.cv3, tgt.one2one_cv3 = make(), make()
    return c3_src


def warm_start_p2(model, source) -> tuple[int, int]:
    """Load a base checkpoint into a P2 `DetectionModel`. Returns (transferred, total)."""
    src = source["model"] if isinstance(source, dict) else source
    w = align_cls_width(model, src)
    if w is not None:
        print(f"p2warm: widened classification branch to c3={w} so the source head transfers")
    sd = src.float().state_dict() if hasattr(src, "state_dict") else src
    remapped = remap_state_dict(sd)

    target = model.state_dict()
    keep = {k: v for k, v in remapped.items()
            if k in target and target[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return len(keep), len(target)
