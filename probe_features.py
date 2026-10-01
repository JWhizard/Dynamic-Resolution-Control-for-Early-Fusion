"""Probe features: statistics of the detector's own 320-px detections.

Global scene statistics barely predict how much a frame gains from higher
resolution (|Spearman rho| <= 0.16 on val), because the gain is decided by small
objects that a thumbnail cannot see. The 320-px detections can: counts, box
sizes and confidences of what the detector already found (or is unsure about).
"""

from __future__ import annotations

import numpy as np

PROBE_FEATURES = [
    "n_conf25", "n_conf10", "n_small_conf10", "n_people_conf10", "n_small_people_conf10",
    "frac_small_conf10", "median_area_conf25", "min_area_conf25", "mean_conf_top10",
    "n_uncertain", "max_conf",
]
PEOPLE = 4
SMALL = 32 ** 2  # native px^2


def probe_features(d: np.ndarray) -> dict[str, float]:
    """d: (n,6) native-pixel detections x1,y1,x2,y2,conf,cls from the 320-px pass."""
    area = (d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]) if len(d) else np.zeros(0)
    c = d[:, 4] if len(d) else np.zeros(0)
    cls = d[:, 5] if len(d) else np.zeros(0)
    m25, m10 = c >= 0.25, c >= 0.10
    small = area < SMALL
    top = np.sort(c)[::-1][:10]
    return {
        "n_conf25": float(m25.sum()),
        "n_conf10": float(m10.sum()),
        "n_small_conf10": float((m10 & small).sum()),
        "n_people_conf10": float((m10 & (cls == PEOPLE)).sum()),
        "n_small_people_conf10": float((m10 & small & (cls == PEOPLE)).sum()),
        "frac_small_conf10": float((m10 & small).sum() / max(m10.sum(), 1)),
        "median_area_conf25": float(np.median(area[m25])) if m25.any() else 0.0,
        "min_area_conf25": float(area[m25].min()) if m25.any() else 0.0,
        "mean_conf_top10": float(top.mean()) if len(top) else 0.0,
        "n_uncertain": float(((c >= 0.10) & (c < 0.40)).sum()),
        "max_conf": float(c.max()) if len(c) else 0.0,
    }
