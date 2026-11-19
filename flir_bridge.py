"""Read calibrated JPEG snapshots from the existing FLIR C++ server.

This keeps RTSP ownership, calibration, and camera controls in the C++ service.
Timestamp pairing in live_dashboard measures HTTP arrival skew, not sensor sync.
"""
import time
import urllib.request

import cv2
import numpy as np


class SnapshotCapture:
    """Small VideoCapture-compatible adapter with bounded HTTP reads."""

    def __init__(self, source, fps=12):
        self.source = source
        self.period = 1 / fps
        self.next_read = 0.0
        self.closed = False

    def isOpened(self):
        return not self.closed

    def set(self, *_args):
        return False

    def get(self, *_args):
        return 0.0

    def read(self):
        time.sleep(max(0, self.next_read - time.monotonic()))
        self.next_read = time.monotonic() + self.period
        try:
            req = urllib.request.Request(self.source, headers={"Cache-Control": "no-cache"})
            with urllib.request.urlopen(req, timeout=2) as response:
                data = response.read(16 * 1024 * 1024)
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            return frame is not None, frame
        except (OSError, ValueError, cv2.error):
            return False, None

    def release(self):
        self.closed = True


def capture_factory(source):
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        return SnapshotCapture(source)
    return cv2.VideoCapture(source)
