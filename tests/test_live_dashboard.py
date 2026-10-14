import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live_dashboard import (
    CaptureWorker,
    FramePacket,
    PairedFrameSource,
    create_app,
    fuse_frames,
    parse_source,
)


def test_parse_source_preserves_paths_and_urls():
    assert parse_source("0") == 0
    assert parse_source("12") == 12
    assert parse_source("/dev/video0") == "/dev/video0"
    assert parse_source("rtsp://camera/stream") == "rtsp://camera/stream"


def test_fuse_frames_uses_bgr_ir_order():
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 30
    thermal = np.full((3, 4), 40, dtype=np.uint8)
    fused, mono = fuse_frames(rgb, thermal)
    assert fused.shape == (3, 4, 4)
    assert fused[0, 0].tolist() == [10, 20, 30, 40]
    assert np.array_equal(mono, thermal)


def test_fuse_frames_converts_three_channel_thermal():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    thermal = np.full((2, 3, 3), 80, dtype=np.uint8)
    fused, mono = fuse_frames(rgb, thermal)
    assert mono.shape == (2, 3)
    assert np.all(fused[..., 3] == 80)


@pytest.mark.parametrize(
    "rgb,thermal,error",
    [
        (np.zeros((2, 3, 3), np.uint8), np.zeros((3, 3), np.uint8), "equal dimensions"),
        (np.zeros((2, 3, 3), np.float32), np.zeros((2, 3), np.uint8), "8-bit"),
        (np.zeros((2, 3, 2), np.uint8), np.zeros((2, 3), np.uint8), "RGB frame"),
    ],
)
def test_fuse_frames_rejects_invalid_camera_data(rgb, thermal, error):
    with pytest.raises(ValueError, match=error):
        fuse_frames(rgb, thermal)


def test_pairer_drops_older_packet_outside_skew_window():
    rgb = CaptureWorker("RGB", 0)
    thermal = CaptureWorker("thermal", 1)
    pairer = PairedFrameSource(rgb, thermal, max_skew_ms=20)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb._latest = FramePacket(frame, 1.00, 1)
    thermal._latest = FramePacket(frame, 1.10, 1)
    assert pairer.next_pair(timeout=0.01) is None
    assert pairer.dropped_rgb == 1
    rgb._latest = FramePacket(frame, 1.11, 2)
    pair = pairer.next_pair(timeout=0.01)
    assert pair is not None
    assert pairer.last_skew_ms == pytest.approx(10.0)


class FakeRuntime:
    def __init__(self, directory: Path):
        self.directory = directory
        self.controls = []
        self.has_frame = False

    def status(self):
        return {"running": True, "policy": "sequence", "confidence": 0.25}

    def set_control(self, payload):
        if payload.get("policy") == "invalid":
            raise ValueError("bad policy")
        self.controls.append(payload)

    def mjpeg(self):
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\nabc\r\n"

    def snapshot(self):
        if not self.has_frame:
            raise RuntimeError("no processed frame is available")
        image, metadata = self.directory / "frame.jpg", self.directory / "frame.json"
        image.write_bytes(b"jpeg")
        metadata.write_text(json.dumps({"frame": 1}))
        return image, metadata


def test_dashboard_routes_and_controls(tmp_path):
    runtime = FakeRuntime(tmp_path)
    client = create_app(runtime).test_client()
    assert client.get("/").status_code == 200
    assert b"Early Fusion" in client.get("/").data
    assert client.get("/api/status").get_json()["policy"] == "sequence"
    assert client.post("/api/control", json={"confidence": 0.4}).status_code == 200
    assert runtime.controls == [{"confidence": 0.4}]
    assert client.post("/api/control", data="bad").status_code == 400
    assert client.post("/api/control", json={"policy": "invalid"}).status_code == 400
    assert client.post("/api/snapshot").status_code == 409
    runtime.has_frame = True
    response = client.post("/api/snapshot")
    assert response.status_code == 200
    assert response.get_json()["image"].endswith("frame.jpg")


def test_mjpeg_endpoint_has_stream_content_type(tmp_path):
    client = create_app(FakeRuntime(tmp_path)).test_client()
    response = client.get("/stream.mjpg")
    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"
    assert b"Content-Type: image/jpeg" in response.data
