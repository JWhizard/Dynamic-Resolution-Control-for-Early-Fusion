"""Local live dashboard for the RGB/LWIR dynamic-resolution detector.

Two OpenCV sources are captured independently, paired by their host timestamps,
stacked in the model's B,G,R,IR order, and processed by either the sequence
controller or a fixed-576 baseline.  Capture and inference are deliberately
decoupled: only the latest frame from each source is retained, so a slow model or
browser drops work instead of growing an increasingly stale queue.

Examples:
    python live_dashboard.py --rgb-source 0 --thermal-source 1
    python live_dashboard.py --rgb-source rgb.mp4 --thermal-source ir.mp4 --loop
    python live_dashboard.py --rgb-source rtsp://rgb --thermal-source rtsp://ir
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request

from controller_heuristic import Fixed
from controller_sequence import SequenceController
from fusion.preprocess import detect, load_model

REPO = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = REPO / "runs" / "ms_fusion_seed42" / "weights" / "best.pt"
DEFAULT_CONTROLLER = REPO / "results" / "processed" / "sequence" / "controller.json"
CLASS_NAMES = {0: "Bus", 1: "Car", 2: "Lamp", 3: "Motorcycle", 4: "People", 5: "Truck"}
COLORS = {
    0: (255, 166, 43),
    1: (54, 211, 153),
    2: (250, 204, 21),
    3: (244, 114, 182),
    4: (96, 165, 250),
    5: (192, 132, 252),
}
VIEWS = {"rgb", "thermal", "split"}
POLICIES = {"sequence", "fixed576"}


def parse_source(value: str) -> int | str:
    """Turn a non-negative decimal source into an OpenCV camera index."""
    return int(value) if value.isdecimal() else value


def fuse_frames(rgb: np.ndarray, thermal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate registered inputs and return (BGRIR, mono thermal).

    The fusion detector was trained on uint8 B,G,R,IR images.  Resizing one
    stream here would hide a broken registration, so mismatched dimensions are
    rejected rather than silently changed.
    """
    if rgb is None or thermal is None:
        raise ValueError("both RGB and thermal frames are required")
    if rgb.dtype != np.uint8 or thermal.dtype != np.uint8:
        raise ValueError("camera frames must be 8-bit (uint8)")
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    elif rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB frame must have 1 or 3 channels, got {rgb.shape}")
    if thermal.ndim == 3:
        if thermal.shape[2] not in (3, 4):
            raise ValueError(f"thermal frame must have 1, 3, or 4 channels, got {thermal.shape}")
        conversion = cv2.COLOR_BGRA2GRAY if thermal.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        thermal = cv2.cvtColor(thermal, conversion)
    elif thermal.ndim != 2:
        raise ValueError(f"thermal frame must have 1, 3, or 4 channels, got {thermal.shape}")
    if rgb.shape[:2] != thermal.shape[:2]:
        raise ValueError(
            f"registered streams must have equal dimensions: RGB {rgb.shape[:2]}, thermal {thermal.shape[:2]}"
        )
    thermal = np.ascontiguousarray(thermal)
    return np.ascontiguousarray(np.dstack((rgb, thermal))), thermal


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    timestamp: float
    sequence: int


class CaptureWorker:
    """Continuously retain the newest frame from one OpenCV source."""

    def __init__(
        self,
        name: str,
        source: int | str,
        loop_files: bool = False,
        capture_factory: Callable = cv2.VideoCapture,
    ):
        self.name = name
        self.source = source
        self.loop_files = loop_files
        self.capture_factory = capture_factory
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: FramePacket | None = None
        self._sequence = 0
        self.connected = False
        self.error: str | None = None
        self.reconnects = 0
        self.frames_read = 0

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, str) and Path(self.source).is_file()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"capture-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def latest(self) -> FramePacket | None:
        with self._lock:
            return self._latest

    def status(self) -> dict:
        packet = self.latest()
        return {
            "source": str(self.source),
            "connected": self.connected,
            "error": self.error,
            "frames_read": self.frames_read,
            "reconnects": self.reconnects,
            "shape": list(packet.frame.shape) if packet else None,
        }

    def _publish(self, frame: np.ndarray) -> None:
        with self._lock:
            self._sequence += 1
            self.frames_read += 1
            self._latest = FramePacket(frame, time.monotonic(), self._sequence)

    def _run(self) -> None:
        backoff = 0.25
        while not self._stop.is_set():
            cap = self.capture_factory(self.source)
            if cap is None or not cap.isOpened():
                self.connected = False
                self.error = f"cannot open {self.name} source"
                if cap is not None:
                    cap.release()
                self._stop.wait(backoff)
                backoff = min(2.0, backoff * 2)
                self.reconnects += 1
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.connected, self.error, backoff = True, None, 0.25
            file_fps = cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0.0
            frame_period = 1.0 / file_fps if file_fps and file_fps > 0 else 0.0
            next_frame_at = time.monotonic()
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        self._publish(frame)
                        if frame_period:
                            next_frame_at += frame_period
                            self._stop.wait(max(0.0, next_frame_at - time.monotonic()))
                        continue
                    if self.is_file and self.loop_files:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        next_frame_at = time.monotonic()
                        continue
                    self.connected = False
                    self.error = "end of stream" if self.is_file else "frame read failed; reconnecting"
                    break
            finally:
                cap.release()
            if self.is_file and not self.loop_files:
                return
            self.reconnects += 1
            self._stop.wait(backoff)


class PairedFrameSource:
    """Pair the newest RGB and thermal packets within a timestamp tolerance."""

    def __init__(self, rgb: CaptureWorker, thermal: CaptureWorker, max_skew_ms: float = 50.0):
        if max_skew_ms <= 0:
            raise ValueError("max_skew_ms must be positive")
        self.rgb = rgb
        self.thermal = thermal
        self.max_skew_s = max_skew_ms / 1000.0
        self.used_rgb = 0
        self.used_thermal = 0
        self.dropped_rgb = 0
        self.dropped_thermal = 0
        self.last_skew_ms: float | None = None

    def start(self) -> None:
        self.rgb.start()
        self.thermal.start()

    def stop(self) -> None:
        self.rgb.stop()
        self.thermal.stop()

    def next_pair(self, timeout: float = 1.0) -> tuple[FramePacket, FramePacket] | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rgb, thermal = self.rgb.latest(), self.thermal.latest()
            if rgb and thermal and rgb.sequence > self.used_rgb and thermal.sequence > self.used_thermal:
                skew = rgb.timestamp - thermal.timestamp
                if abs(skew) <= self.max_skew_s:
                    self.dropped_rgb += max(0, rgb.sequence - self.used_rgb - 1)
                    self.dropped_thermal += max(0, thermal.sequence - self.used_thermal - 1)
                    self.used_rgb, self.used_thermal = rgb.sequence, thermal.sequence
                    self.last_skew_ms = abs(skew) * 1000
                    return rgb, thermal
                if skew < 0:
                    self.dropped_rgb += max(1, rgb.sequence - self.used_rgb)
                    self.used_rgb = rgb.sequence
                else:
                    self.dropped_thermal += max(1, thermal.sequence - self.used_thermal)
                    self.used_thermal = thermal.sequence
            time.sleep(0.003)
        return None

    def status(self) -> dict:
        return {
            "rgb": self.rgb.status(),
            "thermal": self.thermal.status(),
            "skew_ms": self.last_skew_ms,
            "max_skew_ms": self.max_skew_s * 1000,
            "dropped_rgb": self.dropped_rgb,
            "dropped_thermal": self.dropped_thermal,
        }


def annotate(frame: np.ndarray, detections: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    canvas = np.ascontiguousarray(frame.copy())
    counts: Counter[str] = Counter()
    scale = max(0.55, min(canvas.shape[:2]) / 900)
    thickness = max(1, round(scale * 2))
    for row in detections:
        x1, y1, x2, y2, confidence, cls_value = row.tolist()
        cls = int(cls_value)
        name, color = CLASS_NAMES.get(cls, str(cls)), COLORS.get(cls, (255, 255, 255))
        counts[name] += 1
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(canvas, p1, p2, color, thickness, cv2.LINE_AA)
        label = f"{name} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        top = max(0, p1[1] - th - baseline - 5)
        cv2.rectangle(canvas, (p1[0], top), (p1[0] + tw + 6, p1[1]), color, -1)
        cv2.putText(canvas, label, (p1[0] + 3, p1[1] - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 23, 42), thickness, cv2.LINE_AA)
    return canvas, dict(counts)


def compose_view(rgb: np.ndarray, thermal: np.ndarray, detections: np.ndarray, view: str) -> tuple[np.ndarray, dict]:
    if view not in VIEWS:
        raise ValueError(f"unsupported view {view}")
    thermal_color = cv2.applyColorMap(thermal, cv2.COLORMAP_INFERNO)
    if view == "rgb":
        return annotate(rgb, detections)
    if view == "thermal":
        return annotate(thermal_color, detections)
    rgb_annotated, counts = annotate(rgb, detections)
    thermal_annotated, _ = annotate(thermal_color, detections)
    separator = np.full((rgb.shape[0], 4, 3), (30, 41, 59), dtype=np.uint8)
    return np.hstack((rgb_annotated, separator, thermal_annotated)), counts


class DashboardRuntime:
    def __init__(
        self,
        rgb_source: int | str,
        thermal_source: int | str,
        weights: str | Path = DEFAULT_WEIGHTS,
        controller: str | Path = DEFAULT_CONTROLLER,
        device: str = "auto",
        confidence: float = 0.25,
        max_skew_ms: float = 50.0,
        loop_files: bool = False,
        capture_factory: Callable = cv2.VideoCapture,
        snapshot_dir: str | Path = REPO / "captures",
    ):
        self.device = torch.device("cuda:0" if device == "auto" and torch.cuda.is_available() else
                                   "cpu" if device == "auto" else device)
        self.weights = Path(weights)
        self.controller_path = Path(controller)
        self.confidence = float(confidence)
        if not 0.10 <= self.confidence <= 0.95:
            raise ValueError("confidence must be between 0.10 and 0.95")
        self.view = "rgb"
        self.policy_name = "sequence"
        self.snapshot_dir = Path(snapshot_dir)
        self.source = PairedFrameSource(
            CaptureWorker("RGB", rgb_source, loop_files, capture_factory),
            CaptureWorker("thermal", thermal_source, loop_files, capture_factory),
            max_skew_ms,
        )
        self.model = None
        self.policy = None
        self.running = False
        self.loading = False
        self.error: str | None = None
        self._lock = threading.RLock()
        self._frame_condition = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._jpeg: bytes | None = None
        self._frame_sequence = 0
        self._metadata: dict = {}
        self._latencies: deque[float] = deque(maxlen=120)
        self._completed_at: deque[float] = deque(maxlen=120)

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.weights.is_file():
            raise FileNotFoundError(f"model checkpoint not found: {self.weights}")
        if not self.controller_path.is_file():
            raise FileNotFoundError(f"controller artifact not found: {self.controller_path}")
        self.loading = True
        try:
            self.model = load_model(str(self.weights), self.device)
            self.policy = SequenceController.from_json(self.controller_path)
        finally:
            self.loading = False

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self.error = None
            try:
                self._load()
            except Exception as exc:
                self.error = str(exc)
                raise
            self._stop.clear()
            self.policy.reset()
            self._latencies.clear()
            self._completed_at.clear()
            self.source.start()
            self.running = True
            self._thread = threading.Thread(target=self._run, name="fusion-inference", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            self.running = False
        self.source.stop()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
        with self._frame_condition:
            self._frame_condition.notify_all()

    def close(self) -> None:
        self.stop()

    def set_control(self, payload: dict) -> None:
        if "confidence" in payload:
            confidence = float(payload["confidence"])
            if not 0.10 <= confidence <= 0.95:
                raise ValueError("confidence must be between 0.10 and 0.95")
            with self._lock:
                self.confidence = confidence
        if "view" in payload:
            if payload["view"] not in VIEWS:
                raise ValueError(f"view must be one of {sorted(VIEWS)}")
            with self._lock:
                self.view = payload["view"]
        if "policy" in payload:
            if payload["policy"] not in POLICIES:
                raise ValueError(f"policy must be one of {sorted(POLICIES)}")
            with self._lock:
                if payload["policy"] != self.policy_name:
                    self.policy_name = payload["policy"]
                    if self.policy is not None:
                        self.policy.reset()
        if "running" in payload:
            if not isinstance(payload["running"], bool):
                raise ValueError("running must be a boolean")
            self.start() if payload["running"] else self.stop()

    def _active_policy(self):
        return self.policy if self.policy_name == "sequence" else Fixed(576)

    def _run(self) -> None:
        frame_id = 0
        while not self._stop.is_set():
            pair = self.source.next_pair(timeout=0.5)
            if pair is None:
                continue
            rgb_packet, thermal_packet = pair
            try:
                fused, thermal = fuse_frames(rgb_packet.frame, thermal_packet.frame)
                with self._lock:
                    policy = self._active_policy()
                    confidence, view = self.confidence, self.view
                    policy_name = self.policy_name
                frame_id += 1
                identifier = f"live-{frame_id:09d}"
                resolution = policy.choose(identifier, None, frame=torch.from_numpy(fused))
                started = time.perf_counter()
                frame_gpu = torch.from_numpy(fused).to(self.device, non_blocking=self.device.type == "cuda")
                # The controller needs candidates down to 0.10 even when the UI threshold is higher.
                feedback = detect(self.model, frame_gpu, resolution, conf=0.10).cpu()
                policy.update(identifier, feedback)
                detections = feedback[feedback[:, 4] >= confidence].numpy()
                inference_ms = (time.perf_counter() - started) * 1000
                output, counts = compose_view(rgb_packet.frame, thermal, detections, view)
                ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                completed = time.monotonic()
                e2e_ms = (completed - max(rgb_packet.timestamp, thermal_packet.timestamp)) * 1000
                metadata = {
                    "frame": frame_id,
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "policy": policy_name,
                    "resolution": int(resolution),
                    "confidence": confidence,
                    "view": view,
                    "detections": int(len(detections)),
                    "class_counts": counts,
                    "inference_ms": round(inference_ms, 2),
                    "end_to_end_ms": round(e2e_ms, 2),
                }
                with self._frame_condition:
                    self.error = None
                    self._jpeg = encoded.tobytes()
                    self._metadata = metadata
                    self._frame_sequence += 1
                    # CUDA kernel setup makes the first frames unrepresentative. Keep
                    # displaying them, but exclude them from the rolling live rate.
                    if frame_id > 2:
                        self._latencies.append(inference_ms)
                        self._completed_at.append(completed)
                    self._frame_condition.notify_all()
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
                self._stop.wait(0.05)
        with self._lock:
            self.running = False

    def status(self) -> dict:
        with self._lock:
            times = list(self._completed_at)
            fps = 0.0
            if len(times) >= 2 and times[-1] > times[0]:
                fps = (len(times) - 1) / (times[-1] - times[0])
            latency = list(self._latencies)
            return {
                "running": self.running,
                "loading": self.loading,
                "error": self.error,
                "device": str(self.device),
                "model": self.weights.name,
                "policy": self.policy_name,
                "view": self.view,
                "confidence": self.confidence,
                "fps": round(fps, 1),
                "mean_inference_ms": round(float(np.mean(latency)), 2) if latency else None,
                "latest": dict(self._metadata),
                "sources": self.source.status(),
            }

    def mjpeg(self):
        seen = -1
        while True:
            with self._frame_condition:
                self._frame_condition.wait_for(lambda: self._frame_sequence != seen, timeout=2.0)
                if self._jpeg is None or self._frame_sequence == seen:
                    continue
                seen, jpeg = self._frame_sequence, self._jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def snapshot(self) -> tuple[Path, Path]:
        with self._lock:
            if self._jpeg is None:
                raise RuntimeError("no processed frame is available")
            jpeg, metadata = self._jpeg, dict(self._metadata)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("snapshot_%Y%m%d_%H%M%S_%f")
        image_path, json_path = self.snapshot_dir / f"{stem}.jpg", self.snapshot_dir / f"{stem}.json"
        image_path.write_bytes(jpeg)
        json_path.write_text(json.dumps(metadata, indent=2) + "\n")
        return image_path, json_path


def create_app(runtime) -> Flask:
    app = Flask(__name__, template_folder="dashboard/templates", static_folder="dashboard/static")

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/stream.mjpg")
    def stream():
        return Response(runtime.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def api_status():
        return jsonify(runtime.status())

    @app.post("/api/control")
    def api_control():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON object required"}), 400
        try:
            runtime.set_control(payload)
        except (TypeError, ValueError, RuntimeError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(runtime.status())

    @app.post("/api/snapshot")
    def api_snapshot():
        try:
            image_path, metadata_path = runtime.snapshot()
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        return jsonify({"image": str(image_path), "metadata": str(metadata_path)})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-source", required=True, help="camera index, video file, or RTSP URL")
    parser.add_argument("--thermal-source", required=True, help="camera index, video file, or RTSP URL")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--controller", default=str(DEFAULT_CONTROLLER))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-skew-ms", type=float, default=50.0)
    parser.add_argument("--loop", action="store_true", help="loop local video sources")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-autostart", action="store_true")
    args = parser.parse_args()

    runtime = DashboardRuntime(
        parse_source(args.rgb_source), parse_source(args.thermal_source), args.weights, args.controller,
        args.device, args.confidence, args.max_skew_ms, args.loop,
    )
    app = create_app(runtime)

    def shutdown(_signum=None, _frame=None):
        runtime.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    if not args.no_autostart:
        try:
            runtime.start()
        except Exception as exc:
            print(f"Dashboard started with inference stopped: {exc}")
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
