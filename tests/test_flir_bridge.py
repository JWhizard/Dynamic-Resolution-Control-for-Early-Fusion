"""Exercise the real HTTP snapshot adapter, including a failed camera response."""
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flir_bridge import SnapshotCapture


def test_snapshot_capture_decodes_http_and_handles_failure():
    _, jpeg = cv2.imencode('.jpg', np.full((12, 16, 3), 80, np.uint8))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == '/frame' else 503)
            self.end_headers()
            self.wfile.write(jpeg.tobytes() if self.path == '/frame' else b'unavailable')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        base = f'http://127.0.0.1:{server.server_port}'
        capture = SnapshotCapture(base + '/frame')
        ok, frame = capture.read()
        assert ok and frame.shape == (12, 16, 3)
        assert np.all(frame == 80)
        capture.release()
        assert not capture.isOpened()
        assert SnapshotCapture(base + '/missing').read() == (False, None)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
