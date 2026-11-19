# FLIR browser workspace

This checkout runs beside the C++ camera workspace. Open `http://localhost:5000`
on the Jetson, or `http://<Jetson-IP>:5000` from a browser that can reach it.
The existing C++ dashboard remains on port 8080.

## Run

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements-flir.txt
bash scripts/run_flir_dashboard.sh
```

The C++ camera server must be running on `http://127.0.0.1:8080`.
Override it with the `FLIR_SERVER` environment variable. The launcher uses four
CPU threads. CUDA acceleration requires a Jetson-compatible CUDA PyTorch build;
the ordinary ARM64 PyPI wheel used here runs on CPU.

The UI provides annotated RGB/thermal/split views, start/stop, dynamic versus
fixed-576 resolution on the same multi-scale checkpoint, confidence adjustment,
class counts, measured processing rate, latency, snapshots, and a theme switch.
The C++ workspace link opens camera movement and calibration controls.
Snapshots and JSON metadata are saved in `captures/`.

## Camera bridge and measurement limits

`--flir-server` reads `/api/frame/fusion/rgb` and
`/api/frame/fusion/thermal` through a bounded HTTP snapshot adapter at up to
12 requests per second per stream. The C++ service owns RTSP acquisition and
applies its existing zoom-dependent calibration and overlap crop. Different
dimensions are rejected by the model input validation.

The two endpoints return independent JPEG snapshots without sensor timestamps.
The displayed skew measures host arrival times, so it does not establish
hardware synchronization. Frames are JPEG previews rather than raw sensor data.
Calibration quality should be checked visually for the current zoom and subject
distance; indoor parallax can still displace objects across modalities.

Live-camera tests establish operation and speed, not detection accuracy.
Scoring accuracy on FLIR imagery requires labeled, representative paired data.
This deployment uses the committed `ms_fusion_seed42` checkpoint. The README's
separate final fixed-576 fine-tuned checkpoint is absent from the repository.

## Background process

The local installation uses a user service named `early-fusion-dashboard.service`:

```bash
systemctl --user status early-fusion-dashboard
systemctl --user restart early-fusion-dashboard
systemctl --user stop early-fusion-dashboard
journalctl --user -u early-fusion-dashboard -f
```

It is started for this session; reboot autostart is not enabled. The server
listens on all interfaces so other browsers can reach it. It has no login and
is intended for the lab network; snapshots and inference controls are accessible
to anyone who can reach port 5000.

## Verified on this Jetson (2026-11-12)

- 15 targeted tests passed: HTTP snapshot decode/failure, dashboard routes,
  channel order and input validation, source pairing, sequence controller, and
  preprocessing dimensions.
- Live FLIR input: matching 711 × 405 calibrated RGB and thermal previews.
- Dynamic controller exercised 480, 576, and 640 px; fixed mode held 576 px.
- Measured approximately 8–9 processed FPS and 102–114 ms rolling mean inference
  time on four CPU threads. These are live-demo observations, not a controlled
  accuracy/energy benchmark or isolated per-policy latency comparison.
- RGB, thermal, split, confidence adjustment, snapshots/JSON, MJPEG, stop/restart,
  invalid-control rejection, and HTTP access through the LAN IP passed.
- Evidence: `captures/flir_live_validation.json`. Re-run with
  `.venv/bin/python scripts/smoke_flir_dashboard.py` while the service runs.
- Chromium browser automation also passed 16 checks: rendered live video, all
  inference/view controls, confidence slider, snapshot success, stop/start,
  camera workspace link, themes, desktop/mobile bounds, and no JavaScript errors.
  Screenshots were visually inspected at 1440 px and 390 px widths. A mobile
  split-video clipping bug found during this check was fixed and retested.
  Evidence: `captures/browser-test/result.json`, `desktop-dark.png`,
  `desktop-light.png`, and `mobile-dark.png` in the same directory.
- Repeat the browser checks with `NODE_PATH=/tmp/emch585-browser-test/node_modules
  node scripts/test_browser.cjs` (uses the locally installed `playwright-core`
  and Chromium; `CHROMIUM_PATH` can select another Chromium binary).
- `pip check` also reports an inherited system `pynacl` dependency missing `cffi`;
  this is unrelated to the inference stack and did not affect the checks above.
