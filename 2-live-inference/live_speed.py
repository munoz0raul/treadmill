#!/usr/bin/env python3
"""
live_speed.py — vision-powered live speed readout (no Bluetooth).

This is the pre-Bluetooth version of the app: point a camera at the treadmill,
run the mc3_18 CNN on the video, and show the estimated belt speed live on a web
page — your video on the left, the AI speed on the right. That's the whole thing.
Part 3 (game_server.py) is exactly this plus an FTMS Bluetooth peripheral so a
game can consume the speed.

    camera ──▶ CnnSpeedEstimator ──▶ web UI (:8090): live camera + AI speed

The CNN block (constants + CnnSpeedEstimator) is identical to the one in
game_server.py and mirrors the acquisition rig. The one invariant that keeps the
model accurate live is time-based clip sampling: CLIP_SPAN_S=1.0 s here MUST
match CLIP_SPAN_S in preprocess.py. Do not change one without the other.
"""

import argparse
import collections
import json
import logging
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── CNN speed estimator config (mirrors the acquisition rig) ────────────────────
CNN_MODEL       = os.path.expanduser("~/models/speed_cnn.onnx")
CNN_CLIP        = 112        # square resize target (matches preprocess.py)
CNN_N_FRAMES    = 8          # frames per clip
CNN_CLIP_SPAN_S = 1.0        # real-time span of one clip, in SECONDS. Must equal
                             #   CLIP_SPAN_S in preprocess.py.
CNN_CROP_X0     = 240        # central horizontal crop of a 1280-wide frame …
CNN_CROP_X1     = 1040       # … keep x∈[240,1040) — drops clutter, keeps the body
CNN_SMOOTH      = 6          # rolling average over N predictions (~3s window @2Hz)
CNN_CAL_SLOPE   = 1.0        # affine live calibration (identity after retrain)
CNN_CAL_OFFSET  = 0.0

CAM_WIDTH  = 1280
CAM_HEIGHT = 720
CAM_FPS    = 30


class CnnSpeedEstimator:
    """Buffers cropped RGB frames (tagged with wall-clock timestamps) and runs
    speed_cnn.onnx (mc3_18) via onnxruntime.

    Input clip: (1, 3, T, 112, 112) float32 [0..1]. The model bakes in the
    Kinetics normalisation internally, so we feed plain [0..1] RGB. Frames are
    centre-cropped (x∈[CROP_X0,CROP_X1)) then resized to 112×112.

    Time-based sampling: we pick T frames by *timestamp* over CLIP_SPAN_S seconds,
    so the live clip always spans the same real motion as the training clips
    regardless of the (variable) camera rate. Inference runs in a background
    thread; add_frame() is cheap and non-blocking.
    """

    def __init__(self, model_path=CNN_MODEL, n_frames=CNN_N_FRAMES,
                 clip=CNN_CLIP, smooth=CNN_SMOOTH):
        self._n = n_frames
        self._clip = clip
        self._smooth = smooth
        self._span   = CNN_CLIP_SPAN_S   # seconds — time-based, FPS-independent
        self._buf   = collections.deque(maxlen=512)
        self._preds = collections.deque(maxlen=smooth)
        self._speed = None
        self._mean_mov = 0.0
        self._lock  = threading.Lock()
        self._sess  = None
        self._running = False
        self._worker  = None

        if not _ORT_AVAILABLE:
            log.warning("onnxruntime not available — CNN speed estimation disabled")
            return
        if not os.path.exists(model_path):
            log.warning("CNN model not found at %s — CNN speed estimation disabled", model_path)
            return
        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        log.info("CNN speed estimator loaded from %s (clip span %.2fs, time-based)",
                 model_path, self._span)
        self._running = True
        self._worker = threading.Thread(target=self._infer_loop, daemon=True)
        self._worker.start()

    @property
    def ready(self) -> bool:
        return self._sess is not None

    def add_frame(self, frame_bgr):
        """Feed one BGR frame (cheap: crop + resize + append). Non-blocking."""
        if self._sess is None:
            return
        w = frame_bgr.shape[1]
        if w == 1280:
            x0, x1 = CNN_CROP_X0, CNN_CROP_X1
        else:
            x0 = int(w * CNN_CROP_X0 / 1280)
            x1 = int(w * CNN_CROP_X1 / 1280)
        crop = frame_bgr[:, x0:x1]
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (self._clip, self._clip), interpolation=cv2.INTER_AREA)
        with self._lock:
            self._buf.append((time.time(), small))

    def _sample_clip(self):
        """Pick T frames spanning the last `self._span` seconds by timestamp.
        Returns (1,3,T,H,W) float32 or None if the buffer doesn't yet cover the
        full span."""
        with self._lock:
            if len(self._buf) < self._n:
                return None
            buf = list(self._buf)
        t_end = buf[-1][0]
        t_start = t_end - self._span
        if buf[0][0] > t_start + 1e-3:
            return None
        ts_arr = [b[0] for b in buf]
        targets = [t_start + i * self._span / (self._n - 1) for i in range(self._n)]
        picks = []
        j = 0
        for tgt in targets:
            while j + 1 < len(ts_arr) and abs(ts_arr[j + 1] - tgt) <= abs(ts_arr[j] - tgt):
                j += 1
            picks.append(buf[j][1])
        clip = np.stack(picks, axis=0).astype(np.float32) / 255.0   # (T,H,W,3)
        clip = clip.transpose(3, 0, 1, 2)[np.newaxis]               # (1,3,T,H,W)
        return clip

    def _infer_loop(self):
        while self._running:
            clip = self._sample_clip()
            if clip is None:
                time.sleep(0.05)
                continue
            # Cheap motion gate: mean abs frame-to-frame diff of the clip.
            mean_mov = float(np.abs(np.diff(clip, axis=2)).mean())
            with self._lock:
                self._mean_mov = mean_mov
            # Gate at 0.006 so a stopped belt reads 0.0, not ~1 km/h (the model
            # never saw "stopped" in training and can't extrapolate to it).
            if mean_mov < 0.006:
                with self._lock:
                    self._preds.clear()
                    self._speed = 0.0
                time.sleep(0.1)
                continue
            raw = float(self._sess.run(["speed_kmh"], {"frames": clip})[0].flat[0])
            raw = (raw - CNN_CAL_OFFSET) / CNN_CAL_SLOPE   # affine live calibration
            raw = max(0.0, raw)
            with self._lock:
                self._preds.append(raw)
                self._speed = round(sum(self._preds) / len(self._preds), 1)

    def stop(self):
        self._running = False

    def latest_speed(self):
        with self._lock:
            return self._speed

    def debug_info(self):
        with self._lock:
            return {"mean_mov": round(self._mean_mov, 5), "speed": self._speed,
                    "buf": len(self._buf), "span_s": round(self._span, 3)}


# ── camera capture ──────────────────────────────────────────────────────────────
class CameraThread(threading.Thread):
    """Grab frames from the USB camera, feed every frame to the CNN, and keep the
    latest JPEG (with the AI speed drawn on it) for the MJPEG web stream."""

    def __init__(self, camera_dev, estimator):
        super().__init__(daemon=True)
        self._dev = camera_dev
        self._cnn = estimator
        self._jpeg = None
        self._fps = 0.0
        self._lock = threading.Lock()
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self._dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        if not cap.isOpened():
            log.error("Cannot open camera %s", self._dev)
            return
        log.info("Camera %s %dx%d @ %d fps", self._dev,
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(cap.get(cv2.CAP_PROP_FPS)))

        fps_counter, t0, fps = 0, time.time(), 0.0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            ts = time.time()
            if self._cnn is not None:
                self._cnn.add_frame(frame)

            fps_counter += 1
            elapsed = ts - t0
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter, t0 = 0, ts

            spd = self._cnn.latest_speed() if self._cnn is not None else None
            label = f"AI {spd:.1f} km/h" if spd is not None else "AI --.- km/h"
            cv2.putText(frame, label, (12, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 255, 0), 3)
            cv2.putText(frame, f"{fps:.0f} fps", (12, 84), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self._lock:
                self._jpeg = jpeg.tobytes()
                self._fps = fps
        cap.release()

    def get_frame(self):
        with self._lock:
            return self._jpeg

    def get_fps(self):
        with self._lock:
            return self._fps

    def stop(self):
        self._running = False


# ── globals wired up in main() ──────────────────────────────────────────────────
_cam = None
_cnn = None


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Treadmill — live speed</title>
<style>
  :root { --bg:#0b0f14; --bg2:#0d1117; --card:#161b22; --card2:#1c232d;
          --line:#2b333d; --fg:#e6edf3; --accent:#3fb950; --accent2:#2ea043;
          --cyan:#39c5cf; --dim:#8b949e; --radius:14px;
          --shadow:0 1px 3px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25); }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body { margin:0; background:
           radial-gradient(1200px 600px at 80% -10%, #12203a 0%, transparent 55%),
           radial-gradient(900px 500px at -10% 10%, #10261c 0%, transparent 50%),
           var(--bg);
         color:var(--fg); height:100vh; overflow:hidden;
         display:flex; flex-direction:column;
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  header { display:flex; align-items:center; gap:12px; padding:12px 22px; flex:none;
           border-bottom:1px solid var(--line); background:rgba(13,17,23,.6); }
  header .logo { font-size:20px; }
  header .title { font-size:17px; font-weight:700; }
  header .subtitle { font-size:12px; color:var(--dim); }
  .wrap { flex:1 1 auto; min-height:0; display:flex; gap:18px; padding:18px;
          max-width:1280px; width:100%; margin:0 auto; }
  .col { display:flex; flex-direction:column; min-height:0; min-width:0; }
  .col.left  { flex:1.9; }
  .col.right { flex:1; min-width:300px; max-width:420px; }
  .card { background:linear-gradient(180deg, var(--card2), var(--card));
          border:1px solid var(--line); border-radius:var(--radius);
          padding:16px; box-shadow:var(--shadow); }
  #camCard { flex:1 1 auto; min-height:0; display:flex; flex-direction:column; }
  .cam-frame { flex:1 1 auto; min-height:0; display:flex; border-radius:10px;
       overflow:hidden; background:#000; border:1px solid var(--line); }
  .cam { width:100%; height:100%; object-fit:contain; display:block; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em;
       color:var(--dim); margin:0 0 10px; font-weight:700; }
  .speed-card { text-align:center; padding:28px 20px; }
  .speed-wrap { display:flex; align-items:baseline; justify-content:center; gap:6px; }
  .speed { font-size:88px; font-weight:800; line-height:1; letter-spacing:-3px;
           background:linear-gradient(180deg,#5ee27a,#2ea043);
           -webkit-background-clip:text; background-clip:text; color:transparent;
           font-variant-numeric:tabular-nums; }
  .unit { font-size:24px; color:var(--dim); font-weight:600; }
  .src-tag { display:inline-flex; align-items:center; gap:6px; margin-top:12px;
             font-size:12px; color:var(--dim); padding:4px 12px; border-radius:999px;
             background:var(--bg2); border:1px solid var(--line); }
  .src-tag .d { width:7px; height:7px; border-radius:50%; background:var(--cyan); }
</style>
</head>
<body>
<header>
  <span class="logo">&#127939;</span>
  <div>
    <div class="title">AI Treadmill</div>
    <div class="subtitle">Live vision speed — no Bluetooth</div>
  </div>
</header>
<div class="wrap">
  <div class="col left">
    <div class="card" id="camCard">
      <h2>&#128247; Camera</h2>
      <div class="cam-frame"><img class="cam" src="/stream" alt="camera stream"></div>
    </div>
  </div>
  <div class="col right">
    <div class="card speed-card">
      <h2 style="justify-content:center">AI speed</h2>
      <div class="speed-wrap">
        <span class="speed" id="spd">--.-</span><span class="unit">km/h</span>
      </div>
      <div class="src-tag"><span class="d"></span><span>Reading from camera</span></div>
    </div>
  </div>
</div>
<script>
async function jget(u){ const r=await fetch(u); return r.json(); }
async function refresh(){
  try {
    const s = await jget('/ai_speed');
    document.getElementById('spd').textContent =
      (s.speed_kmh==null) ? '--.-' : s.speed_kmh.toFixed(1);
  } catch(e){}
}
setInterval(refresh, 500);
refresh();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_bytes(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj):
        self._send_bytes(json.dumps(obj).encode(), "application/json")

    def _stream_mjpeg(self):
        if _cam is None:
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            self.connection.settimeout(10.0)
        except Exception:
            pass
        try:
            while True:
                frame = _cam.get_frame()
                if frame:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/stream":
            self._stream_mjpeg()
        elif path == "/ai_speed":
            spd = _cnn.latest_speed() if _cnn is not None else None
            self._send_json({"speed_kmh": spd,
                             "ready": _cnn is not None and _cnn.ready})
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(
        description="Vision-powered live treadmill speed readout (no Bluetooth)")
    parser.add_argument("--camera", default="/dev/video0",
                        help="V4L2 device or index (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8090,
                        help="Web UI port (default: %(default)s)")
    parser.add_argument("--cnn-model", default=CNN_MODEL,
                        help="CNN ONNX model path (default: %(default)s)")
    args = parser.parse_args()

    global _cam, _cnn
    _cnn = CnnSpeedEstimator(model_path=args.cnn_model)
    if not _cnn.ready:
        log.warning("CNN estimator not ready — AI speed will read None")
    cam_dev = int(args.camera) if str(args.camera).lstrip("-").isdigit() else args.camera
    _cam = CameraThread(cam_dev, _cnn)
    _cam.start()

    log.info("Waiting for first camera frame ...")
    for _ in range(50):
        if _cam.get_frame():
            break
        time.sleep(0.1)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    log.info("Web UI on http://0.0.0.0:%d", args.port)

    def _graceful(signum, _frame):
        log.warning("Signal %d — shutting down", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if _cam is not None:
            _cam.stop()
        if _cnn is not None:
            _cnn.stop()
        server.server_close()


if __name__ == "__main__":
    main()
