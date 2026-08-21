#!/usr/bin/env python3
"""
game_server.py — vision-powered virtual treadmill for fitness games.

The treadmill has NO Bluetooth. This app points a camera at it, reads the belt
speed with the mc3_18 CNN (exactly as in the acquisition rig), and broadcasts
that speed over BLE as a standard FTMS (Fitness Machine Service) treadmill, so a
game (Zwift, Rouvy, the ftmsemu verifier, a phone app) connects and moves with it.

    camera ──▶ CnnSpeedEstimator ──▶ FtmsTreadmill (BLE peripheral) ──▶ game
                     │
                     └──▶ web UI (:8090): live camera + AI speed + Bluetooth panel

The CNN block below (constants + CnnSpeedEstimator) is the same estimator used in
Part 2's live_speed.py. The one invariant that makes the model accurate live is
time-based clip sampling: CLIP_SPAN_S=1.0 s here MUST match CLIP_SPAN_S in
preprocess.py and CNN_CLIP_SPAN_S in live_speed.py. Do not change one without the
others. (A shared module is a sensible later refactor.)
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

from ftms_peripheral import FtmsTreadmill

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── CNN speed estimator config (mirrors live_speed.py) ──────────────────────────
CNN_MODEL       = os.path.expanduser("~/models/speed_cnn.onnx")
CNN_CLIP        = 112        # square resize target (matches preprocess.py)
CNN_N_FRAMES    = 8          # frames per clip
CNN_FRAME_STRIDE = 4         # legacy — kept only for set_fps() back-compat
CNN_CLIP_SPAN_S = 1.0        # real-time span of one clip, in SECONDS. Must equal
                             #   CLIP_SPAN_S in preprocess.py / live_speed.py.
CNN_TRAIN_FPS   = 24.4       # legacy band-aid, UNUSED for span (span is time-based)
CNN_CROP_X0     = 240        # central horizontal crop of a 1280-wide frame …
CNN_CROP_X1     = 1040       # … keep x∈[240,1040) — drops clutter, keeps the body
CNN_SMOOTH      = 6          # rolling average over N predictions (~3s window @2Hz)
CNN_CAL_SLOPE   = 1.0        # affine live calibration (identity after retrain)
CNN_CAL_OFFSET  = 0.0

CAM_WIDTH  = 1280
CAM_HEIGHT = 720
CAM_FPS    = 30

REC_DIR = os.path.expanduser("~/recordings")   # where the REC button writes clips


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

    (The same estimator as Part 2's live_speed.py — see the module docstring.)
    """

    def __init__(self, model_path=CNN_MODEL, n_frames=CNN_N_FRAMES,
                 clip=CNN_CLIP, stride=CNN_FRAME_STRIDE, smooth=CNN_SMOOTH):
        self._n = n_frames
        self._clip = clip
        self._smooth = smooth
        self._stride = stride
        self._train_fps = CNN_TRAIN_FPS
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
            # Gate at 0.006 so a stopped belt pins to 0.0 immediately. The
            # training set includes stopped/empty-belt clips, but this cheap
            # backstop avoids low-motion jitter and clears the rolling average.
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


# ── camera capture (slim — no pose, no recorder) ────────────────────────────────
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
        # Recording: a cv2.VideoWriter is created on demand by start_recording()
        # and written frame-by-frame from run(). We record the same annotated
        # frame the web stream shows (AI speed burned in) — that's the shot worth
        # keeping for content. Guarded by _rec_lock (separate from the frame lock
        # so a slow disk write never stalls the MJPEG stream's frame handoff).
        self._writer = None
        self._rec_path = None
        self._rec_frames = 0
        self._rec_t0 = 0.0
        self._rec_lock = threading.Lock()

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

            # Draw the current AI speed on the preview (not on what the CNN sees).
            spd = self._cnn.latest_speed() if self._cnn is not None else None
            label = f"AI {spd:.1f} km/h" if spd is not None else "AI --.- km/h"
            cv2.putText(frame, label, (12, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        1.3, (0, 255, 0), 3)
            cv2.putText(frame, f"{fps:.0f} fps", (12, 84), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # If recording, write this annotated frame. A red "REC" dot marks the
            # recorded footage so it's obvious in the saved file (not just live).
            with self._rec_lock:
                if self._writer is not None:
                    cv2.circle(frame, (CAM_WIDTH - 40, 40), 12, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (CAM_WIDTH - 120, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    try:
                        self._writer.write(frame)
                        self._rec_frames += 1
                    except Exception as e:
                        log.error("recording write failed: %s", e)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self._lock:
                self._jpeg = jpeg.tobytes()
                self._fps = fps
        cap.release()
        # Flush any in-progress recording on shutdown so the file is playable.
        self.stop_recording()

    def get_frame(self):
        with self._lock:
            return self._jpeg

    def get_fps(self):
        with self._lock:
            return self._fps

    # ── recording control (thread-safe; called from HTTP handler threads) ───────
    def start_recording(self, fps_hint=None):
        """Open a VideoWriter and begin saving annotated frames to ~/recordings.
        Returns a status dict. No-op (returns the current status) if already
        recording."""
        with self._rec_lock:
            if self._writer is not None:
                return self._rec_status_locked()
            os.makedirs(REC_DIR, exist_ok=True)
            # time.strftime is fine here (real wall clock); avoids Date.now-style
            # issues that only bite workflow scripts, not this process.
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(REC_DIR, f"treadmill-{stamp}.mp4")
            fps = float(fps_hint) if fps_hint else (self._fps or CAM_FPS)
            fps = max(5.0, min(60.0, fps))   # keep playback speed sane
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, fps, (CAM_WIDTH, CAM_HEIGHT))
            if not writer.isOpened():
                log.error("could not open VideoWriter for %s", path)
                return {"ok": False, "error": "cannot open video writer"}
            self._writer = writer
            self._rec_path = path
            self._rec_frames = 0
            self._rec_t0 = time.time()
            log.info("recording started → %s (%.0f fps)", path, fps)
            return self._rec_status_locked()

    def stop_recording(self):
        """Finalise the current recording (safe to call when not recording)."""
        with self._rec_lock:
            if self._writer is None:
                return {"ok": True, "recording": False}
            path, frames = self._rec_path, self._rec_frames
            dur = time.time() - self._rec_t0
            try:
                self._writer.release()
            except Exception as e:
                log.error("recording release failed: %s", e)
            self._writer = None
            self._rec_path = None
            log.info("recording stopped → %s (%d frames, %.1fs)", path, frames, dur)
            return {"ok": True, "recording": False, "file": path,
                    "frames": frames, "seconds": round(dur, 1)}

    def _rec_status_locked(self):
        rec = self._writer is not None
        dur = (time.time() - self._rec_t0) if rec else 0.0
        return {"ok": True, "recording": rec,
                "file": os.path.basename(self._rec_path) if self._rec_path else None,
                "frames": self._rec_frames if rec else 0,
                "seconds": round(dur, 1)}

    def rec_status(self):
        with self._rec_lock:
            return self._rec_status_locked()

    def stop(self):
        self._running = False


# ── manual speed source (for validating the BLE path without a camera) ──────────
class ManualSpeed:
    """A thread-safe, hand-set speed. In --manual mode this replaces the CNN as
    the speed_provider, so you can drag a slider in the web UI and watch the game
    move — a way to validate the whole BLE→game path before trusting the camera."""

    def __init__(self, initial=0.0):
        self._kmh = float(initial)
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._kmh

    def set(self, kmh):
        kmh = max(0.0, min(30.0, float(kmh)))   # clamp to a sane treadmill range
        with self._lock:
            self._kmh = kmh
        return kmh


# ── globals wired up in main() ──────────────────────────────────────────────────
_cam = None
_cnn = None
_ftms = None
_manual = None   # ManualSpeed instance when running with --manual


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Treadmill</title>
<style>
  :root { --bg:#0b0f14; --bg2:#0d1117; --card:#161b22; --card2:#1c232d;
          --line:#2b333d; --fg:#e6edf3; --accent:#3fb950; --accent2:#2ea043;
          --amber:#d29922; --cyan:#39c5cf; --blue:#4c8dff; --red:#f85149;
          --dim:#8b949e; --radius:14px;
          --shadow:0 1px 3px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25); }
  * { box-sizing: border-box; }
  html, body { height:100%; }
  body { margin:0; background:
           radial-gradient(1200px 600px at 80% -10%, #12203a 0%, transparent 55%),
           radial-gradient(900px 500px at -10% 10%, #10261c 0%, transparent 50%),
           var(--bg);
         color:var(--fg); height:100vh; overflow:hidden;
         display:flex; flex-direction:column;
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         -webkit-font-smoothing:antialiased; }
  header { display:flex; align-items:center; gap:12px; padding:12px 22px; flex:none;
           border-bottom:1px solid var(--line);
           background:rgba(13,17,23,.6); backdrop-filter:blur(8px); z-index:10; }
  header .logo { font-size:20px; }
  header .title { font-size:17px; font-weight:700; letter-spacing:.2px; }
  header .subtitle { font-size:12px; color:var(--dim); font-weight:500; }
  header .spacer { flex:1; }
  .wrap { flex:1 1 auto; min-height:0; display:flex; gap:18px; padding:18px;
          max-width:1280px; width:100%; margin:0 auto; }
  .col { display:flex; flex-direction:column; min-height:0; min-width:0; }
  .col.left  { flex:1.9; }         /* camera column — the big one */
  .col.right { flex:1; min-width:330px; max-width:440px; }
  .card { background:linear-gradient(180deg, var(--card2), var(--card));
          border:1px solid var(--line); border-radius:var(--radius);
          padding:16px; box-shadow:var(--shadow); }
  .col > .card + .card { margin-top:16px; }
  /* camera card fills the whole left column; the image grows to fill it */
  #camCard { flex:1 1 auto; min-height:0; display:flex; flex-direction:column; }
  .cam-frame { flex:1 1 auto; min-height:0; display:flex; border-radius:10px;
       overflow:hidden; background:#000; border:1px solid var(--line); }
  .cam { width:100%; height:100%; object-fit:contain; display:block; }
  h2 { display:flex; align-items:center; gap:8px;
       font-size:12px; text-transform:uppercase; letter-spacing:.08em;
       color:var(--dim); margin:0 0 10px; font-weight:700; }
  h2 svg { width:15px; height:15px; opacity:.8; }
  label { display:block; font-size:13px; color:var(--dim); margin:0 0 6px; }

  /* ── speed hero ─────────────────────────────────────────────── */
  .speed-card { text-align:center; padding:14px 20px; flex:none; }
  .speed-wrap { display:flex; align-items:baseline; justify-content:center; gap:6px; }
  .speed { font-size:64px; font-weight:800; line-height:1; letter-spacing:-2px;
           background:linear-gradient(180deg,#5ee27a,#2ea043);
           -webkit-background-clip:text; background-clip:text; color:transparent;
           font-variant-numeric:tabular-nums; }
  .unit { font-size:20px; color:var(--dim); font-weight:600; }
  .src-tag { display:inline-flex; align-items:center; gap:6px; margin-top:8px;
             font-size:12px; color:var(--dim); padding:4px 12px; border-radius:999px;
             background:var(--bg2); border:1px solid var(--line); }
  .src-tag .d { width:7px; height:7px; border-radius:50%; background:var(--cyan); }

  /* ── inputs & buttons ───────────────────────────────────────── */
  input[type=text] { width:100%; padding:11px 13px; border-radius:10px;
       border:1px solid var(--line); background:var(--bg2); color:var(--fg);
       font-size:15px; transition:border-color .15s, box-shadow .15s; }
  input[type=text]:focus { outline:none; border-color:var(--blue);
       box-shadow:0 0 0 3px rgba(76,141,255,.18); }
  .btn { padding:12px 16px; border-radius:10px; border:1px solid var(--line);
       font-size:14px; font-weight:700; cursor:pointer; background:var(--card2);
       color:var(--fg); transition:transform .06s, filter .15s, background .15s;
       display:inline-flex; align-items:center; justify-content:center; gap:8px; }
  .btn:hover { filter:brightness(1.12); }
  .btn:active { transform:translateY(1px); }
  .btn:disabled { opacity:.4; cursor:not-allowed; filter:none; }
  .btn-go { background:linear-gradient(180deg,var(--accent),var(--accent2));
       color:#03260c; border-color:transparent; }
  .btn-stop { background:transparent; color:var(--red); border-color:#5c2b2b; }
  .btn-ghost { background:transparent; color:var(--dim); }
  .btn-row { display:flex; gap:10px; margin-top:12px; }
  .btn-row .btn-go { flex:2; }
  .btn-row .btn-stop, .btn-row .btn-ghost { flex:1; }

  /* ── record ─────────────────────────────────────────────────── */
  .btn-rec { width:100%; margin-top:12px; flex:none; background:var(--card2); }
  .btn-rec.recording { background:linear-gradient(180deg,#ff6b63,var(--red));
       color:#2b0b09; border-color:transparent; }
  .rec-row { display:flex; align-items:center; gap:8px; margin-top:8px; flex:none;
       font-size:13px; color:var(--dim); }
  .rec-row .dot.rec { background:var(--red); animation:pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; transform:scale(1);} 50% { opacity:.25; transform:scale(.8);} }

  /* ── Bluetooth panel ────────────────────────────────────────── */
  .badge { margin-left:auto; font-size:11px; font-weight:700; letter-spacing:.04em;
       padding:4px 10px; border-radius:999px; text-transform:uppercase;
       background:var(--bg2); color:var(--dim); border:1px solid var(--line); }
  .badge.blue  { background:rgba(76,141,255,.15); color:#8ab4ff; border-color:rgba(76,141,255,.4); }
  .badge.amber { background:rgba(210,153,34,.15); color:#e3b341; border-color:rgba(210,153,34,.4); }
  .badge.green { background:rgba(63,185,80,.15); color:#56d364; border-color:rgba(63,185,80,.4); }
  .badge.red   { background:rgba(248,81,73,.15); color:#ff7b72; border-color:rgba(248,81,73,.4); }

  .ble-hero { display:flex; align-items:center; gap:14px; padding:12px 14px;
       border-radius:12px; background:var(--bg2); border:1px solid var(--line);
       margin-bottom:12px; }
  .ble-ring { flex:none; width:48px; height:48px; border-radius:50%;
       display:flex; align-items:center; justify-content:center;
       background:#161b22; border:2px solid var(--line); color:var(--dim);
       transition:all .3s; }
  .ble-ring svg { width:22px; height:22px; }
  .ble-ring.blue  { border-color:var(--blue); color:#8ab4ff;
       box-shadow:0 0 0 4px rgba(76,141,255,.12); animation:breathe 1.6s infinite; }
  .ble-ring.amber { border-color:var(--amber); color:#e3b341;
       box-shadow:0 0 0 4px rgba(210,153,34,.12); }
  .ble-ring.green { border-color:var(--accent); color:#56d364;
       box-shadow:0 0 0 4px rgba(63,185,80,.14); }
  @keyframes breathe { 0%,100%{ box-shadow:0 0 0 4px rgba(76,141,255,.05);} 50%{ box-shadow:0 0 0 8px rgba(76,141,255,.18);} }
  .ble-title { font-size:15px; font-weight:700; line-height:1.3; }
  .ble-sub { font-size:12.5px; color:var(--dim); margin-top:2px; line-height:1.45; }
  .ble-sub b { color:var(--fg); }

  /* connection stepper */
  .stepper { display:flex; align-items:center; margin:2px 2px 14px; }
  .step { display:flex; align-items:center; gap:7px; font-size:12px; font-weight:600;
       color:var(--dim); white-space:nowrap; }
  .step .sd { width:16px; height:16px; border-radius:50%; border:2px solid var(--line);
       background:var(--bg2); display:flex; align-items:center; justify-content:center;
       font-size:9px; color:transparent; transition:all .3s; }
  .step.done .sd { background:var(--accent); border-color:var(--accent); color:#03260c; }
  .step.done { color:var(--fg); }
  .step.active .sd { border-color:var(--blue); box-shadow:0 0 0 3px rgba(76,141,255,.18); }
  .step.active { color:#8ab4ff; }
  .step-line { flex:1; height:2px; background:var(--line); margin:0 8px; border-radius:2px;
       transition:background .3s; }
  .step-line.done { background:var(--accent); }

  /* help note */
  .note { margin-top:12px; border-radius:10px; background:var(--bg2);
       border:1px solid var(--line); font-size:12.5px; line-height:1.6;
       color:var(--dim); overflow:hidden; }
  .note summary { padding:10px 13px; cursor:pointer; font-weight:600; color:var(--fg);
       list-style:none; display:flex; align-items:center; gap:8px; }
  .note summary::-webkit-details-marker { display:none; }
  .note summary::before { content:'?'; width:18px; height:18px; flex:none;
       border-radius:50%; background:var(--card2); border:1px solid var(--line);
       display:flex; align-items:center; justify-content:center; font-size:11px; }
  .note[open] summary { border-bottom:1px solid var(--line); }
  .note .body { padding:11px 13px; }
  .note b { color:var(--fg); }

  /* ── manual slider ──────────────────────────────────────────── */
  .hidden { display:none; }
  input[type=range] { width:100%; margin:10px 0 4px; accent-color:var(--accent);
       height:6px; }
  .presets { display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; }
  .presets button { padding:8px 14px; background:var(--card2); color:var(--fg);
       border:1px solid var(--line); border-radius:8px; font-weight:600;
       cursor:pointer; font-size:14px; }
  .presets button:hover { filter:brightness(1.15); }
  .bigset { font-size:44px; font-weight:800; color:var(--cyan);
       font-variant-numeric:tabular-nums; }
  .status { margin-top:14px; font-size:14px; line-height:1.7; }
  .kv { color:var(--dim); } .kv b { color:var(--fg); font-weight:600; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%;
         background:var(--dim); vertical-align:middle; }
</style>
</head>
<body>
<header>
  <span class="logo">&#127939;</span>
  <div>
    <div class="title">AI Treadmill</div>
    <div class="subtitle">Vision-powered FTMS bridge</div>
  </div>
  <span class="spacer"></span>
  <span class="badge" id="topBadge">Idle</span>
</header>
<div class="wrap">
  <div class="col left">
    <div class="card" id="camCard">
      <h2>&#128247; Camera</h2>
      <div class="cam-frame"><img class="cam" src="/stream" alt="camera stream"></div>
      <button class="btn btn-rec" id="btnRec" onclick="toggleRec()">&#9210; RECORD</button>
      <div class="rec-row" id="recStatus">
        <span class="dot" id="recDot"></span><span id="recText">Not recording</span>
      </div>
    </div>
    <div class="card hidden" id="manualCard">
      <h2>&#127903; Manual speed</h2>
      <div><span class="bigset" id="setVal">0.0</span> <span class="unit">km/h</span></div>
      <input type="range" id="slider" min="0" max="16" step="0.1" value="0"
             oninput="setSpeed(this.value)">
      <div class="presets">
        <button onclick="preset(0)">Stop</button>
        <button onclick="preset(3)">3</button>
        <button onclick="preset(5)">5</button>
        <button onclick="preset(6)">6</button>
        <button onclick="preset(8)">8</button>
        <button onclick="preset(10)">10</button>
        <button onclick="preset(12)">12</button>
      </div>
      <div class="status kv">Drag or tap a preset to "fake walk" &mdash; the value is
        streamed to the connected game over FTMS.</div>
    </div>
  </div>
  <div class="col right">
    <div class="card speed-card">
      <h2 id="speedHdr" style="justify-content:center">AI speed</h2>
      <div class="speed-wrap">
        <span class="speed" id="spd">--.-</span><span class="unit">km/h</span>
      </div>
      <div class="src-tag"><span class="d"></span><span id="srcTag">Reading from camera</span></div>
    </div>
    <div class="card">
      <h2>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
             stroke-linecap="round" stroke-linejoin="round"><path d="m7 7 10 10-5 5V2l5 5L7 17"/></svg>
        Bluetooth &mdash; broadcast to a game
        <span class="badge" id="bleBadge">Idle</span>
      </h2>

      <div class="ble-hero">
        <div class="ble-ring" id="bleRing">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round"><path d="m7 7 10 10-5 5V2l5 5L7 17"/></svg>
        </div>
        <div>
          <div class="ble-title" id="bleText">Not advertising</div>
          <div class="ble-sub" id="bleDetail">Press <b>Start broadcasting</b> to appear in your game.</div>
        </div>
      </div>

      <div class="stepper" id="stepper">
        <div class="step" id="st1"><span class="sd">&#10003;</span>Advertising</div>
        <div class="step-line" id="sl1"></div>
        <div class="step" id="st2"><span class="sd">&#10003;</span>Device linked</div>
        <div class="step-line" id="sl2"></div>
        <div class="step" id="st3"><span class="sd">&#10003;</span>Game active</div>
      </div>

      <label for="name">Device name (as seen by the game)</label>
      <input type="text" id="name" value="AI Treadmill">
      <div class="btn-row">
        <button class="btn btn-go"    id="btnStart" onclick="startBle()">Start broadcasting</button>
        <button class="btn btn-stop"  id="btnStop"  onclick="stopBle()">Stop</button>
        <button class="btn btn-ghost" id="btnReset" onclick="resetBle()">Reset</button>
      </div>

      <details class="note">
        <summary>Can't find "AI Treadmill" in your game?</summary>
        <div class="body">
          A BLE fitness device does <b>not</b> show up in your
          phone/Mac/tablet <b>Bluetooth settings</b> &mdash; only inside a fitness
          <b>game</b> or a BLE scanner app (nRF&nbsp;Connect, LightBlue). In Zwift,
          look under <b>Run &rarr; Run Speed</b>. If a stray device grabbed the link
          and others can't find it, press <b>Reset</b>.
        </div>
      </details>
    </div>
  </div>
</div>
<script>
async function jget(u){ const r=await fetch(u); return r.json(); }
async function jpost(u,b){ const r=await fetch(u,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
    return r.json(); }

let MANUAL = false;
let RECORDING = false;

async function toggleRec(){
  const r = await jpost(RECORDING ? '/rec/stop' : '/rec/start', {});
  if (r && r.ok === false && r.error) {
    document.getElementById('recText').textContent = 'Cannot record: ' + r.error;
  }
  refreshRec();
}

async function refreshRec(){
  if (MANUAL) return;   // no camera in manual/fake mode
  try {
    const r = await jget('/rec/status');
    RECORDING = !!r.recording;
    const btn = document.getElementById('btnRec');
    const dot = document.getElementById('recDot');
    const txt = document.getElementById('recText');
    if (RECORDING) {
      btn.classList.add('recording');
      btn.innerHTML = '&#9209; STOP RECORDING';
      dot.className = 'dot rec';
      txt.innerHTML = 'Recording <b>' + (r.file||'') + '</b> — ' +
        (r.seconds||0).toFixed(0) + 's';
    } else {
      btn.classList.remove('recording');
      btn.innerHTML = '&#9210; RECORD';
      dot.className = 'dot';
      txt.textContent = r.file ? ('Saved ' + r.file) : 'Not recording';
    }
  } catch(e){}
}

async function initMode(){
  const m = await jget('/mode');
  MANUAL = !!m.manual;
  if (MANUAL) {
    document.getElementById('camCard').classList.add('hidden');
    document.getElementById('manualCard').classList.remove('hidden');
    document.getElementById('speedHdr').textContent = 'Broadcast speed';
    document.getElementById('srcTag').textContent = 'Manual — set by hand';
  }
}

async function setSpeed(v){
  document.getElementById('setVal').textContent = Number(v).toFixed(1);
  await jpost('/manual/speed', {speed_kmh: Number(v)});
}
function preset(v){
  document.getElementById('slider').value = v;
  setSpeed(v);
}

async function startBle(){
  const name=document.getElementById('name').value.trim()||'AI Treadmill';
  await jpost('/ble/name',{name});
  await jpost('/ble/start',{});
  refresh();
}
async function stopBle(){ await jpost('/ble/stop',{}); refresh(); }
async function resetBle(){
  document.getElementById('bleText').textContent = 'Resetting…';
  await jpost('/ble/restart',{});
  refresh();
}

async function refresh(){
  try {
    const s=await jget('/ai_speed');
    document.getElementById('spd').textContent =
      (s.speed_kmh==null)?'--.-':s.speed_kmh.toFixed(1);

    const b=await jget('/ble/status');
    const ring=document.getElementById('bleRing');
    const txt=document.getElementById('bleText');
    const det=document.getElementById('bleDetail');
    const badge=document.getElementById('bleBadge');
    const top=document.getElementById('topBadge');

    // stepper stage: 0 idle, 1 advertising, 2 device linked, 3 game active
    let stage = 0;
    let ringCls='ble-ring', badgeCls='badge', badgeTxt='Idle';

    if(!b.available){
      ring.className='ble-ring'; badge.className='badge red'; badge.textContent='N/A';
      top.className='badge red'; top.textContent='N/A';
      txt.textContent='Bluetooth unavailable';
      det.innerHTML='<b>bless</b> is not installed on the board.';
      setStepper(0); return;
    } else if(!b.advertising && !b.link_up){
      stage=0; ringCls='ble-ring'; badgeCls='badge'; badgeTxt='Idle';
      txt.textContent='Not advertising';
      det.innerHTML='Press <b>Start broadcasting</b> to appear in your game.';
    } else if(b.game_active){
      stage=3; ringCls='ble-ring green'; badgeCls='badge green'; badgeTxt='Live';
      txt.innerHTML='Game connected to <b>'+b.name+'</b>';
      det.innerHTML='Streaming <b>'+(b.speed_kmh==null?'0.0':b.speed_kmh.toFixed(1))+'</b> km/h &mdash; you\\'re good to go!';
    } else if(b.link_up){
      stage=2; ringCls='ble-ring amber'; badgeCls='badge amber'; badgeTxt='Linked';
      txt.innerHTML='A device is linked to <b>'+b.name+'</b>';
      det.innerHTML='Waiting for the game to start the treadmill handshake. '
        +'If it\\'s a stray device, press <b>Reset</b>.';
    } else {
      stage=1; ringCls='ble-ring blue'; badgeCls='badge blue'; badgeTxt='Advertising';
      txt.innerHTML='Advertising as <b>'+b.name+'</b>';
      det.innerHTML='Scan for it in your game (Zwift: <b>Run &rarr; Run Speed</b>).';
    }
    ring.className=ringCls;
    badge.className=badgeCls; badge.textContent=badgeTxt;
    top.className=badgeCls; top.textContent=badgeTxt;
    setStepper(stage);
  } catch(e){}
}

function setStepper(stage){
  // stage 1 = advertising active, 2 = linked, 3 = game active
  const steps=[['st1','sl1'],['st2','sl2'],['st3',null]];
  for(let i=0;i<3;i++){
    const el=document.getElementById(steps[i][0]);
    el.classList.remove('done','active');
    if(stage>i+1) el.classList.add('done');
    else if(stage===i+1) el.classList.add(stage===3?'done':'active');
  }
  document.getElementById('sl1').classList.toggle('done', stage>=2);
  document.getElementById('sl2').classList.toggle('done', stage>=3);
}
setInterval(refresh, 700);
setInterval(refreshRec, 700);
initMode().then(() => { refresh(); refreshRec(); });
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

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(body) if body else {}
        except Exception:
            return {}

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
        elif path == "/mode":
            self._send_json({"manual": _manual is not None})
        elif path == "/ai_speed":
            if _manual is not None:
                self._send_json({"speed_kmh": round(_manual.get(), 1), "ready": True})
            else:
                spd = _cnn.latest_speed() if _cnn is not None else None
                self._send_json({"speed_kmh": spd,
                                 "ready": _cnn is not None and _cnn.ready})
        elif path == "/ble/status":
            self._send_json(_ftms.status() if _ftms is not None
                            else {"available": False})
        elif path == "/rec/status":
            self._send_json(_cam.rec_status() if _cam is not None
                            else {"ok": False, "recording": False,
                                  "error": "no camera"})
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        data = self._read_json()
        # Manual-speed control works regardless of BLE state (validate the UI even
        # with --no-ble).
        if path == "/manual/speed":
            if _manual is None:
                self._send_json({"ok": False, "error": "not in manual mode"}); return
            v = _manual.set(data.get("speed_kmh", 0.0))
            self._send_json({"ok": True, "speed_kmh": round(v, 1)}); return
        # Recording control is independent of BLE — you can record with --no-ble.
        if path == "/rec/start":
            if _cam is None:
                self._send_json({"ok": False, "error": "no camera (manual/fake mode)"})
            else:
                self._send_json(_cam.start_recording())
            return
        if path == "/rec/stop":
            if _cam is None:
                self._send_json({"ok": False, "error": "no camera"})
            else:
                self._send_json(_cam.stop_recording())
            return
        if _ftms is None:
            self._send_json({"ok": False, "error": "BLE disabled"}); return
        if path == "/ble/start":
            self._send_json(_ftms.start())
        elif path == "/ble/stop":
            self._send_json(_ftms.stop())
        elif path == "/ble/restart":
            self._send_json(_ftms.restart())
        elif path == "/ble/name":
            self._send_json(_ftms.set_name(data.get("name", "AI Treadmill")))
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(
        description="Vision-powered virtual FTMS treadmill for fitness games")
    parser.add_argument("--camera", default="/dev/video0",
                        help="V4L2 device or index (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8090,
                        help="Web UI port (default: %(default)s)")
    parser.add_argument("--name", default="AI Treadmill",
                        help="BLE device name shown to the game (default: %(default)s)")
    parser.add_argument("--cnn-model", default=CNN_MODEL,
                        help="CNN ONNX model path (default: %(default)s)")
    parser.add_argument("--no-ble", action="store_true",
                        help="Disable the BLE peripheral (camera + UI only)")
    parser.add_argument("--manual", action="store_true",
                        help="Manual mode: no camera/CNN — set the broadcast speed "
                             "by hand from the web UI (slider + presets). Use this to "
                             "validate the BLE connection and 'fake walk' into a game.")
    parser.add_argument("--fake-speed", action="store_true",
                        help="Ignore the camera and broadcast a synthetic 0->8->0 "
                             "ramp (for testing the BLE side without walking)")
    args = parser.parse_args()

    global _cam, _cnn, _ftms, _manual

    # Speed source, in priority order:
    #   --manual     → hand-set via the web UI (validate BLE without a camera)
    #   --fake-speed → synthetic 0->8->0 ramp (unattended BLE smoke test)
    #   default      → the CNN reading the live camera
    if args.manual:
        _manual = ManualSpeed(0.0)
        speed_provider = _manual.get
        log.info("Manual mode: set the broadcast speed from the web UI")
    elif args.fake_speed:
        _t0 = time.time()
        def speed_provider():
            phase = (time.time() - _t0) % 32.0
            return phase / 2.0 if phase < 16.0 else (32.0 - phase) / 2.0
        log.info("Fake-speed mode: broadcasting a synthetic 0->8->0 km/h ramp")
    else:
        _cnn = CnnSpeedEstimator(model_path=args.cnn_model)
        if not _cnn.ready:
            log.warning("CNN estimator not ready — AI speed will read None")
        cam_dev = int(args.camera) if str(args.camera).lstrip("-").isdigit() else args.camera
        _cam = CameraThread(cam_dev, _cnn)
        _cam.start()
        speed_provider = _cnn.latest_speed

    if not args.no_ble:
        _ftms = FtmsTreadmill(name=args.name, speed_provider=speed_provider)
        if not _ftms.available:
            log.warning("bless not installed — BLE disabled (pip install bless)")
    else:
        log.info("BLE disabled (--no-ble)")

    if _cam is not None:
        log.info("Waiting for first camera frame ...")
        for _ in range(50):
            if _cam.get_frame():
                break
            time.sleep(0.1)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    log.info("Web UI on http://0.0.0.0:%d  (press START BROADCASTING to advertise)",
             args.port)

    def _graceful(signum, _frame):
        log.warning("Signal %d — shutting down", signum)
        try:
            if _ftms is not None:
                _ftms.stop()
        finally:
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if _ftms is not None:
            _ftms.stop()
        if _cam is not None:
            _cam.stop()
        if _cnn is not None:
            _cnn.stop()
        server.server_close()


if __name__ == "__main__":
    main()
