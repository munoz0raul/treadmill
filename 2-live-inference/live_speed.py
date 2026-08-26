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

CPU-vs-NPU benchmark mode (opt-in, off by default)
--------------------------------------------------
The app also doubles as the reference-capture tool for the CPU-vs-accelerator
comparison. Three flags turn on instrumentation; without them the app is byte-for
-byte the plain live readout:

    --log run.jsonl      one JSON line per inference with the RAW (pre-smoothing)
                         prediction, per-inference latency, motion-gate value, and
                         the operator-dialled panel speed (ground truth).
    --dump-clips DIR     saves the EXACT (1,3,8,112,112) float32 tensor fed to the
                         model as clip_NNNNNN.npy — the same bytes can be replayed
                         through the NPU offline, so the comparison isolates "did
                         the engine change the answer?" from walk-to-walk variation.
    --set-speed N        (web button) records the current panel speed into the log
                         so accuracy can be reported per speed band.

The intended workflow: record one structured 0→8 km/h ramp on the CPU with
--log + --dump-clips, then run the dumped clips through the NPU offline and diff
clip-by-clip (numeric fidelity), plus a short live NPU session for latency/FPS.
See 2-live-inference/README.md → "Comparing CPU vs NPU" for the full protocol.
"""

import argparse
import collections
import hashlib
import json
import logging
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

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


def _sha256(path):
    """Short sha256 of a file, or None if unreadable — stamps the model identity
    into the bench log so a CPU run and an NPU run can't be silently mismatched."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=os.path.dirname(os.path.abspath(__file__)),
                             capture_output=True, text=True, timeout=2)
        return out.stdout.strip() or None
    except Exception:
        return None


class BenchLogger:
    """Opt-in recorder for the CPU-vs-NPU comparison. Writes a JSONL header
    (session-level: provider, model sha, sampling params, host, commit) followed
    by one line per inference. Optionally dumps the exact model-input tensors so
    the NPU can replay bit-identical clips offline. A no-op unless --log or
    --dump-clips is given, so the published app is unaffected by default."""

    def __init__(self, log_path=None, dump_dir=None, dump_every=1,
                 provider="cpu", model_path=CNN_MODEL):
        self.enabled = bool(log_path or dump_dir)
        self._fp = None
        self._dump_dir = dump_dir
        self._dump_every = max(1, int(dump_every))
        self._provider = provider
        self._lock = threading.Lock()
        self._clip_id = 0
        self._set_speed = None          # latest operator-dialled panel speed
        if not self.enabled:
            return
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        if log_path:
            self._fp = open(log_path, "a", buffering=1)   # line-buffered
            header = {
                "type": "header", "ts": time.time(), "provider": provider,
                "model": os.path.basename(model_path),
                "model_sha256": _sha256(model_path),
                "clip_span_s": CNN_CLIP_SPAN_S, "n_frames": CNN_N_FRAMES,
                "clip": CNN_CLIP, "smooth": CNN_SMOOTH,
                "cal_slope": CNN_CAL_SLOPE, "cal_offset": CNN_CAL_OFFSET,
                "host": os.uname().nodename, "git_commit": _git_commit(),
                "dump_clips": bool(dump_dir), "dump_every": self._dump_every,
            }
            self._fp.write(json.dumps(header) + "\n")
        log.info("BenchLogger active — provider=%s log=%s dump=%s",
                 provider, log_path, dump_dir)

    def set_speed(self, kmh):
        with self._lock:
            self._set_speed = kmh

    def record(self, raw_kmh, speed_kmh, infer_ms, mean_mov, clip=None):
        """Called once per inference by the estimator. Returns the clip_id used."""
        if not self.enabled:
            return None
        with self._lock:
            cid = self._clip_id
            self._clip_id += 1
            set_speed = self._set_speed
        if self._dump_dir is not None and clip is not None \
                and cid % self._dump_every == 0:
            np.save(os.path.join(self._dump_dir, f"clip_{cid:06d}.npy"),
                    clip.astype(np.float32))
        if self._fp is not None:
            self._fp.write(json.dumps({
                "type": "infer", "ts": round(time.time(), 4), "clip_id": cid,
                "provider": self._provider,
                "raw_kmh": round(raw_kmh, 4), "speed_kmh": speed_kmh,
                "infer_ms": round(infer_ms, 3), "mean_mov": round(mean_mov, 6),
                "set_kmh": set_speed,
            }) + "\n")
        return cid

    def close(self):
        if self._fp is not None:
            self._fp.close()


class RampGuide:
    """Optional on-screen guided speed ramp. When active the web page shows a big
    'NOW x km/h' banner with a countdown, so the operator just follows along on
    the treadmill instead of touching a terminal. start() walks start→stop in
    `step` increments, holding each `hold` seconds, and stamps each target into
    the bench log via set_speed_cb so the labels line up with the recording."""

    def __init__(self, set_speed_cb, hold=10.0, start=0.0, stop=8.0, step=0.5,
                 countdown=5.0):
        self._cb = set_speed_cb
        self._hold = hold
        self._countdown = countdown
        self._speeds = []
        v = start
        while v <= stop + 1e-6:
            self._speeds.append(round(v, 1))
            v += step
        self._lock = threading.Lock()
        self._state = "idle"      # idle | countdown | running | done
        self._now = None
        self._next = self._speeds[0] if self._speeds else None
        self._remaining = 0
        self._thread = None

    def start(self):
        with self._lock:
            if self._state in ("countdown", "running"):
                return False
            self._state = "countdown"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def _set(self, **kw):
        with self._lock:
            for k, val in kw.items():
                setattr(self, "_" + k, val)

    def _run(self):
        for c in range(int(self._countdown), 0, -1):
            self._set(state="countdown", remaining=c, now=None,
                      next=self._speeds[0] if self._speeds else None)
            time.sleep(1)
        for i, s in enumerate(self._speeds):
            nxt = self._speeds[i + 1] if i + 1 < len(self._speeds) else None
            if self._cb:
                self._cb(s)
            log.info("ramp step: %.1f km/h", s)
            for r in range(int(self._hold), 0, -1):
                self._set(state="running", now=s, next=nxt, remaining=r)
                time.sleep(1)
        self._set(state="done", now=None, next=None, remaining=0)
        log.info("ramp complete")

    def status(self):
        with self._lock:
            return {"state": self._state, "now": self._now, "next": self._next,
                    "remaining": self._remaining, "steps": len(self._speeds),
                    "hold": self._hold}


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
                 clip=CNN_CLIP, smooth=CNN_SMOOTH, bench=None):
        self._n = n_frames
        self._clip = clip
        self._smooth = smooth
        self._span   = CNN_CLIP_SPAN_S   # seconds — time-based, FPS-independent
        self._buf   = collections.deque(maxlen=512)
        self._preds = collections.deque(maxlen=smooth)
        self._speed = None
        self._mean_mov = 0.0
        self._bench = bench              # optional BenchLogger (opt-in)
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
            gated = mean_mov < 0.006   # stopped/near-stopped belt
            bench_on = self._bench is not None and self._bench.enabled
            # Gate at 0.006 so a stopped belt pins the *displayed* speed to 0.0
            # immediately. Normally we skip inference entirely when gated (saves
            # CPU on a dead belt). But in bench mode we still run + log every clip
            # — the comparison must include the stopped/slow clips, and the gate
            # would otherwise silently drop the whole 0 km/h segment.
            if gated and not bench_on:
                with self._lock:
                    self._preds.clear()
                    self._speed = 0.0
                time.sleep(0.1)
                continue
            t_infer = time.perf_counter()
            raw = float(self._sess.run(["speed_kmh"], {"frames": clip})[0].flat[0])
            infer_ms = (time.perf_counter() - t_infer) * 1000.0
            raw = (raw - CNN_CAL_OFFSET) / CNN_CAL_SLOPE   # affine live calibration
            raw = max(0.0, raw)
            with self._lock:
                if gated:
                    # Preserve the live UI behaviour: gated belt shows 0.0.
                    self._preds.clear()
                    self._speed = 0.0
                else:
                    self._preds.append(raw)
                    self._speed = round(sum(self._preds) / len(self._preds), 1)
                speed_now = self._speed
            if bench_on:
                # Log the RAW prediction (pre-smoothing/pre-gate) + latency, and
                # dump the exact tensor we fed the model so the NPU can replay it.
                self._bench.record(raw, speed_now, infer_ms, mean_mov, clip=clip)
            if gated:
                time.sleep(0.1)

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
_bench = None
_ramp = None


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
  .bench { margin-top:14px; }
  .bench label { font-size:12px; color:var(--dim); display:block; margin-bottom:6px; }
  .bench .row { display:flex; gap:8px; align-items:center; justify-content:center; }
  .bench input { width:90px; font-size:20px; text-align:center; padding:6px 8px;
      background:var(--bg2); color:var(--fg); border:1px solid var(--line);
      border-radius:8px; font-variant-numeric:tabular-nums; }
  .bench button { font-size:14px; padding:7px 14px; border-radius:8px; cursor:pointer;
      background:var(--accent2); color:#fff; border:1px solid var(--accent);
      font-weight:600; }
  .bench .hint { font-size:11px; color:var(--dim); margin-top:8px; }
  .bench .ack { font-size:12px; color:var(--accent); margin-top:6px; min-height:16px; }
  /* guided ramp banner */
  .ramp { text-align:center; }
  .ramp .big { font-size:64px; font-weight:800; line-height:1; letter-spacing:-2px;
      font-variant-numeric:tabular-nums;
      background:linear-gradient(180deg,#ffd27a,#f0883e);
      -webkit-background-clip:text; background-clip:text; color:transparent; }
  .ramp .big.stopped { background:linear-gradient(180deg,#8b949e,#6e7681);
      -webkit-background-clip:text; background-clip:text; }
  .ramp .sub { font-size:15px; color:var(--fg); margin-top:8px; min-height:20px; }
  .ramp .nextup { font-size:13px; color:var(--dim); margin-top:4px; min-height:18px; }
  .ramp .count { font-size:13px; color:var(--cyan); font-weight:700; }
  .ramp button { font-size:16px; padding:10px 22px; margin-top:14px; border-radius:10px;
      cursor:pointer; background:var(--accent2); color:#fff; border:1px solid var(--accent);
      font-weight:700; }
  .ramp button:disabled { opacity:.5; cursor:default; }
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
    __RAMP_CARD__
    __BENCH_CARD__
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
__BENCH_JS__
__RAMP_JS__
</script>
</body>
</html>
"""

# Injected into the right column only when a bench recording is active
# (--log/--dump-clips). Lets the operator stamp the panel speed into the log so
# accuracy can later be reported per speed band. Empty otherwise.
BENCH_CARD_HTML = """
    <div class="card bench">
      <label>&#128207; Panel speed (ground truth)</label>
      <div class="row">
        <input id="setSpd" type="number" step="0.5" min="0" max="12" value="0">
        <button onclick="stampSpeed()">Set label</button>
      </div>
      <div class="ack" id="setAck"></div>
      <div class="hint">Type the speed shown on the treadmill panel, press
        <b>Set label</b> at each step of the 0&rarr;8 km/h ramp.</div>
    </div>"""

BENCH_JS = """
async function stampSpeed(){
  const v = parseFloat(document.getElementById('setSpd').value);
  try {
    await fetch('/set_speed?kmh=' + encodeURIComponent(v));
    document.getElementById('setAck').textContent = 'labelled ' + v.toFixed(1) + ' km/h';
  } catch(e){ document.getElementById('setAck').textContent = 'error'; }
}
"""

# Injected only when a guided ramp is active (--ramp). Turns the page into a
# hands-free coach: press Start once, then follow the big NOW banner while the
# server stamps each panel-speed label into the bench log automatically.
RAMP_CARD_HTML = """
    <div class="card ramp">
      <h2 style="justify-content:center">Guided ramp</h2>
      <div class="big stopped" id="rampNow">--</div>
      <div class="sub" id="rampSub">Press Start when you're on the belt, holding the rails.</div>
      <div class="nextup" id="rampNext"></div>
      <button id="rampBtn" onclick="startRamp()">Start ramp</button>
    </div>"""

RAMP_JS = """
async function startRamp(){
  document.getElementById('rampBtn').disabled = true;
  try { await fetch('/ramp_start'); } catch(e){}
}
async function rampPoll(){
  try {
    const s = await (await fetch('/ramp_status')).json();
    const now = document.getElementById('rampNow');
    const sub = document.getElementById('rampSub');
    const nxt = document.getElementById('rampNext');
    const btn = document.getElementById('rampBtn');
    if (s.state === 'idle'){
      now.textContent = '--'; now.className = 'big stopped';
      sub.textContent = 'Press Start when you are on the belt, holding the rails.';
      nxt.textContent = '';
    } else if (s.state === 'countdown'){
      btn.disabled = true;
      now.textContent = s.remaining; now.className = 'big';
      sub.innerHTML = '<span class="count">Get ready — starting…</span>';
      nxt.textContent = 'first step: ' + (s.next!=null? s.next.toFixed(1)+' km/h':'');
    } else if (s.state === 'running'){
      btn.disabled = true;
      now.textContent = (s.now!=null? s.now.toFixed(1):'--');
      now.className = (s.now===0.0? 'big stopped':'big');
      sub.innerHTML = 'Set the panel to <b>' + (s.now!=null?s.now.toFixed(1):'--') +
                      ' km/h</b> &nbsp;·&nbsp; <span class="count">' + s.remaining + 's</span>';
      nxt.textContent = (s.next!=null? 'next: ' + s.next.toFixed(1) + ' km/h' : 'last step');
    } else if (s.state === 'done'){
      now.textContent = '✓'; now.className = 'big';
      sub.textContent = 'Ramp complete — stop the belt. Recording can be closed.';
      nxt.textContent = ''; btn.disabled = true;
    }
  } catch(e){}
}
setInterval(rampPoll, 500);
rampPoll();
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
            bench_on = _bench is not None and _bench.enabled
            ramp_on = _ramp is not None
            html = INDEX_HTML.replace(
                "__RAMP_CARD__", RAMP_CARD_HTML if ramp_on else "").replace(
                "__BENCH_CARD__", BENCH_CARD_HTML if (bench_on and not ramp_on) else "").replace(
                "__BENCH_JS__", BENCH_JS if (bench_on and not ramp_on) else "").replace(
                "__RAMP_JS__", RAMP_JS if ramp_on else "")
            self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/stream":
            self._stream_mjpeg()
        elif path == "/ai_speed":
            spd = _cnn.latest_speed() if _cnn is not None else None
            self._send_json({"speed_kmh": spd,
                             "ready": _cnn is not None and _cnn.ready})
        elif path == "/ramp_start":
            if _ramp is not None:
                ok = _ramp.start()
                self._send_json({"ok": ok})
            else:
                self.send_error(404)
        elif path == "/ramp_status":
            if _ramp is not None:
                self._send_json(_ramp.status())
            else:
                self._send_json({"state": "idle"})
        elif path == "/set_speed":
            qs = parse_qs(urlparse(self.path).query)
            try:
                kmh = float(qs.get("kmh", ["0"])[0])
            except ValueError:
                self.send_error(400); return
            if _bench is not None:
                _bench.set_speed(kmh)
                log.info("panel speed labelled: %.1f km/h", kmh)
            self._send_json({"ok": True, "set_kmh": kmh})
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
    # ── CPU-vs-NPU benchmark mode (opt-in; off by default) ──────────────────
    bench = parser.add_argument_group("benchmark / comparison (opt-in)")
    bench.add_argument("--log", metavar="FILE", default=None,
                       help="append a per-inference JSONL bench log (raw_kmh, "
                            "infer_ms, mean_mov, set_kmh). Enables the panel-speed "
                            "label button in the UI.")
    bench.add_argument("--dump-clips", metavar="DIR", default=None,
                       help="save the exact (1,3,8,112,112) model-input tensors as "
                            "clip_NNNNNN.npy for offline NPU replay")
    bench.add_argument("--dump-every", type=int, default=1, metavar="N",
                       help="dump only every Nth clip (default: %(default)s)")
    bench.add_argument("--provider", default="cpu", choices=["cpu", "qnn"],
                       help="engine label stamped into the log (default: %(default)s)")
    bench.add_argument("--ramp", action="store_true",
                       help="show an on-screen guided 0→8 km/h ramp so the operator "
                            "just follows the page (auto-stamps each panel label)")
    bench.add_argument("--ramp-hold", type=float, default=10.0, metavar="S",
                       help="seconds to hold each ramp step (default: %(default)s)")
    args = parser.parse_args()

    global _cam, _cnn, _bench, _ramp
    _bench = BenchLogger(log_path=args.log, dump_dir=args.dump_clips,
                         dump_every=args.dump_every, provider=args.provider,
                         model_path=args.cnn_model)
    if args.ramp:
        _ramp = RampGuide(set_speed_cb=_bench.set_speed, hold=args.ramp_hold)
        log.info("guided ramp active — %d steps, %.0fs each",
                 len(_ramp._speeds), args.ramp_hold)
    _cnn = CnnSpeedEstimator(model_path=args.cnn_model, bench=_bench)
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
        if _bench is not None:
            _bench.close()
        server.server_close()


if __name__ == "__main__":
    main()
