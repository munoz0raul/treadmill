#!/usr/bin/env python3
"""
acquire.py — record a labeled training session for the vision treadmill.

The treadmill has no sensors of its own, so we make our own ground truth: a side
camera films the belt while YOU tell the app what speed it is set to. Every time
you change the speed on the treadmill you also tap the matching button in this
web UI, and the app writes that value — with a timestamp — next to the video. The
dataset therefore assumes the value you picked is the true belt speed from that
moment until the next change.

    treadmill (dumb) ──▶ side camera ──▶ acquire.py ──▶ ~/dataset/session_<ts>/
                                              ▲
                              you tap the current speed here

What a session directory contains:
  side.mp4            raw camera video (no overlays — a clean training signal)
  speed_manual.jsonl  one row per speed change (+ on start/stop):
                        {"ts": <epoch>, "speed_kmh": <value>}
  manifest.json       session name, start/end time, measured fps, note

This is deliberately simple: no Bluetooth, no second machine, no network calls to
anything. Point it at a camera, open the page, walk, label. `preprocess.py` turns
these sessions into the clip cache the model trains on.
"""

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

CAM_WIDTH  = 1280
CAM_HEIGHT = 720
CAM_FPS    = 30

# Manual speed presets shown as buttons in the UI: Stopped, then 0.5 km/h steps
# up to 8.0 km/h (a comfortable walk→brisk-walk range for a home treadmill). The
# value you tap becomes the label written to speed_manual.jsonl.
SPEED_PRESETS = [round(0.5 * i, 1) for i in range(0, 17)]   # 0.0, 0.5, … 8.0


class ManualSpeed:
    """The current hand-set belt speed (km/h), thread-safe. Whatever the operator
    last tapped in the UI is the label we record."""

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


class SessionRecorder:
    """Owns one recording session: the raw-video writer plus the label sidecar.

    A speed row is appended whenever the operator changes speed and on
    start/stop, so the timeline always has an explicit value at both ends. The
    label file is a step function — a set speed holds until the next row.
    """

    def __init__(self, dataset_dir):
        self._dataset_dir = os.path.expanduser(dataset_dir)
        self._lock = threading.Lock()
        self._dir = None
        self._writer = None
        self._label_fp = None
        self._name = None
        self._frames = 0
        self._t0 = 0.0
        self._fps_hint = CAM_FPS

    @property
    def active(self):
        with self._lock:
            return self._writer is not None

    def start(self, speed_kmh, fps_hint=None):
        with self._lock:
            if self._writer is not None:
                return self._status_locked()
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._name = f"session_{stamp}"
            self._dir = os.path.join(self._dataset_dir, self._name)
            os.makedirs(self._dir, exist_ok=True)

            fps = float(fps_hint) if fps_hint else CAM_FPS
            fps = max(5.0, min(60.0, fps))
            self._fps_hint = fps
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            path = os.path.join(self._dir, "side.mp4")
            writer = cv2.VideoWriter(path, fourcc, fps, (CAM_WIDTH, CAM_HEIGHT))
            if not writer.isOpened():
                return {"ok": False, "error": "cannot open video writer"}
            self._writer = writer
            self._frames = 0
            self._t0 = time.time()

            # Open the label sidecar and stamp the starting speed.
            self._label_fp = open(os.path.join(self._dir, "speed_manual.jsonl"), "w")
            self._write_label_locked(speed_kmh)
            return self._status_locked()

    def write_frame(self, frame_bgr):
        """Append one RAW frame (called from the camera thread). Non-blocking-ish;
        guarded by the same lock as start/stop so a stop can't race a write."""
        with self._lock:
            if self._writer is None:
                return
            try:
                self._writer.write(frame_bgr)
                self._frames += 1
            except Exception:
                pass

    def note_speed(self, speed_kmh):
        """Record a speed change into the label timeline (no-op if not recording)."""
        with self._lock:
            if self._label_fp is None:
                return
            self._write_label_locked(speed_kmh)

    def _write_label_locked(self, speed_kmh):
        row = {"ts": time.time(), "speed_kmh": round(float(speed_kmh), 1)}
        self._label_fp.write(json.dumps(row) + "\n")
        self._label_fp.flush()

    def stop(self, speed_kmh):
        with self._lock:
            if self._writer is None:
                return {"ok": True, "recording": False}
            # Final label row so the timeline is closed at the last known speed.
            self._write_label_locked(speed_kmh)
            dur = time.time() - self._t0
            measured_fps = (self._frames / dur) if dur > 0 else self._fps_hint
            try:
                self._writer.release()
            except Exception:
                pass
            try:
                self._label_fp.close()
            except Exception:
                pass

            manifest = {
                "name": self._name,
                "start_ts": self._t0,
                "end_ts": time.time(),
                "seconds": round(dur, 1),
                "frames": self._frames,
                "writer_fps": self._fps_hint,
                "measured_fps": round(measured_fps, 2),
                "note": "Raw side video. Labels in speed_manual.jsonl are the "
                        "hand-set belt speed (step function).",
            }
            with open(os.path.join(self._dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

            result = {"ok": True, "recording": False, "session": self._name,
                      "frames": self._frames, "seconds": round(dur, 1),
                      "measured_fps": round(measured_fps, 2)}
            self._writer = None
            self._label_fp = None
            self._dir = None
            return result

    def _status_locked(self):
        rec = self._writer is not None
        dur = (time.time() - self._t0) if rec else 0.0
        return {"ok": True, "recording": rec, "session": self._name if rec else None,
                "frames": self._frames if rec else 0, "seconds": round(dur, 1)}

    def status(self):
        with self._lock:
            return self._status_locked()


class CameraThread(threading.Thread):
    """Grab frames from the USB camera, hand each RAW frame to the recorder, and
    keep the latest JPEG (with a light on-screen overlay for the operator) for the
    MJPEG preview. The recorded video is raw; only the live preview is annotated."""

    def __init__(self, camera_dev, recorder, manual):
        super().__init__(daemon=True)
        self._dev = camera_dev
        self._rec = recorder
        self._manual = manual
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
            print(f"ERROR: cannot open camera {self._dev}")
            return
        print(f"Camera {self._dev} "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
              f"{int(cap.get(cv2.CAP_PROP_FPS))} fps")

        fps_counter, t0, fps = 0, time.time(), 0.0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            ts = time.time()

            # Record the RAW frame first — before we draw anything on the preview.
            self._rec.write_frame(frame)

            fps_counter += 1
            elapsed = ts - t0
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter, t0 = 0, ts

            # Preview overlay only (never recorded): current label + fps + REC dot.
            preview = frame.copy()
            spd = self._manual.get()
            cv2.putText(preview, f"label {spd:.1f} km/h", (12, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(preview, f"{fps:.0f} fps", (12, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self._rec.active:
                cv2.circle(preview, (CAM_WIDTH - 40, 40), 12, (0, 0, 255), -1)
                cv2.putText(preview, "REC", (CAM_WIDTH - 120, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            _, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self._lock:
                self._jpeg = jpeg.tobytes()
                self._fps = fps
        cap.release()
        # Flush any in-progress recording so the file is playable.
        if self._rec.active:
            self._rec.stop(self._manual.get())

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
_rec = None
_manual = None


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Treadmill — data acquisition</title>
<style>
  :root { --bg:#0b0f14; --card:#161b22; --card2:#1c232d; --line:#2b333d;
          --fg:#e6edf3; --accent:#3fb950; --accent2:#2ea043; --cyan:#39c5cf;
          --red:#f85149; --dim:#8b949e; --radius:14px;
          --shadow:0 1px 3px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25); }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body { margin:0; background:var(--bg); color:var(--fg); height:100vh;
         overflow:hidden; display:flex; flex-direction:column;
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  header { display:flex; align-items:center; gap:12px; padding:12px 22px; flex:none;
           border-bottom:1px solid var(--line); background:rgba(13,17,23,.6); }
  header .title { font-size:17px; font-weight:700; }
  header .subtitle { font-size:12px; color:var(--dim); }
  header .spacer { flex:1; }
  .badge { font-size:11px; font-weight:700; padding:4px 10px; border-radius:999px;
           text-transform:uppercase; background:var(--card2); color:var(--dim);
           border:1px solid var(--line); }
  .badge.rec { background:rgba(248,81,73,.15); color:#ff7b72;
               border-color:rgba(248,81,73,.4); }
  .wrap { flex:1 1 auto; min-height:0; display:flex; gap:18px; padding:18px;
          max-width:1280px; width:100%; margin:0 auto; }
  .col { display:flex; flex-direction:column; min-height:0; min-width:0; }
  .col.left { flex:1.9; }
  .col.right { flex:1; min-width:320px; max-width:420px; }
  .card { background:linear-gradient(180deg, var(--card2), var(--card));
          border:1px solid var(--line); border-radius:var(--radius);
          padding:16px; box-shadow:var(--shadow); }
  .col > .card + .card { margin-top:16px; }
  #camCard { flex:1 1 auto; min-height:0; display:flex; flex-direction:column; }
  .cam-frame { flex:1 1 auto; min-height:0; display:flex; border-radius:10px;
       overflow:hidden; background:#000; border:1px solid var(--line); }
  .cam { width:100%; height:100%; object-fit:contain; display:block; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em;
       color:var(--dim); margin:0 0 10px; font-weight:700; }
  .bigset { font-size:52px; font-weight:800; color:var(--cyan);
            font-variant-numeric:tabular-nums; text-align:center; }
  .unit { font-size:20px; color:var(--dim); font-weight:600; }
  .presets { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
  .presets button { padding:10px 0; flex:1 1 22%; background:var(--card2);
       color:var(--fg); border:1px solid var(--line); border-radius:8px;
       font-weight:700; cursor:pointer; font-size:15px; }
  .presets button:hover { filter:brightness(1.15); }
  .presets button.active { background:linear-gradient(180deg,var(--cyan),#2a9aa2);
       color:#04252a; border-color:transparent; }
  .btn { width:100%; padding:14px; border-radius:10px; border:1px solid var(--line);
       font-size:15px; font-weight:800; cursor:pointer; background:var(--card2);
       color:var(--fg); margin-top:12px; }
  .btn-rec { background:linear-gradient(180deg,var(--accent),var(--accent2));
       color:#03260c; border-color:transparent; }
  .btn-rec.recording { background:linear-gradient(180deg,#ff6b63,var(--red));
       color:#2b0b09; }
  .status { margin-top:14px; font-size:14px; line-height:1.7; color:var(--dim); }
  .status b { color:var(--fg); }
  .hint { margin-top:14px; font-size:12.5px; line-height:1.6; color:var(--dim);
       border-top:1px solid var(--line); padding-top:12px; }
</style>
</head>
<body>
<header>
  <div>
    <div class="title">Treadmill · data acquisition</div>
    <div class="subtitle">Side camera + manual speed labels</div>
  </div>
  <span class="spacer"></span>
  <span class="badge" id="topBadge">Idle</span>
</header>
<div class="wrap">
  <div class="col left">
    <div class="card" id="camCard">
      <h2>&#128247; Camera preview</h2>
      <div class="cam-frame"><img class="cam" src="/stream" alt="camera stream"></div>
    </div>
  </div>
  <div class="col right">
    <div class="card">
      <h2>Current belt speed (label)</h2>
      <div><span class="bigset" id="setVal">0.0</span> <span class="unit">km/h</span></div>
      <div class="presets" id="presets"></div>
      <button class="btn btn-rec" id="btnRec" onclick="toggleRec()">&#9210; START RECORDING</button>
      <div class="status" id="recStatus">Not recording.</div>
      <div class="hint">
        Set the speed on the treadmill, then tap the matching button here so the
        label timeline stays in sync. Hold each speed for a while before changing.
      </div>
    </div>
  </div>
</div>
<script>
async function jget(u){ const r=await fetch(u); return r.json(); }
async function jpost(u,b){ const r=await fetch(u,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
    return r.json(); }

let PRESETS = [];
let CURRENT = 0.0;
let RECORDING = false;

function buildPresets(){
  const box = document.getElementById('presets');
  box.innerHTML = '';
  PRESETS.forEach(v => {
    const b = document.createElement('button');
    b.textContent = (v === 0) ? 'Stopped' : v.toFixed(1);
    b.dataset.v = v;
    b.onclick = () => setSpeed(v);
    box.appendChild(b);
  });
}
function markActive(){
  document.querySelectorAll('#presets button').forEach(b => {
    b.classList.toggle('active', Number(b.dataset.v) === CURRENT);
  });
}
async function setSpeed(v){
  CURRENT = Number(v);
  document.getElementById('setVal').textContent = CURRENT.toFixed(1);
  markActive();
  await jpost('/speed/set', {speed_kmh: CURRENT});
}
async function toggleRec(){
  await jpost(RECORDING ? '/rec/stop' : '/rec/start', {});
  refresh();
}
async function refresh(){
  try {
    const r = await jget('/rec/status');
    RECORDING = !!r.recording;
    const btn = document.getElementById('btnRec');
    const st = document.getElementById('recStatus');
    const top = document.getElementById('topBadge');
    if (RECORDING) {
      btn.classList.add('recording');
      btn.innerHTML = '&#9209; STOP RECORDING';
      top.className = 'badge rec'; top.textContent = 'REC';
      st.innerHTML = 'Recording <b>' + (r.session||'') + '</b> — ' +
        (r.seconds||0).toFixed(0) + 's, ' + (r.frames||0) + ' frames';
    } else {
      btn.classList.remove('recording');
      btn.innerHTML = '&#9210; START RECORDING';
      top.className = 'badge'; top.textContent = 'Idle';
      st.innerHTML = r.session ? ('Saved <b>' + r.session + '</b>') : 'Not recording.';
    }
  } catch(e){}
}
async function init(){
  const s = await jget('/speed');
  PRESETS = s.presets || [];
  CURRENT = s.speed_kmh || 0.0;
  buildPresets();
  document.getElementById('setVal').textContent = CURRENT.toFixed(1);
  markActive();
  refresh();
}
setInterval(refresh, 700);
init();
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
        elif path == "/speed":
            self._send_json({"speed_kmh": round(_manual.get(), 1),
                             "presets": SPEED_PRESETS})
        elif path == "/rec/status":
            self._send_json(_rec.status())
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        data = self._read_json()
        if path == "/speed/set":
            v = _manual.set(data.get("speed_kmh", 0.0))
            _rec.note_speed(v)   # record the change into the label timeline
            self._send_json({"ok": True, "speed_kmh": round(v, 1)})
        elif path == "/rec/start":
            fps = _cam.get_fps() if _cam is not None else None
            self._send_json(_rec.start(_manual.get(), fps_hint=fps))
        elif path == "/rec/stop":
            self._send_json(_rec.stop(_manual.get()))
        else:
            self.send_error(404)


def main():
    parser = argparse.ArgumentParser(
        description="Record a labeled treadmill session (side camera + manual speed)")
    parser.add_argument("--camera", default="/dev/video0",
                        help="V4L2 device or index (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Web UI port (default: %(default)s)")
    parser.add_argument("--dataset", default="~/dataset",
                        help="Where session_<ts>/ folders are written "
                             "(default: %(default)s)")
    args = parser.parse_args()

    global _cam, _rec, _manual
    _manual = ManualSpeed(0.0)
    _rec = SessionRecorder(args.dataset)
    cam_dev = int(args.camera) if str(args.camera).lstrip("-").isdigit() else args.camera
    _cam = CameraThread(cam_dev, _rec, _manual)
    _cam.start()

    print("Waiting for first camera frame ...")
    for _ in range(50):
        if _cam.get_frame():
            break
        time.sleep(0.1)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Web UI on http://0.0.0.0:{args.port}  "
          f"(sessions → {os.path.expanduser(args.dataset)})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if _rec.active:
            _rec.stop(_manual.get())
        _cam.stop()
        server.server_close()


if __name__ == "__main__":
    main()
