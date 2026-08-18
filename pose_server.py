#!/usr/bin/env python3
# Treadmill pose detection + data collection server
# IQ-8275 NPU (HTP) + EMEET C960 | records labeled keypoint sessions for LSTM training

import json
import os
import signal
import threading
import time
import argparse
import logging
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import collections
import cv2
import numpy as np
import ai_edge_litert.interpreter as litert
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH    = "/etc/models/hrnet_pose_quantized.tflite"
LABELS_PATH   = "/etc/labels/hrnet_pose.json"
SETTINGS_PATH = "/etc/labels/hrnet_settings.json"
CAMERA_DEV    = "/dev/video0"
DATASET_DIR   = os.path.expanduser("~/dataset")   # sessions dir (was /home/weston — now user-relative)

# LSTM speed estimator (legacy — replaced by CnnSpeedEstimator)
LSTM_MODEL    = os.path.expanduser("~/models/speed_lstm.onnx")
LSTM_WINDOW   = 30   # frames
LSTM_KP       = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
LSTM_SMOOTH   = 5    # rolling average over N predictions

# CNN speed estimator (mc3_18 / Kinetics-400 — RGB clip regressor)
CNN_MODEL     = os.path.expanduser("~/models/speed_cnn.onnx")
CNN_CLIP      = 112        # square resize target (matches preprocess.py)
CNN_N_FRAMES  = 8          # frames per clip
CNN_FRAME_STRIDE = 4       # legacy — kept only for set_fps()/​/cnn/tune back-compat
CNN_CLIP_SPAN_S = 1.0      # real-time span of one clip, in SECONDS. Must equal
                           #   CLIP_SPAN_S in preprocess.py — this is what makes
                           #   train and serve cover the same real motion regardless
                           #   of recording FPS (fixes the bimodal 24/29.5 fps skew).
CNN_TRAIN_FPS = 24.4       # legacy band-aid, now UNUSED for span (span is time-based
                           #   via CNN_CLIP_SPAN_S). /cnn/tune still overrides span
                           #   live if you need to sweep, but a correctly retrained
                           #   model needs no fps tuning at all.
CNN_CROP_X0   = 240        # central horizontal crop of 1280-wide frame …
CNN_CROP_X1   = 1040       # … keep x∈[240,1040) — drops side clutter, keeps body
CNN_SMOOTH    = 6          # rolling average over N predictions (~3s window @2Hz):
                           #   halves the settling lag after a speed change vs the
                           #   old 12 (~6s) while still filtering per-frame noise.
POSE_INTERVAL = 0.5        # min seconds between HRNet runs on the side camera
                           #   (overlay only — must not starve CNN frame capture)
# Live calibration (2026-08-17): the raw mc3_18 output tracks true belt speed
# with a clean affine bias, measured against the panel:
#   raw = 0.9*true + 1.5   (pairs 3→4.2, 4→5.1, 5→6.0, 6→6.9, 7→7.8)
# Invert to recover true speed:  true = (raw - 1.5) / 0.9
CNN_CAL_SLOPE = 1.0
CNN_CAL_OFFSET = 0.0

# Guided panel-driven session. The SCREEN tells the user what speed to set on
# the treadmill's own PANEL — BLE on this ZiYou unit is READ-ONLY for speed
# (confirmed 2026-08-14: even the official app only reads), so the belt CANNOT
# be driven over Bluetooth. BLE is logged purely as the ground-truth label.
# Schedule: 0.0 → 8.0 km/h in 0.5 steps, 30 s each (17 steps, ~8.5 min + warmup).
# Starts at 0.0 (belt STOPPED) so the model learns "stopped" (label 0) — the
# stopped step is logged via AutoSession's override (BLE emits nothing at rest).
# The Mac BLE bridge base URL is configurable via the MAC_BLE_URL env var (set it
# to the LAN address of the machine running mac_ble_server.py, e.g. a Mac).
AUTO_RAMP_URL      = os.environ.get("MAC_BLE_URL", "http://mac-ble-host.local:8765")   # mac_ble_server base URL (speed readback only)
AUTO_START_KMH     = 0.0
AUTO_END_KMH       = 8.0
AUTO_STEP_KMH      = 0.5
AUTO_DWELL_S       = 30     # seconds to hold each speed step (more samples/step → balance)
AUTO_WARMUP_S      = 8      # "get ready" countdown before the first cue (user starts the belt on the panel)
CAM_WIDTH     = 1280
CAM_HEIGHT    = 720
CAM_FPS       = 30
MODEL_W       = 192
MODEL_H       = 256
HEATMAP_W     = 48
HEATMAP_H     = 64

SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SKELETON_COLOR = (0, 200, 255)
REC_COLOR      = (0, 0, 220)


# ---------------------------------------------------------------------------
# HTTP speed logger — polls mac_ble_server.py, drop-in for BleSpeedLogger
# ---------------------------------------------------------------------------

class HttpSpeedLogger:
    """Polls GET <url> (mac_ble_server.py) every second and exposes the same
    public API as BleSpeedLogger so all Recorder / Handler call sites are unchanged."""

    POLL_INTERVAL = 1.0

    def __init__(self, url: str):
        self._url  = url
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._latest      = None
        self._latest_ts   = 0.0
        self._sample_count = 0
        self._connected   = False
        self._override    = None   # when set, log this speed even if BLE is silent (stopped-belt step)
        self._file        = None
        self._log_path    = None
        self._thread      = None
        self._stop_evt    = threading.Event()

    # ── same public API as BleSpeedLogger ────────────────────────────────────

    def set_log_path(self, log_path: str):
        self._log_path = log_path

    def attach_log(self, log_path: str):
        with self._file_lock:
            if self._file:
                self._file.close()
            self._log_path = log_path
            self._file = open(log_path, "w", buffering=1)
        log.info("Speed log attached: %s", log_path)

    def detach_log(self):
        with self._file_lock:
            if self._file:
                self._file.close()
                self._file = None
        log.info("Speed log detached (HTTP poller keeps running)")

    def latest_speed(self):
        with self._lock:
            return self._latest

    def set_override(self, value: float):
        """Force-log `value` km/h while the belt emits no BLE (stopped step).
        Only honoured while connected — keeps the logged 0 auditable."""
        with self._lock:
            self._override = float(value)

    def clear_override(self):
        with self._lock:
            self._override = None

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def status(self) -> dict:
        with self._lock:
            return {
                "connected":       self._connected,
                "latest_speed_kmh": self._latest,
                "latest_ts":       round(self._latest_ts, 4) if self._latest_ts else None,
                "samples":         self._sample_count,
                "belt_started":    False,
                "url":             self._url,
            }

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        with self._lock:
            self._latest = None
            self._latest_ts = 0.0
            self._sample_count = 0
            self._connected = False
        if self._log_path and not self._file:
            with self._file_lock:
                self._file = open(self._log_path, "w", buffering=1)
        self._thread = threading.Thread(target=self._run, daemon=True, name="http-speed")
        self._thread.start()
        log.info("HttpSpeedLogger started (%s)", self._url)

    def stop(self):
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        with self._file_lock:
            if self._file:
                self._file.close()
                self._file = None
        log.info("HttpSpeedLogger stopped (%d samples)", self._sample_count)

    def reset(self):
        log.info("HTTP speed logger reset — reconnecting")
        self.stop()
        self.start()

    def _run(self):
        import urllib.request
        import urllib.error
        while not self._stop_evt.is_set():
            try:
                with urllib.request.urlopen(self._url, timeout=3.0) as resp:
                    data = json.loads(resp.read())
                speed      = data.get("speed_kmh")
                mac_ts     = data.get("ts") or time.time()
                mac_conn   = bool(data.get("connected"))
                # Effective speed to log: real BLE reading, or the override
                # (0 km/h at the stopped step) when the belt is silent but
                # still connected. The override keeps the logged 0 auditable.
                with self._lock:
                    self._connected = mac_conn
                    override = self._override
                    if speed is not None:
                        eff_speed = float(speed)
                    elif override is not None and mac_conn:
                        eff_speed = float(override)
                    else:
                        eff_speed = None
                    if eff_speed is not None:
                        self._latest    = eff_speed
                        self._latest_ts = float(mac_ts)
                        self._sample_count += 1
                    else:
                        self._latest = None
                if eff_speed is not None:
                    with self._file_lock:
                        if self._file:
                            self._file.write(
                                json.dumps({"ts": round(float(mac_ts), 4),
                                            "speed_kmh": eff_speed}) + "\n"
                            )
            except Exception as e:
                log.debug("HTTP speed poll error: %s", e)
                with self._lock:
                    self._connected = False
                    self._latest    = None
            self._stop_evt.wait(self.POLL_INTERVAL)


# ---------------------------------------------------------------------------
# CNN speed estimator (mc3_18 RGB-clip regressor)
# ---------------------------------------------------------------------------

class CnnSpeedEstimator:
    """Buffers cropped RGB frames from the side camera (tagged with wall-clock
    timestamps) and runs speed_cnn.onnx (mc3_18) via onnxruntime.

    Input clip: (1, 3, T, 112, 112) float32 [0..1]. The model bakes in the
    Kinetics normalisation internally, so we feed plain [0..1] RGB. Frames are
    centre-cropped (x∈[CROP_X0,CROP_X1)) then resized to 112×112, matching
    preprocess.py.

    CRITICAL — temporal alignment: training clips span
    (T-1)*FRAME_STRIDE frames at the ~30.8fps recording rate ≈ 0.9s. The live
    capture loop runs far slower and at a variable rate, so we CANNOT subsample
    by buffer index — that would stretch the clip to many seconds and make the
    model wildly overestimate speed. Instead we pick T frames by *timestamp*, so
    the live clip always spans the same ~0.9s of real motion as in training.

    Inference (mc3_18 on CPU, ~1.2s) runs in a background thread so it never
    throttles the camera capture loop; add_frame() is cheap and non-blocking.

    Thread-safe: add_frame() is called from CameraThread; latest_speed() from
    Handler; a private worker thread runs inference.
    """

    def __init__(self, model_path=CNN_MODEL, n_frames=CNN_N_FRAMES,
                 clip=CNN_CLIP, stride=CNN_FRAME_STRIDE, smooth=CNN_SMOOTH):
        self._n = n_frames
        self._clip = clip
        self._smooth = smooth
        self._stride = stride
        # Real-time span the clip must cover, matching training:
        #   (T-1)*FRAME_STRIDE frames at the recording fps.
        # NOTE: _train_fps is tunable at runtime via /cnn/tune?fps=… so we can
        # sweep it live against BLE ground truth without a restart. The dataset's
        # real capture rate is bimodal (~24 and ~29.5 fps); 30.8 was wrong.
        self._train_fps = CNN_TRAIN_FPS
        self._span   = CNN_CLIP_SPAN_S   # seconds — time-based, FPS-independent
        # Ring buffer of (ts, small_rgb). Keep a bit more than one span so the
        # worker always has a full window to sample from.
        self._buf   = collections.deque(maxlen=512)
        self._preds = collections.deque(maxlen=smooth)
        self._speed = None
        self._mean_mov = 0.0
        self._lock  = threading.Lock()
        self._sess  = None
        self._running = False
        self._worker  = None
        # Optional live log of the AI speed during a recording (speed_ai.jsonl),
        # mirroring the BLE logger so a session captures BOTH the real belt speed
        # and the model's live estimate, timestamp-aligned for offline overlay.
        self._ai_file      = None
        self._ai_file_lock = threading.Lock()
        self._ai_log_path  = None

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
        # Centre horizontal crop (scale the constants if frame isn't 1280 wide)
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
        # Need genuine coverage: oldest buffered frame must reach back to t_start.
        if buf[0][0] > t_start + 1e-3:
            return None
        ts_arr = [b[0] for b in buf]
        targets = [t_start + i * self._span / (self._n - 1) for i in range(self._n)]
        picks = []
        j = 0
        for tgt in targets:
            # advance to the buffered frame nearest tgt
            while j + 1 < len(ts_arr) and abs(ts_arr[j + 1] - tgt) <= abs(ts_arr[j] - tgt):
                j += 1
            picks.append(buf[j][1])
        clip = np.stack(picks, axis=0).astype(np.float32) / 255.0   # (T,H,W,3)
        clip = clip.transpose(3, 0, 1, 2)[np.newaxis]               # (1,3,T,H,W)
        return clip

    def set_log_path(self, log_path: str):
        self._ai_log_path = log_path

    def attach_log(self, log_path: str):
        """Open speed_ai.jsonl for this session. Mirrors the BLE logger so the
        Recorder can start/stop AI logging with the same call sites."""
        with self._ai_file_lock:
            if self._ai_file:
                self._ai_file.close()
            self._ai_log_path = log_path
            self._ai_file = open(log_path, "w", buffering=1)
        log.info("AI speed log attached: %s", log_path)

    def detach_log(self):
        with self._ai_file_lock:
            if self._ai_file:
                self._ai_file.close()
                self._ai_file = None
        log.info("AI speed log detached")

    def _infer_loop(self):
        while self._running:
            clip = self._sample_clip()
            if clip is None:
                time.sleep(0.05)
                continue
            # Cheap motion gate: mean abs frame-to-frame diff of the clip
            mean_mov = float(np.abs(np.diff(clip, axis=2)).mean())
            with self._lock:
                self._mean_mov = mean_mov
            # Motion gate: idle camera noise (MJPEG + sensor) sits ~0.0036;
            # walking is ≥0.01. Gate at 0.006 so a stopped belt reads 0.0, not
            # ~1 km/h (the model never saw speeds <0.5 in training so it can't
            # extrapolate to "stopped" — the gate handles that case explicitly).
            if mean_mov < 0.006:
                with self._lock:
                    self._preds.clear()
                    self._speed = 0.0
                self._log_ai(0.0)
                time.sleep(0.1)
                continue
            raw = float(self._sess.run(["speed_kmh"], {"frames": clip})[0].flat[0])
            # Apply the affine live calibration (see CNN_CAL_* constants).
            raw = (raw - CNN_CAL_OFFSET) / CNN_CAL_SLOPE
            raw = max(0.0, raw)
            with self._lock:
                self._preds.append(raw)
                self._speed = round(sum(self._preds) / len(self._preds), 1)
                smoothed = self._speed
            self._log_ai(smoothed)

    def _log_ai(self, speed):
        """Append one {ts, speed_kmh} row to speed_ai.jsonl if a log is attached."""
        with self._ai_file_lock:
            if self._ai_file:
                self._ai_file.write(
                    json.dumps({"ts": round(time.time(), 4),
                                "speed_kmh": float(speed)}) + "\n"
                )

    def stop(self):
        self._running = False

    def set_fps(self, fps):
        """Retune the clip's real-time span live (sweep against BLE truth)."""
        with self._lock:
            self._train_fps = float(fps)
            self._span = (self._n - 1) * self._stride / self._train_fps
            self._preds.clear()
        return {"fps": self._train_fps, "span_s": round(self._span, 3)}

    def latest_speed(self):
        with self._lock:
            return self._speed

    def debug_info(self):
        with self._lock:
            return {"mean_mov": round(self._mean_mov, 5), "speed": self._speed,
                    "buf": len(self._buf), "fps": round(self._train_fps, 2),
                    "span_s": round(self._span, 3)}


# ---------------------------------------------------------------------------
# Speed estimator (LSTM via ONNX runtime)
# ---------------------------------------------------------------------------
class SpeedEstimator:
    def __init__(self, model_path=LSTM_MODEL, window=LSTM_WINDOW, smooth=LSTM_SMOOTH):
        self._window = window
        self._smooth = smooth
        self._buf = []          # sliding window of feature vectors
        self._preds = []        # recent predictions for smoothing
        self._sess = None
        self._fw = 1280
        self._fh = 720
        if not _ORT_AVAILABLE:
            log.warning("onnxruntime not available — speed estimation disabled")
            return
        if not os.path.exists(model_path):
            log.warning("LSTM model not found at %s — speed estimation disabled", model_path)
            return
        self._sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        log.info("LSTM speed estimator loaded (%d-frame window)", window)

    def update(self, keypoints, frame_w, frame_h):
        """Call once per frame with the 17-keypoint list. Returns estimated speed or None."""
        if self._sess is None:
            return None
        self._fw = frame_w
        self._fh = frame_h
        vec = []
        for idx in LSTM_KP:
            kp = keypoints[idx] if idx < len(keypoints) else None
            if kp is not None:
                vec += [kp[0] / frame_w, kp[1] / frame_h, float(kp[2])]
            else:
                vec += [0.0, 0.0, 0.0]
        self._buf.append(vec)
        if len(self._buf) > self._window:
            self._buf.pop(0)
        if len(self._buf) < self._window:
            return None
        x = np.array(self._buf, dtype=np.float32)[np.newaxis]  # [1, W, F]
        pred = float(self._sess.run(None, {"keypoints": x})[0][0])
        pred = max(0.0, pred)
        self._preds.append(pred)
        if len(self._preds) > self._smooth:
            self._preds.pop(0)
        return round(sum(self._preds) / len(self._preds), 1)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class _CamStream:
    """Per-camera recording state: raw video (mp4) + pose keypoints (jsonl)."""
    __slots__ = ("label", "jsonl", "writer", "frames", "vframes", "w", "h", "fps")

    def __init__(self, label):
        self.label  = label
        self.jsonl  = None    # open pose JSONL file
        self.writer = None    # cv2.VideoWriter (opened lazily on first frame)
        self.frames = 0       # pose rows written
        self.vframes = 0      # video frames written
        self.w      = 0
        self.h      = 0
        self.fps    = 0.0


class Recorder:
    """Session recorder: one directory per session holding, per camera, a raw
    .mp4 video and a pose .jsonl, plus a BLE speed log and a manifest.json.

    Frames are labeled with speed offline via timestamp-join against
    speed_ble.jsonl — the label is NOT baked in live, so alignment is auditable.
    """

    def __init__(self, dataset_dir: str):
        os.makedirs(dataset_dir, exist_ok=True)
        self._dir = dataset_dir
        self._lock = threading.Lock()
        self._recording = False
        self._session_name = ""
        self._session_dir = None
        self._start_ts = 0.0
        self._cams = {}          # label -> _CamStream
        self._labels = []        # registered camera labels (declaration order)
        self._ble = None         # BleSpeedLogger (or None)
        self._cnn = None         # CnnSpeedEstimator (or None) — logs speed_ai.jsonl

    def register_camera(self, label: str):
        """Called once per camera at startup so the recorder knows what to open."""
        if label not in self._labels:
            self._labels.append(label)

    def start(self, ble_logger=None, cnn_logger=None) -> str:
        with self._lock:
            if self._recording:
                self._stop_locked()
            self._start_ts = time.time()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._session_name = f"session_{ts}"
            self._session_dir = os.path.join(self._dir, self._session_name)
            os.makedirs(self._session_dir, exist_ok=True)

            self._cams = {}
            for label in self._labels:
                cs = _CamStream(label)
                cs.jsonl = open(os.path.join(self._session_dir, f"{label}.jsonl"),
                                "w", buffering=1)
                self._cams[label] = cs

            # BLE ground-truth speed logger writes speed_ble.jsonl in this
            # session. The connection is kept alive by the server across
            # sessions (preflight), so we just ATTACH a fresh log file here —
            # if already connected, logging starts instantly with no reconnect.
            self._ble = ble_logger
            if self._ble is not None:
                log_path = os.path.join(self._session_dir, "speed_ble.jsonl")
                if self._ble.is_alive:
                    self._ble.attach_log(log_path)
                else:
                    self._ble.set_log_path(log_path)
                    self._ble.start()
                # NOTE: do NOT send 0x07 Start/Resume here. The treadmill streams
                # 0x2ACD automatically once the user presses Start on the panel.
                # Sending 0x07 makes the treadmill treat our BLE client as the
                # session owner — if the connection drops the belt STOPS. Listen-
                # only is safe and confirmed working (see ble_sniff.py test).

            # AI speed logger: the CNN worker is always running; here we just
            # ATTACH a fresh speed_ai.jsonl so this session also captures the
            # model's live estimate alongside the real belt speed.
            self._cnn = cnn_logger
            if self._cnn is not None:
                self._cnn.attach_log(
                    os.path.join(self._session_dir, "speed_ai.jsonl"))

            self._recording = True
            log.info("Recording started: %s (%d cameras)", self._session_name, len(self._cams))
            return self._session_name

    def write_pose(self, label: str, ts: float, keypoints: list, frame_w: int, frame_h: int):
        with self._lock:
            cs = self._cams.get(label)
            if not self._recording or cs is None or cs.jsonl is None:
                return
            row = {
                "ts": round(ts, 4),
                "frame_w": frame_w,
                "frame_h": frame_h,
                "keypoints": [
                    {"x": kp[0], "y": kp[1], "conf": round(kp[2], 3)} if kp else None
                    for kp in keypoints
                ],
            }
            cs.jsonl.write(json.dumps(row) + "\n")
            cs.frames += 1

    def write_frame(self, label: str, frame_bgr, fps_hint: float = None):
        """Append a raw (un-annotated) BGR frame to this camera's mp4.

        The mp4 is opened at a FIXED nominal CAM_FPS — the live fps estimate is
        noisy during warmup and would bake a wrong playback rate into the header.
        The authoritative per-frame timing lives in the .jsonl timestamps, and
        the true measured fps (frames / duration) is written to the manifest at
        stop() so playback can be corrected offline if needed.
        """
        with self._lock:
            cs = self._cams.get(label)
            if not self._recording or cs is None:
                return
            if cs.writer is None:
                h, w = frame_bgr.shape[:2]
                path = os.path.join(self._session_dir, f"{label}.mp4")
                cs.writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*"mp4v"), float(CAM_FPS), (w, h))
                cs.w, cs.h, cs.fps = w, h, float(CAM_FPS)
                log.info("  video writer opened: %s.mp4 %dx%d @ %d fps (nominal)",
                         label, w, h, CAM_FPS)
            cs.writer.write(frame_bgr)
            cs.vframes += 1

    def stop(self) -> dict:
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self) -> dict:
        if not self._recording:
            return {}
        duration = round(time.time() - self._start_ts, 1)

        cams_meta = []
        for label in self._labels:
            cs = self._cams.get(label)
            if cs is None:
                continue
            if cs.jsonl:
                cs.jsonl.close()
            if cs.writer:
                cs.writer.release()
            cams_meta.append({
                "label": label,
                "pose_frames": cs.frames,
                "video_frames": cs.vframes,
                "video_size": [cs.w, cs.h],
                "video_fps_nominal": round(cs.fps, 2),
                "video_fps_measured": round(cs.vframes / duration, 2) if duration > 0 else 0.0,
            })

        ble_status = None
        if self._ble is not None:
            # Grab the status for the manifest, then detach the log but KEEP the
            # BLE connection alive so the preflight check stays green and the
            # next recording starts logging instantly (no reconnect needed).
            ble_status = self._ble.status
            self._ble.detach_log()

        # Close speed_ai.jsonl (the CNN worker keeps running for the live UI).
        ai_logged = self._cnn is not None
        if self._cnn is not None:
            self._cnn.detach_log()

        manifest = {
            "session": self._session_name,
            "start_ts": round(self._start_ts, 4),
            "end_ts": round(self._start_ts + duration, 4),
            "duration_s": duration,
            "cameras": cams_meta,
            "ble": ble_status,
            "ai_speed_logged": ai_logged,
            "clock": "time.time()",
            "note": "real belt speed in speed_ble.jsonl; live model estimate in speed_ai.jsonl; both timestamp-aligned to *.jsonl frames",
        }
        if self._session_dir:
            with open(os.path.join(self._session_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)

        total_frames = sum(c["pose_frames"] for c in cams_meta)
        log.info("Recording stopped: %s  %d frames across %d cams (%.1fs)",
                 self._session_name, total_frames, len(cams_meta), duration)

        info = {"session": self._session_name, "frames": total_frames,
                "duration_s": duration, "cameras": cams_meta}
        self._recording = False
        self._session_name = ""
        self._session_dir = None
        self._cams = {}
        self._ble = None
        return info

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def status(self) -> dict:
        with self._lock:
            if not self._recording:
                return {"recording": False}
            cams = []
            for label in self._labels:
                cs = self._cams.get(label)
                if cs is None:
                    continue
                cams.append({
                    "label": label,
                    "pose_frames": cs.frames,
                    "video_frames": cs.vframes,
                })
            frames = sum(cs.frames for cs in self._cams.values())
            ble_status = self._ble.status if self._ble is not None else None
            ble_speed = self._ble.latest_speed() if self._ble is not None else None
            return {
                "recording": True,
                "frames": frames,
                "duration_s": round(time.time() - self._start_ts, 1),
                "session": self._session_name,
                "ble_speed_kmh": ble_speed,
                "cameras": cams,
                "ble": ble_status,
            }

    def list_sessions(self) -> list:
        sessions = []
        for name in sorted(os.listdir(self._dir), reverse=True):
            sdir = os.path.join(self._dir, name)
            if not (name.startswith("session_") and os.path.isdir(sdir)):
                continue
            manifest_path = os.path.join(sdir, "manifest.json")
            entry = {"session": name, "cameras": [], "duration_s": 0.0,
                     "frames": 0, "speed_range": None}
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path) as f:
                        m = json.load(f)
                    entry["duration_s"] = m.get("duration_s", 0.0)
                    entry["cameras"] = [c["label"] for c in m.get("cameras", [])]
                    entry["frames"] = sum(c.get("pose_frames", 0) for c in m.get("cameras", []))
                except Exception:
                    pass
            # BLE speed range from speed_ble.jsonl if present
            ble_path = os.path.join(sdir, "speed_ble.jsonl")
            if os.path.exists(ble_path):
                try:
                    speeds = []
                    with open(ble_path) as f:
                        for line in f:
                            if line.strip():
                                speeds.append(json.loads(line)["speed_kmh"])
                    if speeds:
                        entry["speed_range"] = [min(speeds), max(speeds)]
                except Exception:
                    pass
            sessions.append(entry)
        return sessions


# ---------------------------------------------------------------------------
# Pose detector
# ---------------------------------------------------------------------------

class PoseDetector:
    def __init__(self, use_npu: bool = True):
        with open(SETTINGS_PATH) as f:
            settings = json.load(f)
        self.conf_thresh = settings.get("confidence", 51.0) / 100.0
        # The litert interpreter is NOT thread-safe. Multiple CameraThreads share
        # one detector, so every invoke() must be serialized behind this lock.
        self._lock = threading.Lock()

        if use_npu:
            log.info("Loading NPU (HTP) delegate ...")
            try:
                delegate = litert.load_delegate(
                    "/usr/lib/libQnnTFLiteDelegate.so", {"backend_type": "htp"}
                )
                self.interp = litert.Interpreter(MODEL_PATH, experimental_delegates=[delegate])
                log.info("NPU delegate loaded")
            except Exception as e:
                log.warning("NPU failed (%s), falling back to CPU", e)
                self.interp = litert.Interpreter(MODEL_PATH)
        else:
            log.info("Using CPU interpreter")
            self.interp = litert.Interpreter(MODEL_PATH)

        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        log.info("Model ready  input %s  output %s", self.inp["shape"], self.out["shape"])

    def _preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        scale = min(MODEL_W / w, MODEL_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (nw, nh))
        canvas = np.full((MODEL_H, MODEL_W, 3), 128, dtype=np.uint8)
        pad_x = (MODEL_W - nw) // 2
        pad_y = (MODEL_H - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[np.newaxis], scale, pad_x, pad_y

    def _decode(self, heatmaps, scale, pad_x, pad_y, fw, fh):
        hm = heatmaps[0].astype(np.float32)
        keypoints = []
        for k in range(17):
            ch = hm[:, :, k]
            conf = ch.max() / 255.0
            if conf < self.conf_thresh:
                keypoints.append(None)
                continue
            iy, ix = np.unravel_index(np.argmax(ch), ch.shape)
            mx = (ix + 0.5) * (MODEL_W / HEATMAP_W)
            my = (iy + 0.5) * (MODEL_H / HEATMAP_H)
            fx = int(np.clip((mx - pad_x) / scale, 0, fw - 1))
            fy = int(np.clip((my - pad_y) / scale, 0, fh - 1))
            keypoints.append((fx, fy, float(conf)))
        return keypoints

    def detect(self, frame_bgr):
        inp_t, scale, pad_x, pad_y = self._preprocess(frame_bgr)
        h, w = frame_bgr.shape[:2]
        with self._lock:
            self.interp.set_tensor(self.inp["index"], inp_t)
            self.interp.invoke()
            # copy() so we don't hold a reference into interpreter-internal memory
            # (litert raises if a tensor view outlives the next invoke()).
            hm = self.interp.get_tensor(self.out["index"]).copy()
        return self._decode(hm, scale, pad_x, pad_y, w, h)

    def draw(self, frame, keypoints, is_recording: bool):
        for a, b in SKELETON:
            if keypoints[a] and keypoints[b]:
                cv2.line(frame, keypoints[a][:2], keypoints[b][:2],
                         SKELETON_COLOR, 2, cv2.LINE_AA)
        for kp in keypoints:
            if kp:
                cv2.circle(frame, kp[:2], 5, (0, 255, 0), -1, cv2.LINE_AA)
        if is_recording:
            cv2.circle(frame, (frame.shape[1] - 30, 30), 12, REC_COLOR, -1, cv2.LINE_AA)
            cv2.putText(frame, "REC", (frame.shape[1] - 70, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, REC_COLOR, 2)


# ---------------------------------------------------------------------------
# Camera thread
# ---------------------------------------------------------------------------

class CameraThread(threading.Thread):
    def __init__(self, detector: PoseDetector, recorder: Recorder,
                 camera_dev: str = CAMERA_DEV, label: str = "front"):
        super().__init__(daemon=True)
        self.detector    = detector
        self.recorder    = recorder
        self.label       = label
        self._camera_dev = camera_dev
        self._lock       = threading.Lock()
        self._jpeg       = b""
        self._keypoints  = []
        self._fps        = 0.0
        self._running    = True

    def run(self):
        cap = cv2.VideoCapture(self._camera_dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        if not cap.isOpened():
            log.error("[%s] Cannot open %s", self.label, self._camera_dev)
            return
        log.info("[%s] Camera %dx%d @ %d fps", self.label,
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(cap.get(cv2.CAP_PROP_FPS)))

        fps_counter, t0 = 0, time.time()
        fps = 0.0
        last_pose = 0.0
        last_keypoints = []
        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            ts = time.time()
            rec = self.recorder.is_recording

            # Save the RAW (un-annotated) frame to video before we draw on it,
            # so the mp4 is pristine and pose can be reprocessed offline forever.
            if rec:
                self.recorder.write_frame(self.label, frame, fps_hint=fps)

            # Feed side camera frames to the CNN speed estimator (every frame).
            # This is cheap (crop+resize+append); inference runs in a background
            # thread inside the estimator, so the capture loop stays fast and the
            # RGB buffer fills at near camera fps — essential for the clip to span
            # the same ~0.9s of real motion the model saw in training.
            if self.label == "side" and _cnn is not None:
                _cnn.add_frame(frame)

            # Pose/HRNet is only used for the live overlay, not for speed and not
            # for the training pipeline (preprocess.py reads only frame timestamps
            # from side.jsonl). It's the slowest step (~0.5s), so throttle it hard
            # while the side camera is driving the CNN, so it can't starve capture.
            now = ts
            run_pose = (self.label != "side") or (now - last_pose >= POSE_INTERVAL)
            if run_pose:
                keypoints = self.detector.detect(frame)
                last_pose = now
                last_keypoints = keypoints
            else:
                keypoints = last_keypoints
            self.detector.draw(frame, keypoints, rec)

            if rec:
                self.recorder.write_pose(self.label, ts, keypoints,
                                         frame.shape[1], frame.shape[0])

            fps_counter += 1
            elapsed = ts - t0
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter, t0 = 0, ts

            cv2.putText(frame, f"[{self.label}] {fps:.1f} fps", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self._lock:
                self._jpeg = jpeg.tobytes()
                self._keypoints = keypoints
                self._fps = fps

        cap.release()

    def get_frame(self):
        with self._lock:
            return self._jpeg

    def get_keypoints(self):
        with self._lock:
            return list(self._keypoints)

    def get_fps(self):
        with self._lock:
            return self._fps

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Auto session — one-button fully autonomous data collection
# ---------------------------------------------------------------------------

class AutoSession:
    """Guided PANEL-DRIVEN capture: start recording → tell the user (big on
    screen) what speed to set on the treadmill's own panel, step by step →
    stop recording.

    This does NOT command the belt. BLE on the ZiYou 250-S is read-only for
    speed (the official app only reads too), so the human sets each speed on
    the panel when cued; BLE logs the true speed as the training label. The
    exposed `status` carries the current cue (`current_kmh`, `step`,
    `total_steps`, `dwell_s`, `step_elapsed`) so the UI can render the big
    "SET PANEL TO X km/h" prompt and a countdown for the current step.
    """

    def __init__(self, recorder, ble_logger, cnn_logger=None,
                 mac_url=AUTO_RAMP_URL,
                 start_kmh=AUTO_START_KMH, end_kmh=AUTO_END_KMH,
                 step_kmh=AUTO_STEP_KMH, dwell_s=AUTO_DWELL_S,
                 warmup_s=AUTO_WARMUP_S):
        self._recorder   = recorder
        self._ble        = ble_logger
        self._cnn        = cnn_logger
        self._mac_url    = mac_url
        self._start_kmh  = start_kmh
        self._end_kmh    = end_kmh
        self._step_kmh   = step_kmh
        self._dwell_s    = dwell_s
        self._warmup_s   = warmup_s
        self._lock       = threading.Lock()
        self._thread     = None
        self._stop_evt   = threading.Event()
        self._state      = {
            "running": False, "phase": "idle",
            "step": 0, "total_steps": 0,
            "current_kmh": None, "next_kmh": None,
            "dwell_s": dwell_s, "step_elapsed": 0,
            "session": None, "error": None,
        }

    def _upd(self, **kw):
        with self._lock:
            self._state.update(kw)

    @property
    def status(self):
        with self._lock:
            return dict(self._state)

    def start(self):
        with self._lock:
            if self._state["running"]:
                return False, "already running"
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="auto-session")
        self._thread.start()
        return True, "started"

    def stop(self):
        self._stop_evt.set()
        return True, "stopping"

    def _schedule(self):
        speeds, s = [], self._start_kmh
        while s <= self._end_kmh + 0.001:
            speeds.append(round(s, 1))
            s = round(s + self._step_kmh, 1)
        return speeds

    def _run(self):
        self._upd(running=True, phase="starting", error=None, session=None)
        try:
            # 1 — start recording (opens mp4 + pose jsonl + attaches BLE log)
            session_name = self._recorder.start(ble_logger=self._ble,
                                                 cnn_logger=self._cnn)
            self._upd(phase="recording", session=session_name)
            log.info("AutoSession: recording started (%s)", session_name)

            speeds = self._schedule()
            self._upd(total_steps=len(speeds))

            # 2 — warmup: user steps on and starts the belt on the PANEL at the
            #     first speed. We do NOT touch the belt (BLE can't set speed).
            self._upd(phase="warmup", current_kmh=speeds[0] if speeds else None,
                      next_kmh=None, step=0, step_elapsed=0)
            log.info("AutoSession: warmup %ds — user starts belt on panel at %.1f",
                     self._warmup_s, speeds[0] if speeds else 0.0)
            for elapsed in range(self._warmup_s):
                if self._stop_evt.wait(1.0):
                    raise InterruptedError("cancelled during warmup")
                self._upd(step_elapsed=elapsed + 1)

            # 3 — guided cues: show each target; the user sets it on the panel.
            for i, spd in enumerate(speeds):
                if self._stop_evt.is_set():
                    break
                nxt = speeds[i + 1] if i + 1 < len(speeds) else None
                self._upd(phase="ramp", step=i + 1, current_kmh=spd,
                          next_kmh=nxt, step_elapsed=0)
                # Stopped step: the belt emits no BLE, so tell the logger to
                # force-log 0 km/h for this step (else these frames stay
                # unlabeled and get dropped in preprocess).
                if spd == 0.0:
                    self._ble.set_override(0.0)
                else:
                    self._ble.clear_override()
                log.info("AutoSession: cue step %d/%d → SET PANEL TO %.1f km/h",
                         i + 1, len(speeds), spd)
                # Hold the cue for dwell_s, ticking step_elapsed so the UI can
                # count down and warn when the next change is imminent.
                for elapsed in range(self._dwell_s):
                    if self._stop_evt.wait(1.0):
                        break
                    self._upd(step_elapsed=elapsed + 1)
                if self._stop_evt.is_set():
                    break

            self._upd(phase="finishing", current_kmh=None, next_kmh=None)
            log.info("AutoSession: cues done — stop the belt on the panel")

        except InterruptedError:
            log.info("AutoSession: cancelled")
        except Exception as e:
            log.error("AutoSession error: %s", e)
            self._upd(error=str(e))
        finally:
            # clear any stopped-step override so it can't leak into a later run
            try:
                self._ble.clear_override()
            except Exception:
                pass
            # stop recording — flush mp4 moov atom + close jsonl + write manifest
            info = self._recorder.stop()
            log.info("AutoSession: done — %s", info)
            self._upd(running=False, phase="idle", current_kmh=None,
                      next_kmh=None, step=0, step_elapsed=0)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

_cameras:  list        = []
_recorder: Recorder    = None
_ble:      object      = None
_cnn:      object      = None
_auto:     object      = None   # AutoSession


def _camera_by_index(idx: int):
    if 0 <= idx < len(_cameras):
        return _cameras[idx]
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        p = urlparse(self.path)
        path = p.path

        if path == "/":
            self._send_bytes(INDEX_HTML, "text/html; charset=utf-8")
        elif path == "/stream" or path.startswith("/stream/"):
            idx = self._parse_index(path, "/stream")
            self._stream_mjpeg(idx)
        elif path == "/keypoints" or path.startswith("/keypoints/"):
            idx = self._parse_index(path, "/keypoints")
            self._send_json(self._keypoints_payload(idx))
        elif path == "/cameras":
            self._send_json({"cameras": [c.label for c in _cameras]})
        elif path == "/record/status":
            self._send_json(_recorder.status)
        elif path == "/speed":
            # BLE ground-truth speed (the training label). No live AI estimate
            # during data collection — that's what we're recording to build.
            self._send_json({
                "ble_speed_kmh": _ble.latest_speed() if _ble is not None else None,
            })
        elif path == "/ai_speed":
            cnn_spd = _cnn.latest_speed() if _cnn is not None else None
            ble_spd = _ble.latest_speed() if _ble is not None else None
            dbg = _cnn.debug_info() if _cnn is not None else {}
            self._send_json({
                "cnn_speed_kmh": cnn_spd,
                "ble_speed_kmh": ble_spd,
                "cnn_ready": _cnn is not None and _cnn.ready,
                **dbg,
            })
        elif path == "/cnn/tune":
            # Live-retune the CNN clip span: /cnn/tune?fps=26.0
            # Lets us sweep the effective training-fps against BLE truth with no
            # restart. Returns the new span so the sweep can log it.
            qs = parse_qs(urlparse(self.path).query)
            if _cnn is None or not _cnn.ready:
                self._send_json({"error": "cnn not ready"})
            elif "fps" not in qs:
                self._send_json({"error": "missing fps param"})
            else:
                self._send_json(_cnn.set_fps(float(qs["fps"][0])))
        elif path == "/ble/status":
            # Full BLE health for the preflight check card.
            if _ble is None:
                self._send_json({"enabled": False})
            else:
                st = dict(_ble.status)
                st["enabled"] = True
                st["alive"] = _ble.is_alive
                self._send_json(st)
        elif path == "/sessions":
            self._send_json(_recorder.list_sessions())
        elif path == "/session/auto/status":
            self._send_json(_auto.status if _auto else {"running": False, "phase": "idle"})
        elif path.startswith("/sessions/"):
            self._download_session(path[len("/sessions/"):])
        else:
            self.send_error(404)

    def do_POST(self):
        p = urlparse(self.path)
        path = p.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"

        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        if path == "/record/start":
            name = _recorder.start(ble_logger=_ble, cnn_logger=_cnn)
            self._send_json({"ok": True, "session": name})
        elif path == "/record/stop":
            info = _recorder.stop()
            self._send_json({"ok": True, **info})
        elif path == "/session/auto/start":
            if _auto is None:
                self._send_json({"ok": False, "error": "auto session not initialised"}); return
            ok, msg = _auto.start()
            self._send_json({"ok": ok, "msg": msg})
        elif path == "/session/auto/stop":
            if _auto is None:
                self._send_json({"ok": False, "error": "auto session not initialised"}); return
            ok, msg = _auto.stop()
            self._send_json({"ok": ok, "msg": msg})
        elif path == "/ble/connect":
            # Preflight: bring the BLE link up WITHOUT recording, so the user can
            # confirm the treadmill is connected/streaming before pressing REC.
            if _ble is None:
                self._send_json({"ok": False, "error": "BLE disabled"})
            elif _ble.is_alive:
                self._send_json({"ok": True, "already": True, **_ble.status})
            else:
                _ble.start()
                self._send_json({"ok": True, "started": True})
        elif path == "/ble/reset":
            # Force a full reconnect (drop + rescan + handshake). Runs in a
            # worker thread because reset() joins the old BLE thread (~blocks).
            if _ble is None:
                self._send_json({"ok": False, "error": "BLE disabled"})
            else:
                threading.Thread(target=_ble.reset, daemon=True).start()
                self._send_json({"ok": True, "resetting": True})
        else:
            self.send_error(404)

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_index(path: str, prefix: str) -> int:
        rest = path[len(prefix):].strip("/")
        if rest.isdigit():
            return int(rest)
        return 0

    def _keypoints_payload(self, idx: int = 0):
        cam = _camera_by_index(idx)
        if cam is None:
            return {"fps": 0.0, "keypoints": []}
        kps = cam.get_keypoints()
        return {
            "label": cam.label,
            "fps": round(cam.get_fps(), 1),
            "keypoints": [
                {"id": i, "label": KEYPOINT_NAMES[i],
                 "x": kp[0], "y": kp[1], "conf": round(kp[2], 2)}
                if kp else
                {"id": i, "label": KEYPOINT_NAMES[i], "x": None, "y": None, "conf": 0}
                for i, kp in enumerate(kps)
            ] if kps else [],
        }

    def _stream_mjpeg(self, idx: int = 0):
        cam = _camera_by_index(idx)
        if cam is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        # Set a write timeout so a stale browser connection doesn't hold a
        # server thread forever (ThreadingHTTPServer has a finite pool).
        try:
            self.connection.settimeout(10.0)
        except Exception:
            pass
        try:
            while True:
                frame = cam.get_frame()
                if frame:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass

    def _download_session(self, rel: str):
        # rel is "<session>" (list files) or "<session>/<file>" (download a file)
        if ".." in rel:
            self.send_error(400)
            return
        parts = rel.strip("/").split("/")
        if len(parts) == 1:
            sdir = os.path.join(_recorder._dir, parts[0])
            if not os.path.isdir(sdir):
                self.send_error(404)
                return
            self._send_json({"session": parts[0], "files": sorted(os.listdir(sdir))})
            return
        if len(parts) != 2:
            self.send_error(400)
            return
        path = os.path.join(_recorder._dir, parts[0], parts[1])
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{parts[1]}"')
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _send_json(self, obj):
        data = json.dumps(obj).encode()
        self._send_bytes(data, "application/json")

    def _send_bytes(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Treadmill Trainer - Qualcomm IQ-8275</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

header {
  display: flex; align-items: center; justify-content: space-between;
  background: #1a1a2e; padding: 10px 20px; border-bottom: 1px solid #333;
}
header h1 { font-size: 1.05em; color: #00cfff; font-weight: 600; }
#fps-badge {
  background: #0f3; color: #000; font-size: 0.75em; font-weight: 700;
  padding: 3px 10px; border-radius: 12px; font-family: monospace;
}

.layout { display: flex; gap: 16px; padding: 14px; flex-wrap: wrap; }

/* --- Video panel (kept small: only for checking framing) --- */
.video-panel { flex: 0 1 400px; min-width: 0; display: flex; flex-direction: row; gap: 12px; flex-wrap: wrap; }
.video-wrap { position: relative; background: #000; border-radius: 8px; overflow: hidden; flex: 1 1 0; min-width: 220px; }
.video-wrap img { display: block; width: 100%; }
.cam-label {
  position: absolute; top: 8px; left: 8px; z-index: 2;
  background: rgba(0,0,0,0.6); color: #00cfff; font-size: 0.75em; font-weight: 700;
  padding: 3px 10px; border-radius: 10px; text-transform: uppercase; letter-spacing: 1px;
}

/* --- BIG panel-speed cue, overlaid on the video --- */
#cue-overlay {
  position: absolute; inset: 0; z-index: 3; display: none;
  flex-direction: column; align-items: center; justify-content: center;
  pointer-events: none; text-align: center;
  background: radial-gradient(ellipse at center, rgba(0,0,0,0.35) 0%, rgba(0,0,0,0.65) 100%);
}
#cue-overlay.show { display: flex; }
#cue-kicker {
  font-size: 2vw; font-weight: 700; letter-spacing: 4px; text-transform: uppercase;
  color: #fff; text-shadow: 0 2px 8px #000; margin-bottom: 0.2em;
}
#cue-speed {
  font-size: 18vw; font-weight: 900; line-height: 0.9; font-family: monospace;
  color: #ffcc00; text-shadow: 0 4px 24px #000, 0 0 40px rgba(255,204,0,0.5);
}
#cue-speed .unit { font-size: 5vw; color: #fff; }
#cue-count {
  font-size: 3.5vw; font-weight: 900; font-family: monospace; color: #fff;
  text-shadow: 0 2px 8px #000; margin-top: 0.15em;
}
#cue-next { font-size: 2vw; color: #ccc; text-shadow: 0 2px 6px #000; margin-top: 0.4em; }
#cue-overlay.warmup #cue-speed { color: #ff5555; }
#cue-overlay.changing #cue-speed { animation: cueflash 0.5s ease-in-out 3; }
@keyframes cueflash { 0%,100% { color: #ffcc00; } 50% { color: #fff; transform: scale(1.06); } }

/* --- Control panel (wide: the panel-speed cue is the main thing) --- */
.ctrl-panel { flex: 1 1 420px; display: flex; flex-direction: column; gap: 12px; }

.card {
  background: #1c1c2e; border: 1px solid #2a2a4a; border-radius: 10px; padding: 16px;
}
.card h2 { font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 12px; }

/* auto session button */
#auto-btn {
  width: 100%; padding: 22px; border-radius: 10px; border: none; cursor: pointer;
  font-size: 1.3em; font-weight: 900; letter-spacing: 2px;
  background: #00aa44; color: #fff; transition: all 0.15s; margin-bottom: 12px;
}
#auto-btn:hover { filter: brightness(1.15); }
#auto-btn.running { background: #cc2222; animation: pulse 1.2s infinite; }
#auto-btn:disabled { opacity: 0.5; cursor: default; }
#auto-phase {
  font-size: 0.85em; color: #aaa; text-align: center; min-height: 1.4em; margin-bottom: 6px;
}
#auto-phase .highlight { color: #0f0; font-weight: 700; }
#auto-phase .warn { color: #fa0; }
#auto-prog-wrap { height: 8px; background: #222; border-radius: 4px; overflow: hidden; margin-bottom: 10px; }
#auto-prog-bar { height: 100%; background: #00aa44; border-radius: 4px; width: 0; transition: width 0.5s; }

#rec-status {
  margin-top: 10px; font-size: 0.82em; color: #888; text-align: center;
  min-height: 2.4em; line-height: 1.5;
}
#rec-status .highlight { color: #0f0; font-weight: 700; }
#rec-status .warn { color: #fa0; }

.speed-big { font-size: 2.2rem; font-weight: 700; text-align: center; font-family: monospace; }
.speed-sub { font-size: 0.72rem; color: #888; text-align: center; margin-top: 2px; }

/* --- Big recording timer --- */
#rec-timer-box {
  display: none; text-align: center; margin: 4px 0 8px 0;
}
#rec-timer-big {
  font-size: 5.5rem; font-weight: 900; font-family: monospace;
  color: #0f0; line-height: 1; letter-spacing: 2px;
}
#rec-timer-big.warn { color: #fa0; }
#speed-suggest-box {
  margin-top: 8px; text-align: center;
  background: #12121e; border: 2px solid #2a2a4a; border-radius: 12px; padding: 14px 10px 18px;
}
#speed-suggest-box.changing { animation: cueflash 0.5s ease-in-out 3; }
#speed-step-label {
  font-size: 1rem; color: #fff; margin-bottom: 6px; font-weight: 900; letter-spacing: 2px;
  text-transform: uppercase;
}
#speed-suggest {
  font-size: 7rem; font-weight: 900; font-family: monospace; color: #ffcc00;
  line-height: 0.9; text-shadow: 0 0 30px rgba(255,204,0,0.35);
}
#speed-suggest.stopped { font-size: 4.2rem; color: #ff5555; }
#speed-suggest-count {
  font-size: 3rem; font-weight: 900; font-family: monospace; color: #fff; margin-top: 6px; line-height: 1;
}
#speed-suggest-arrow {
  font-size: 1.15rem; color: #ccc; margin-top: 8px; font-weight: 600;
}
#step-bar-wrap {
  display: flex; gap: 3px; margin-top: 8px; align-items: flex-end; height: 18px;
}
.step-seg {
  flex: 1; border-radius: 2px; background: #333; height: 10px; transition: height 0.3s, background 0.3s;
}
.step-seg.done { background: #1e0; height: 14px; }
.step-seg.active { background: #ffcc00; height: 18px; }
.step-seg.upcoming { background: #333; }

/* --- Capture health checks --- */
.hrow { display: flex; align-items: center; gap: 8px; padding: 5px 0; font-size: 0.85em; }
.hlabel { color: #ccc; }
.hval { margin-left: auto; font-family: monospace; color: #888; font-size: 0.9em; }
.hdot {
  width: 11px; height: 11px; border-radius: 50%; flex: 0 0 auto;
  background: #555; box-shadow: 0 0 0 0 rgba(0,0,0,0);
}
.hdot.ok   { background: #1e0; box-shadow: 0 0 8px #1e0; }
.hdot.warn { background: #fa0; box-shadow: 0 0 8px #fa0; }
.hdot.bad  { background: #e33; box-shadow: 0 0 8px #e33; animation: hpulse 1s infinite; }
.hdot.idle { background: #555; }
@keyframes hpulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* --- Bluetooth preflight --- */
.ble-pf {
  display: flex; align-items: center; gap: 10px; padding: 10px 12px;
  border-radius: 8px; font-weight: 700; font-size: 0.95em; margin-bottom: 10px;
  background: #1a1a1a; border: 1px solid #333;
}
.ble-pf.ok   { background: #0b2a0b; border-color: #1e0; color: #7f7; }
.ble-pf.warn { background: #2a220b; border-color: #fa0; color: #fd8; }
.ble-pf.bad  { background: #2a0f0f; border-color: #e33; color: #f99; }
.ble-pf.checking { color: #aaa; }
.ble-btns { display: flex; gap: 8px; }
.ble-btn {
  flex: 1; padding: 9px 8px; border-radius: 8px; border: none; cursor: pointer;
  font-weight: 700; font-size: 0.85em; background: #0a84ff; color: #fff;
}
.ble-btn:hover { filter: brightness(1.1); }
.ble-btn:disabled { opacity: 0.5; cursor: default; }
.ble-btn.secondary { background: #444; }
.pf-hint { font-size: 0.7rem; color: #777; margin-top: 8px; text-align: center; }

/* --- Sessions --- */
#sessions-list { font-size: 0.78em; }
.session-row {
  display: flex; align-items: center; gap: 8px; padding: 6px 0;
  border-bottom: 1px solid #222; flex-wrap: wrap;
}
.session-row:last-child { border-bottom: none; }
.s-name { color: #00cfff; font-weight: 700; flex-basis: 100%; }
.s-meta { color: #aaa; }
.s-dl { margin-left: auto; color: #0af; text-decoration: none; }
.s-dl:hover { text-decoration: underline; }
.no-sessions { color: #555; font-style: italic; }

#summary {
  font-size: 0.75em; color: #555; margin-top: 8px; padding-top: 8px;
  border-top: 1px solid #222;
}
#summary span { color: #888; }
</style>
</head>
<body>

<header>
  <h1>Treadmill Trainer &mdash; Qualcomm IQ-8275 NPU</h1>
  <div id="fps-badge">0.0 fps</div>
</header>

<div class="layout">

  <!-- Video feeds (one per camera, populated from /cameras) -->
  <div class="video-panel" id="video-panel"></div>

  <!-- Controls -->
  <div class="ctrl-panel">

    <!-- Mac speed bridge preflight -->
    <div class="card">
      <h2>Mac Speed Bridge</h2>
      <div id="ble-pf" class="ble-pf checking">
        <span class="hdot idle" id="pf-dot"></span>
        <span id="pf-text">checking…</span>
      </div>
      <div class="ble-btns">
        <button id="pf-connect" class="ble-btn" onclick="bleConnect()">Check</button>
        <button id="pf-reset" class="ble-btn secondary" onclick="bleReset()">Reconnect</button>
      </div>
      <div id="pf-hint" class="pf-hint">
        Mac must be running mac_ble_server.py on port 8765.
      </div>
    </div>

    <!-- Auto Session -->
    <div class="card">
      <h2>Auto Session</h2>
      <button id="auto-btn" onclick="toggleAuto()">&#9654; START SESSION</button>
      <div id="auto-phase">Ready — press START</div>
      <div id="auto-prog-wrap"><div id="auto-prog-bar"></div></div>

      <!-- Big recording timer (shown while running) -->
      <div id="rec-timer-box">
        <div id="rec-timer-big">00:00</div>
        <div id="speed-suggest-box">
          <div id="speed-step-label">SET ON PANEL</div>
          <div id="speed-suggest">-- km/h</div>
          <div id="speed-suggest-count"></div>
          <div id="speed-suggest-arrow"></div>
        </div>
        <div id="step-bar-wrap"></div>
      </div>
      <div id="rec-status"></div>
    </div>

    <!-- Live health checks -->
    <div class="card">
      <h2>Capture Health</h2>
      <div id="health-list">
        <div class="hrow"><span class="hdot idle" id="h-rec-dot"></span>
          <span class="hlabel">Recording</span><span class="hval" id="h-rec">idle</span></div>
        <div class="hrow"><span class="hdot idle" id="h-ble-dot"></span>
          <span class="hlabel">Mac Speed</span><span class="hval" id="h-ble">--</span></div>
        <div id="h-cams"></div>
      </div>
    </div>

    <!-- Real belt speed from the treadmill (via Mac BLE bridge) -->
    <div class="card">
      <h2>Treadmill Real Speed</h2>
      <div id="ble-speed" class="speed-big" style="color:#00cfff">-- km/h</div>
      <div class="speed-sub">via mac_ble_server.py &bull; real belt speed (ground-truth label)</div>
    </div>

    <!-- AI speed estimate from camera -->
    <div class="card">
      <h2>Camera AI Speed</h2>
      <div id="cnn-speed" class="speed-big" style="color:#a0ff80">-- km/h</div>
      <div class="speed-sub" id="cnn-sub">CNN &bull; side camera &bull; warming up…</div>
    </div>

    <!-- Sessions -->
    <div class="card">
      <h2>Recorded Sessions</h2>
      <div id="sessions-list"><span class="no-sessions">No sessions yet.</span></div>
      <div id="summary"></div>
    </div>

  </div>
</div>

<script>
let recording = false;
let cameras = [];

// --- Build camera feeds from /cameras ---
async function buildFeeds() {
  try {
    const r = await fetch("/cameras");
    const d = await r.json();
    cameras = d.cameras || [];
  } catch(e) { cameras = ["front"]; }
  if (!cameras.length) cameras = ["front"];
  const panel = document.getElementById("video-panel");
  panel.innerHTML = cameras.map((label, i) => `
    <div class="video-wrap">
      <span class="cam-label">${label}</span>
      <img src="/stream/${i}" alt="${label} feed">
      ${i === 0 ? `
      <div id="cue-overlay">
        <div id="cue-kicker">Set on panel</div>
        <div id="cue-speed">-- <span class="unit">km/h</span></div>
        <div id="cue-count"></div>
        <div id="cue-next"></div>
      </div>` : ``}
    </div>`).join("");
}

// --- Auto Session ---
let autoRunning = false;
let autoTimer = null;
let autoStartedAt = null;

const PHASE_LABELS = {
  idle: "Ready — press START",
  starting: "Starting…",
  recording: "Recording started",
  warmup: "Step on and start the belt on the PANEL",
  ramp: null,   // built dynamically
  finishing: "Done — stop the belt on the panel",
};

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return String(m).padStart(2, "0") + ":" + String(sec).padStart(2, "0");
}

async function toggleAuto() {
  const btn = document.getElementById("auto-btn");
  const phase = document.getElementById("auto-phase");
  btn.disabled = true;
  // Immediate visual feedback before the network round-trip
  if (!autoRunning) {
    btn.className = "running";
    btn.innerHTML = "&#9646;&#9646; STOP SESSION";
    phase.innerHTML = `<span class="highlight">Iniciando…</span>`;
  } else {
    phase.innerHTML = `<span class="warn">Parando…</span>`;
  }
  try {
    if (!autoRunning) {
      const r = await fetch("/session/auto/start", {method: "POST"});
      const d = await r.json();
      if (!d.ok) {
        btn.className = ""; btn.innerHTML = "&#9654; START SESSION";
        phase.innerHTML = `<span class="warn">Error: ${d.error || d.msg}</span>`;
      }
    } else {
      await fetch("/session/auto/stop", {method: "POST"});
    }
  } catch(e) {
    btn.className = ""; btn.innerHTML = "&#9654; START SESSION";
    phase.textContent = "Network error";
  } finally {
    btn.disabled = false;
  }
}

async function pollAuto() {
  try {
    const r = await fetch("/session/auto/status");
    const s = await r.json();
    const btn  = document.getElementById("auto-btn");
    const phase = document.getElementById("auto-phase");
    const bar  = document.getElementById("auto-prog-bar");
    const timerBox = document.getElementById("rec-timer-box");
    const timerEl  = document.getElementById("rec-timer-big");
    const suggest  = document.getElementById("speed-suggest");
    const arrow    = document.getElementById("speed-suggest-arrow");

    autoRunning = s.running;
    btn.className = s.running ? "running" : "";
    btn.innerHTML = s.running ? "&#9646;&#9646; STOP SESSION" : "&#9654; START SESSION";

    const overlay = document.getElementById("cue-overlay");
    const cueSpeed = document.getElementById("cue-speed");
    const cueKick  = document.getElementById("cue-kicker");
    const cueCount = document.getElementById("cue-count");
    const cueNext  = document.getElementById("cue-next");
    const suggestBox = document.getElementById("speed-suggest-box");
    const stepLabel  = document.getElementById("speed-step-label");
    const countEl    = document.getElementById("speed-suggest-count");

    if (s.running) {
      if (!autoStartedAt) autoStartedAt = Date.now();
      const elapsedS = Math.floor((Date.now() - autoStartedAt) / 1000);
      timerBox.style.display = "block";
      timerEl.textContent = fmtTime(elapsedS);

      if (s.phase === "ramp" && s.total_steps > 0) {
        const pct = Math.round(s.step / s.total_steps * 100);
        bar.style.width = pct + "%";
        bar.style.background = "#00aa44";

        const remain = Math.max(0, (s.dwell_s || 0) - (s.step_elapsed || 0));
        const justChanged = (s.step_elapsed || 0) <= 1;
        const stopped = (s.current_kmh === 0.0);

        // --- side-panel card cue (the dominant readout) ---
        suggestBox.className = justChanged ? "changing" : "";
        if (stopped) {
          stepLabel.textContent = "STOP — STAND STILL";
          suggest.className = "stopped";
          suggest.textContent = "STOP";
        } else {
          stepLabel.textContent = "SET ON PANEL";
          suggest.className = "";
          suggest.textContent = (s.current_kmh != null ? s.current_kmh.toFixed(1) : "--") + " km/h";
        }
        countEl.textContent = remain + "s";
        arrow.textContent = (s.next_kmh != null)
          ? `step ${s.step}/${s.total_steps} · next: ${s.next_kmh.toFixed(1)}`
          : `step ${s.step}/${s.total_steps} · last`;
        phase.innerHTML = stopped
          ? `<span class="warn">STOP — stand still</span>`
          : `<span class="highlight">PANEL → ${s.current_kmh} km/h</span>`;

        // --- big on-video overlay ---
        overlay.className = "show" + (justChanged ? " changing" : "");
        if (stopped) {
          cueKick.textContent = "Stand still";
          if (cueSpeed) cueSpeed.innerHTML = `STOP`;
        } else {
          cueKick.textContent = "Set on panel";
          if (cueSpeed) cueSpeed.innerHTML =
            `${(s.current_kmh != null ? s.current_kmh.toFixed(1) : "--")} <span class="unit">km/h</span>`;
        }
        cueCount.textContent = `${remain}s  ·  step ${s.step}/${s.total_steps}`;
        cueNext.textContent = (s.next_kmh != null)
          ? `next: ${s.next_kmh.toFixed(1)} km/h` : "last step — hold";
      } else if (s.phase === "warmup") {
        suggestBox.className = "";
        stepLabel.textContent = "GET READY";
        suggest.className = "";
        suggest.textContent = "step on!";
        countEl.textContent = "";
        arrow.textContent = "start the belt on the panel";
        bar.style.width = "5%";
        bar.style.background = "#fa0";
        phase.innerHTML = `<span class="warn">Step on and start the belt on the PANEL</span>`;

        overlay.className = "show warmup";
        cueKick.textContent = "Step on and start the belt";
        if (cueSpeed) cueSpeed.innerHTML =
          `${(s.current_kmh != null ? s.current_kmh.toFixed(1) : "0.0")} <span class="unit">km/h</span>`;
        cueCount.textContent = "starting…";
        cueNext.textContent = "the cue advances on its own";
      } else {
        suggestBox.className = "";
        stepLabel.textContent = "SET ON PANEL";
        suggest.className = "";
        suggest.textContent = "--";
        countEl.textContent = "";
        arrow.textContent = "";
        bar.style.width = "2%";
        bar.style.background = "#0af";
        phase.textContent = PHASE_LABELS[s.phase] || s.phase;
        overlay.className = "";
      }
    } else {
      autoStartedAt = null;
      const ov = document.getElementById("cue-overlay");
      if (ov) ov.className = "";
      timerBox.style.display = "none";
      bar.style.width = s.error ? "100%" : "0";
      bar.style.background = s.error ? "#e33" : "#00aa44";
      if (s.error) {
        phase.innerHTML = `<span class="warn">Error: ${s.error}</span>`;
      } else if (s.phase === "idle" && s.session) {
        phase.innerHTML = `<span class="highlight">Done — ${s.session}</span>`;
        loadSessions();
      } else {
        phase.textContent = "Ready — press START";
        bar.style.width = "0";
      }
    }
  } catch(e) {}
  setTimeout(pollAuto, 800);
}
pollAuto();

// --- FPS poll (front camera) ---
async function pollFPS() {
  try {
    const r = await fetch("/keypoints/0");
    const d = await r.json();
    document.getElementById("fps-badge").textContent = d.fps + " fps";
  } catch(e) {}
  setTimeout(pollFPS, 500);
}
pollFPS();

// --- Speed poll (BLE ground-truth + CNN AI) ---
async function pollSpeed() {
  try {
    const r = await fetch("/ai_speed");
    const d = await r.json();
    const ble = document.getElementById("ble-speed");
    if (d.ble_speed_kmh !== null && d.ble_speed_kmh !== undefined) {
      ble.textContent = d.ble_speed_kmh.toFixed(1) + " km/h"; ble.style.color = "#00cfff";
    } else {
      ble.textContent = "-- km/h"; ble.style.color = "#555";
    }
    const cnn = document.getElementById("cnn-speed");
    const cnnSub = document.getElementById("cnn-sub");
    if (!d.cnn_ready) {
      cnn.textContent = "n/a"; cnn.style.color = "#555";
      cnnSub.textContent = "CNN model not loaded";
    } else if (d.cnn_speed_kmh !== null && d.cnn_speed_kmh !== undefined) {
      cnn.textContent = d.cnn_speed_kmh.toFixed(1) + " km/h"; cnn.style.color = "#a0ff80";
      const err = (d.ble_speed_kmh !== null && d.ble_speed_kmh !== undefined)
        ? " Δ" + Math.abs(d.cnn_speed_kmh - d.ble_speed_kmh).toFixed(1) + " km/h vs Real"
        : "";
      cnnSub.textContent = "CNN • side camera" + err;
    } else {
      cnn.textContent = "-- km/h"; cnn.style.color = "#888";
      cnnSub.textContent = "CNN • warming up…";
    }
  } catch(e) {}
  setTimeout(pollSpeed, 500);
}
pollSpeed();

// --- Mac speed bridge preflight ---
let _blePrev = { samples: 0, t: 0 };
async function bleConnect() {
  const btn = document.getElementById("pf-connect");
  btn.disabled = true; btn.textContent = "Checking…";
  try { await fetch("/ble/connect", {method: "POST"}); } catch(e) {}
  setTimeout(() => { btn.disabled = false; btn.textContent = "Check"; }, 1500);
}
async function bleReset() {
  const btn = document.getElementById("pf-reset");
  btn.disabled = true; btn.textContent = "Reconnecting…";
  _blePrev = { samples: 0, t: 0 };
  try { await fetch("/ble/reset", {method: "POST"}); } catch(e) {}
  setTimeout(() => { btn.disabled = false; btn.textContent = "Reconnect"; }, 3000);
}
function setPf(cls, dot, text) {
  document.getElementById("ble-pf").className = "ble-pf " + cls;
  document.getElementById("pf-dot").className = "hdot " + dot;
  document.getElementById("pf-text").textContent = text;
}
async function pollBle() {
  try {
    const r = await fetch("/ble/status");
    const s = await r.json();
    const now = Date.now() / 1000;
    if (!s.enabled) {
      setPf("bad", "bad", "Speed logging disabled on server");
    } else if (s.connected) {
      const streaming = s.samples > _blePrev.samples;
      const spd = (s.latest_speed_kmh != null) ? s.latest_speed_kmh.toFixed(1) : "0.0";
      if (streaming) {
        setPf("ok", "ok", `✔ Mac connected — ${spd} km/h (${s.samples} samples)`);
      } else if (s.samples > 0) {
        setPf("ok", "ok", `✔ Mac connected — ${spd} km/h`);
      } else {
        setPf("warn", "warn", "Mac connected — waiting for speed (start belt on panel)");
      }
    } else if (s.alive) {
      setPf("warn", "bad", "Polling Mac… (is mac_ble_server.py running?)");
    } else {
      setPf("bad", "bad", "Not connected — press Check");
    }
    if (now - _blePrev.t >= 0.9) _blePrev = { samples: s.samples || 0, t: now };
  } catch(e) {
    setPf("bad", "bad", "server unreachable");
  }
  setTimeout(pollBle, 1000);
}
pollBle();

// --- Capture health poll (verifies counters are actually GROWING) ---
let _prevHealth = null;   // { t, frames:{label:poseCount}, video:{label:vCount}, bleSamples }
function setDot(id, cls) {
  const el = document.getElementById(id);
  if (el) el.className = "hdot " + cls;
}
async function pollHealth() {
  const recDot = "h-rec-dot", bleDot = "h-ble-dot";
  try {
    const r = await fetch("/record/status");
    const s = await r.json();
    const now = Date.now() / 1000;

    if (!s.recording) {
      setDot(recDot, "idle");
      document.getElementById("h-rec").textContent = "idle";
      // Mac speed can be probed even when idle; show neutral when idle.
      setDot(bleDot, "idle");
      document.getElementById("h-ble").textContent = "idle";
      document.getElementById("h-cams").innerHTML = "";
      _prevHealth = null;
    } else {
      setDot(recDot, "ok");
      document.getElementById("h-rec").textContent =
        `${s.duration_s}s · ${s.frames} frames`;

      const cams = s.cameras || [];
      const ble = s.ble || {};
      const prev = _prevHealth;

      // Per-camera: green only if BOTH pose and video frame counts advanced.
      let camHtml = "";
      for (const c of cams) {
        let cls = "warn", note = "starting…";
        if (prev && prev.frames[c.label] !== undefined) {
          const dPose = c.pose_frames  - prev.frames[c.label];
          const dVid  = c.video_frames - (prev.video[c.label] || 0);
          if (dPose > 0 && dVid > 0)      { cls = "ok";  note = `${c.video_frames} vid · ${c.pose_frames} pose`; }
          else if (dVid > 0)              { cls = "warn"; note = "video ok, pose stalled"; }
          else                            { cls = "bad"; note = "STALLED — no new frames"; }
        }
        camHtml += `<div class="hrow"><span class="hdot ${cls}"></span>` +
                   `<span class="hlabel">cam: ${c.label}</span>` +
                   `<span class="hval">${note}</span></div>`;
      }
      document.getElementById("h-cams").innerHTML = camHtml;

      // BLE: green only if connected AND sample count is climbing.
      let bcls = "bad", btxt = "not connected";
      if (ble && ble.connected) {
        if (prev && ble.samples > prev.bleSamples) {
          bcls = "ok";
          btxt = `${ble.latest_speed_kmh != null ? ble.latest_speed_kmh.toFixed(1)+" km/h" : "0.0 km/h"} · ${ble.samples} samples`;
        } else {
          bcls = "warn"; btxt = "connected, waiting for data…";
        }
      } else if (ble) {
        bcls = "bad"; btxt = "no data from Mac (mac_ble_server.py running?)";
      }
      setDot(bleDot, bcls);
      document.getElementById("h-ble").textContent = btxt;

      _prevHealth = {
        t: now,
        frames: Object.fromEntries(cams.map(c => [c.label, c.pose_frames])),
        video:  Object.fromEntries(cams.map(c => [c.label, c.video_frames])),
        bleSamples: (ble && ble.samples) || 0,
      };
    }
  } catch(e) {
    setDot(recDot, "bad");
    document.getElementById("h-rec").textContent = "server unreachable";
  }
  setTimeout(pollHealth, 1000);
}
pollHealth();

// --- Sessions ---
async function loadSessions() {
  try {
    const r = await fetch("/sessions");
    const list = await r.json();
    const cont = document.getElementById("sessions-list");
    if (!list.length) {
      cont.innerHTML = '<span class="no-sessions">No sessions yet.</span>';
      document.getElementById("summary").innerHTML = "";
      return;
    }
    cont.innerHTML = list.map(s => {
      const range = s.speed_range ? `${s.speed_range[0].toFixed(1)}&ndash;${s.speed_range[1].toFixed(1)} km/h` : "no BLE";
      const cams = (s.cameras || []).join(", ") || "&mdash;";
      return `
        <div class="session-row">
          <span class="s-name">${s.session}</span>
          <span class="s-meta">${s.frames} fr &bull; ${s.duration_s}s &bull; ${cams} &bull; ${range}</span>
          <a class="s-dl" href="/sessions/${encodeURIComponent(s.session)}">files</a>
        </div>`;
    }).join("");

    const totalFrames = list.reduce((a, s) => a + (s.frames || 0), 0);
    document.getElementById("summary").innerHTML =
      `<span>${list.length} sessions &mdash; ${totalFrames} pose frames total</span>`;
  } catch(e) {}
}

// --- init ---
buildFeeds();
loadSessions();
setInterval(loadSessions, 10000);
</script>
</body>
</html>
""".encode("utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Treadmill pose + data collection server")
    parser.add_argument("--port",    type=int, default=8080)
    parser.add_argument("--cpu",     action="store_true", help="Force CPU (no NPU)")
    parser.add_argument("--camera",  action="append", default=None,
                        help="V4L2 device or index; repeat for multiple cameras "
                             "(e.g. --camera /dev/video26 --camera /dev/video0). "
                             "First is labeled 'front', second 'side', etc.")
    parser.add_argument("--labels",  default=None,
                        help="comma-separated labels matching --camera order (default: front,side,cam2...)")
    parser.add_argument("--dataset", default=DATASET_DIR,  help="Session output dir (default: %(default)s)")
    parser.add_argument("--lstm",    default=LSTM_MODEL,   help="LSTM ONNX model path (default: %(default)s)")
    parser.add_argument("--no-ble",   action="store_true",  help="Disable speed ground-truth logging")
    parser.add_argument("--no-cnn",   action="store_true",  help="Disable CNN live speed estimation")
    parser.add_argument("--cnn-model", default=CNN_MODEL,   help="CNN ONNX model path (default: %(default)s)")
    parser.add_argument("--speed-url", default=AUTO_RAMP_URL + "/speed",
                        help="URL of mac_ble_server.py speed endpoint (default: $MAC_BLE_URL/speed)")
    # legacy / unused — kept so existing run_capture.sh / reset_and_start.sh don't break
    parser.add_argument("--ble-addr",    default=None, help=argparse.SUPPRESS)
    parser.add_argument("--allow-start", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    os.makedirs(args.dataset, exist_ok=True)

    global _cameras, _recorder, _ble, _cnn, _auto
    log.info("Initialising detector (NPU=%s) ...", not args.cpu)
    detector   = PoseDetector(use_npu=not args.cpu)
    _recorder  = Recorder(args.dataset)

    # CNN speed estimator — runs on every frame from the "side" camera.
    _cnn = None
    if not args.no_cnn:
        _cnn = CnnSpeedEstimator(model_path=args.cnn_model)
        if not _cnn.ready:
            log.warning("CNN estimator not ready — live AI speed disabled")
            _cnn = None
        else:
            log.info("CNN speed estimator ready (%s)", args.cnn_model)

    # HTTP speed poller — polls mac_ble_server.py running on the Mac.
    # The Mac does all BLE; the board just GETs the speed over the local network.
    _ble = None
    if not args.no_ble:
        try:
            _ble = HttpSpeedLogger(url=args.speed_url)
            log.info("HTTP speed logging enabled (%s)", args.speed_url)
            _ble.start()
        except Exception as e:
            log.warning("HTTP speed logger failed (%s) — recording without ground-truth speed", e)
            _ble = None

    # Cameras: default to the single CAMERA_DEV if none specified.
    cam_devs = args.camera if args.camera else [CAMERA_DEV]
    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
    else:
        default_labels = ["front", "side", "cam2", "cam3"]
        labels = [default_labels[i] if i < len(default_labels) else f"cam{i}"
                  for i in range(len(cam_devs))]

    _cameras = []
    for i, dev in enumerate(cam_devs):
        label = labels[i] if i < len(labels) else f"cam{i}"
        cam_dev = int(dev) if str(dev).lstrip("-").isdigit() else dev
        _recorder.register_camera(label)
        cam = CameraThread(detector, _recorder, camera_dev=cam_dev, label=label)
        cam.start()
        _cameras.append(cam)
        log.info("Camera %d '%s' -> %s", i, label, dev)

    log.info("Waiting for first frame ...")
    for _ in range(50):
        if _cameras and _cameras[0].get_frame():
            break
        time.sleep(0.1)

    log.info("Server on http://0.0.0.0:%d  (%d camera%s)",
             args.port, len(_cameras), "s" if len(_cameras) != 1 else "")

    _auto = AutoSession(_recorder, _ble, cnn_logger=_cnn)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)

    # Finalize any in-progress recording on SIGTERM/SIGINT so an accidental
    # kill (pkill, systemd stop, terminal close) still flushes the mp4 moov
    # atom via VideoWriter.release() — otherwise the video is left truncated
    # and unplayable, forcing a re-record. jsonl logs are line-buffered and
    # survive regardless, but the raw video is the one thing we can't recover.
    def _graceful(signum, _frame):
        log.warning("Signal %d received — finalizing recording and shutting down", signum)
        try:
            if _recorder.is_recording:
                info = _recorder.stop()
                log.warning("Recording finalized on signal: %s", info)
        finally:
            # server_close from serve_forever's thread; stop the loop.
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _recorder.stop()
        for cam in _cameras:
            cam.stop()
        server.server_close()


if __name__ == "__main__":
    main()
