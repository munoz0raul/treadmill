# Treadmill Speed Estimation from Video

Estimate the speed of a treadmill belt **from a side-view camera alone**, using a
3D convolutional neural network, and validate it live against the treadmill's own
Bluetooth (FTMS) speed readout.

The end result is a model that reads belt speed to within **~0.24 km/h** across
the 1–8 km/h range, running live on an edge board with no calibration constant —
purely from RGB video of a person walking/running.

<p align="center">
  <em>Suggested&nbsp;/&nbsp;Treadmill&nbsp;Real&nbsp;/&nbsp;AI overlay from a live 0→8&nbsp;km/h test session.</em>
</p>

---

## Table of contents

1. [What this is](#what-this-is)
2. [How it works](#how-it-works)
3. [Hardware & topology](#hardware--topology)
4. [Repository layout](#repository-layout)
5. [End-to-end reproduction](#end-to-end-reproduction)
6. [The core insight: train/serve temporal skew](#the-core-insight-trainserve-temporal-skew)
7. [Results](#results)
8. [Design notes & gotchas](#design-notes--gotchas)
9. [License](#license)

---

## What this is

A treadmill reports its belt speed on its own panel, and many models broadcast it
over Bluetooth Low Energy using the standard **FTMS (Fitness Machine Service)**
profile. That gives us a free, accurate **ground-truth label** for supervised
learning.

The idea: point a cheap USB camera at the side of the treadmill, record video of a
person on it together with the true BLE speed, and train a model to predict speed
from the video clips. Once trained, the model runs on an edge board in real time —
the camera becomes a soft speed sensor.

Two independent speed sources are involved:

| Source | What it is | Role |
|---|---|---|
| **Treadmill Real Speed** | The true belt speed, read over BLE/FTMS | Ground-truth training label + live reference |
| **AI Speed** | The CNN's estimate from video | The thing we're building |
| **Suggested Speed** | The value a guided script asks you to set every 30 s | Drives a balanced 0→8 km/h data-collection ramp |

> **The belt is never driven over Bluetooth.** On the treadmill used here (a ZiYou
> unit advertising as `TRMZU`), FTMS speed is **read-only** — even the vendor app
> can only read it. So during data collection a human sets each speed on the panel
> when cued; BLE just logs whatever the belt actually reports.

---

## How it works

### The model

- **Backbone:** torchvision [`mc3_18`](https://pytorch.org/vision/stable/models/generated/torchvision.models.video.mc3_18.html)
  — a mixed 2D/3D ResNet (~11.7 M params) pretrained on **Kinetics-400**, which
  already contains walking / running / treadmill actions. We strip its 400-way
  classification head and attach a small regression head (`512 → 128 → 1`).
- **Input:** an RGB clip of shape `(batch, 3, T=8, 112, 112)` in `[0..1]`.
  Kinetics normalisation is baked into the graph as buffers, so the runtime feeds
  plain `[0..1]` RGB — no host-side preprocessing.
- **Output:** a single scalar — speed in km/h.
- **Loss:** Huber (`delta=1.0`), robust to the occasional label outlier.
- **Transfer learning, two phases:** (1) freeze the backbone and warm up the head
  so its random weights don't wreck the pretrained features; (2) unfreeze and
  fine-tune everything at a lower LR with cosine annealing.

See [`model.py`](model.py) and [`train.py`](train.py).

### The clips (this is the important part)

Every clip is **8 frames sampled by timestamp over a fixed 1.0-second real-time
window** — *not* by frame count. One clip therefore always spans the same real
motion (~one gait cycle) regardless of the camera's actual FPS. This time-based
sampling is applied **identically** in offline preprocessing and in the live
server, which is what makes the model accurate live. See
[the skew section](#the-core-insight-trainserve-temporal-skew) for why this
matters — it's the single most important design decision in the project.

### Live inference

On the edge board, [`pose_server.py`](pose_server.py):

1. Grabs frames from the side camera into a timestamped ring buffer.
2. Every ~0.5 s, samples an 8-frame / 1.0-s clip and runs the ONNX model.
3. Applies a cheap **motion gate** (if the clip barely changes, force 0 km/h — the
   model never saw "stopped" as a smooth extrapolation, so we handle it explicitly).
4. Smooths the last N predictions (rolling mean, default N=6 ≈ 3 s) for a stable
   display value.
5. Serves everything over HTTP + a web UI, and can record labelled sessions.

---

## Hardware & topology

```
  ┌─────────────────┐   BLE / FTMS    ┌──────────────────────┐
  │    Treadmill    │ ──────────────▶ │  BLE bridge host      │
  │  (FTMS, TRMZU)  │  read-only      │  (e.g. a Mac)         │
  └─────────────────┘  speed          │  mac_ble_server.py    │
                                       │  → HTTP :8765/speed   │
                                       └──────────┬───────────┘
                                                  │ LAN (HTTP poll)
        ┌───────────────┐   USB video             ▼
        │  Side camera  │ ─────────────▶ ┌──────────────────────┐
        │  (1280×720)   │                │  Edge board           │
        └───────────────┘                │  pose_server.py       │
                                          │  + speed_cnn.onnx     │
                                          │  → HTTP :8080 web UI  │
                                          └──────────────────────┘
```

- **BLE bridge host** — any machine with Bluetooth and Python. Runs
  [`mac_ble_server.py`](mac_ble_server.py), which does the FTMS handshake and
  serves the current belt speed as JSON on port 8765. (Developed on macOS, hence
  the name, but it's plain [`bleak`](https://github.com/hbldh/bleak) and portable.)
- **Edge board** — the machine with the camera. Runs the capture/inference server.
  Point it at the bridge with `export MAC_BLE_URL="http://<bridge-ip>:8765"`.
- **Training** — done off-device on a CUDA GPU (see below); the board only runs
  inference via ONNX Runtime.

All hosts just need to be on the same LAN. No cloud, no accounts.

---

## Repository layout

| File | What it does |
|---|---|
| [`model.py`](model.py) | The `SpeedCNN` (mc3_18 backbone + regression head). |
| [`preprocess.py`](preprocess.py) | Turns recorded sessions into a balanced clip cache (`.npz`), sampling clips **by timestamp** over a 1.0 s window. |
| [`train.py`](train.py) | Two-phase transfer-learning trainer. Auto-detects CUDA. Holds out one session for cross-day validation. |
| [`export_onnx.py`](export_onnx.py) | Exports the trained `.pt` to a single self-contained ONNX file and numerically verifies it. |
| [`pose_server.py`](pose_server.py) | The edge server: camera capture, live CNN inference, guided data-collection UI, session recording, HTTP API. |
| [`mac_ble_server.py`](mac_ble_server.py) | BLE/FTMS bridge — connects to the treadmill and serves belt speed as JSON. |
| [`run_capture.sh`](run_capture.sh) | Convenience launcher for `pose_server.py` on the board. |
| [`video_overlay.py`](video_overlay.py) | Renders a recorded session to a new video with a Suggested / Real / AI speed table overlaid (aligned per-frame by timestamp). |

**Not in the repo** (see [`.gitignore`](.gitignore)): the recorded `dataset/`,
the `.npz` cache, and the `.onnx`/`.pt`/`.mp4` binaries. You generate these by
following the steps below.

---

## End-to-end reproduction

### 0. Prerequisites

- Python 3.10+ on all hosts.
- **Bridge host:** `pip install bleak` and a working Bluetooth adapter.
- **Board:** `pip install opencv-python-headless onnxruntime numpy` (plus whatever
  pose/NPU delegate your board uses).
- **Training box:** an NVIDIA GPU. `pip install torch torchvision onnx onnxruntime
  onnxscript numpy opencv-python-headless`. The PyTorch CUDA wheel bundles the CUDA
  libraries — you only need the NVIDIA **driver** installed, not the full CUDA
  toolkit. Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

### 1. Start the BLE bridge (on the bridge host)

```bash
python3 mac_ble_server.py
# → serves http://<bridge-ip>:8765/speed
```

Confirm it sees the treadmill:

```bash
curl http://localhost:8765/speed
# {"speed_kmh": 0.0, "ts": ..., "connected": true, ...}
```

If your treadmill advertises a different name, change `TREADMILL_NAME` /
`FTMS_SERVICE` at the top of `mac_ble_server.py`.

### 2. Collect data (on the board)

```bash
export MAC_BLE_URL="http://<bridge-ip>:8765"
bash run_capture.sh
# open http://<board-ip>:8080 in a browser
```

In the web UI, press **START SESSION**. The guided **Auto Session** walks a
**0.0 → 8.0 km/h ramp in 0.5 km/h steps, 30 s each** (17 steps, ~8.5 min plus an
8 s warmup). It shows a big **"SET ON PANEL → X km/h"** cue; you set that speed on
the treadmill panel when asked. Each session is saved under `dataset/session_*/`:

```
session_YYYYMMDD_HHMMSS/
├── side.mp4            # the video
├── side.jsonl         # per-frame pose keypoints + timestamp (one line/frame)
├── speed_ble.jsonl    # Treadmill Real Speed: {"ts":..,"speed_kmh":..}
├── speed_ai.jsonl     # live AI estimate (if the CNN was loaded during capture)
└── manifest.json      # session metadata
```

Record several sessions across days / outfits / people for generalisation. The
project here used 9 sessions.

### 3. Build the clip cache (anywhere with the recordings)

```bash
python3 preprocess.py --sessions dataset/session_A dataset/session_B ...
# → dataset/cache_side.npz
```

This central-crops each frame (drops off-belt clutter), resizes to 112×112,
samples **8-frame / 1.0-s** clips by timestamp, labels each with the mean BLE
speed over its window, and **flattens the histogram** (caps clips per 0.5 km/h
bin) to reduce shrinkage-to-mean. It prints a per-speed histogram at the end —
check that it's roughly flat and covers your full range.

### 4. Train (on the GPU box)

```bash
python3 train.py --cache dataset/cache_side.npz \
                 --warmup 3 --finetune 15 --bs 48 --workers 8
# → models/speed_cnn.pt  (+ *_history.json)
```

One session (`VAL_SESSION` in `train.py`) is held out for **cross-day validation**
— a different recording day, so the reported MAE reflects generalisation, not
memorisation. Expect best val MAE around **0.15 km/h**.

### 5. Export to ONNX

```bash
python3 export_onnx.py --ckpt models/speed_cnn.pt --out models/speed_cnn.onnx
```

> **Important:** the export uses `dynamo=False`. PyTorch's newer dynamo exporter
> splits weights into a separate `.onnx.data` sidecar file; many edge runtimes
> expect a **single self-contained `.onnx`**. The legacy exporter produces one
> ~45 MB file. The script also verifies ONNX vs PyTorch numerically
> (`max_diff < 1e-3`).

### 6. Deploy & run live (on the board)

Copy `models/speed_cnn.onnx` to the board (default path `~/models/speed_cnn.onnx`)
and restart `pose_server.py`. The web UI now shows three live readouts:
**Treadmill Real Speed** (BLE), **AI** (the model), and their delta.

### 7. Make an annotated video (optional)

```bash
python3 video_overlay.py --session dataset/session_YYYYMMDD_HHMMSS
# → dataset/session_.../side_overlay.mp4  (original untouched)
```

Renders a small corner table — **Suggested / Treadmill Real / AI** — aligned to
each frame by its real capture timestamp, and writes the output at the true
capture FPS (the mp4 header's nominal 30 fps is not the real rate).

---

## The core insight: train/serve temporal skew

This is the bug that made the whole thing work once fixed, and it's worth
understanding because it's an easy trap.

**Symptom:** the model validated beautifully offline (val MAE 0.15) but
**systematically under-read speed live** — by a roughly constant factor of ~0.85.

**Root cause:** the training sessions were recorded at a **bimodal frame rate** —
some at ~29.5 fps, some at ~24 fps. The original preprocessing sampled clips **by
frame count** (e.g. "every 4th frame, 8 frames"). That means a clip labelled
"6 km/h" from a 24 fps session covered a *longer real-time window* than the same
label from a 30 fps session. The model was being shown inconsistent amounts of
real motion for the same label — the temporal axis was contaminated. Live, the
server assumed yet another fps, so its clips covered a *different* real duration
than any training clip → the model saw less displacement than it expected →
it under-predicted.

**Fix:** sample clips **by timestamp over a fixed real-time window** (`CLIP_SPAN_S
= 1.0 s`, 8 frames), applied **identically** in `preprocess.py` and
`pose_server.py`. Now every clip — offline or live — spans exactly 1.0 s of real
motion regardless of recording FPS. The constant ~0.85 under-read **vanished**
(measured ratio went to ~1.03, i.e. within noise), with **no calibration constant**
needed.

The one invariant to preserve: **`CLIP_SPAN_S` in `preprocess.py` must equal
`CNN_CLIP_SPAN_S` in `pose_server.py`.** They're both 1.0; keep them in lockstep.

---

## Results

Live 0→8 km/h test, **steady-state** error per 0.5 km/h step (first 12 s of each
30 s dwell discarded as transition/lag), model with no calibration constant:

| Metric (1–8 km/h) | Value |
|---|---|
| Steady-state MAE | **0.24 km/h** |
| Bias | **+0.09 km/h** |
| Mean AI/Real ratio | **1.03** (was ~0.85 before the fix) |
| AI noise (σ) per step | **0.11 km/h** |

Observations:

- **The temporal-skew fix worked** — the constant under-read is gone.
- A mild **regression-to-the-mean S-curve** remains: the model reads slightly high
  mid-range (~+0.4 near 4 km/h) and slightly low at the top (~−0.4 near 8 km/h),
  typical of a bounded regressor. Correctable with a gentle affine calibration if
  desired, but 0.24 MAE rarely justifies it.
- The only meaningful *dynamic* error is **~3 s of smoothing lag** right after a
  speed change — response latency, not measurement bias.
- The treadmill's real minimum speed is **1.0 km/h**: a "0.5 km/h" cue reads 1.0
  on both BLE and AI, so don't trust that one point as a label.

---

## Design notes & gotchas

- **Motion gate for "stopped":** the model was never trained to extrapolate to a
  motionless belt, so a cheap frame-difference gate forces 0 km/h when the clip
  barely changes. Without it a stopped belt reads ~1 km/h.
- **Central crop, not full frame:** frames are cropped to the central horizontal
  band before resize to drop static background clutter (furniture, plants) that
  would otherwise be free capacity spent on noise. Tune `CROP_X0/CROP_X1` in
  `preprocess.py` for your camera placement.
- **Balanced histogram:** capping clips per speed bin (`MAX_CLIPS_PER_BIN`) matters
  — an imbalanced label distribution pushes the regressor's slope below 1
  (shrinkage to the mean).
- **Real FPS ≠ header FPS:** recorded mp4s carry a nominal 30 fps header but the
  true capture rate can be ~18 fps. Anything time-sensitive (clip sampling, the
  overlay) is driven off the per-frame timestamps in `side.jsonl`, never the
  header.
- **ONNX single-file export:** always export with `dynamo=False` (needs
  `onnxscript` installed) so the board loads one file, not an `.onnx` +
  `.onnx.data` pair.
- **No MPS training:** the mc3_18 3D-conv backward pass hangs on Apple MPS
  (PyTorch bug), so `train.py` uses CUDA when available and otherwise CPU.
- **The belt can't be driven over BLE** on this treadmill — FTMS speed is
  read-only. Data collection is human-in-the-loop by design.

---

## License

MIT. See [`LICENSE`](LICENSE).
