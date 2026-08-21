# Reproducible training dataset

This folder is the training dataset for the AI Treadmill speed model, in the
exact format the documented [`1-data-acquisition/`](../1-data-acquisition/)
pipeline expects. It is self-contained: anyone can rebuild the training cache
from it with the documented `preprocess.py` and get the same result.

> **Large files live outside git.** Only the small text records
> (`speed_manual.jsonl`, `manifest.json`) and this README are versioned here. The
> `side.mp4` videos and the prebuilt `cache_side.npz` are large binaries hosted
> externally — download the archive from the link in the [root README](../README.md#get-the-dataset-and-model)
> and unpack it over this folder so the `side.mp4` files land next to their
> manifests. You can then rebuild `cache_side.npz` yourself (below).

## What's here

```
dataset_repro/
├── session_<timestamp>/          ← 10 recording sessions
│   ├── side.mp4                  raw side-view video, honest constant 30 fps  (hosted externally)
│   ├── speed_manual.jsonl        the SCREEN speed timeline (step function)    (in git)
│   └── manifest.json             session metadata                            (in git)
├── build_repro_dataset.py        how the sessions were prepared               (in git)
└── cache_side.npz                the built training cache (see below)         (hosted externally)
```

### `side.mp4`
The side-view recording of the walker on the belt, 1280×720, encoded at a
**true, constant 30 fps** — the header fps equals the real frame rate, so
sampling frames by `index / fps` recovers correct wall-clock times.

### `speed_manual.jsonl`
The **speed shown on the treadmill's screen** (the value the operator dialled
in), as a step function — one row each time the setpoint changes:

```json
{"ts": 1787084945.7368, "speed_kmh": 0.0}
{"ts": 1787084969.77,   "speed_kmh": 1.0}
{"ts": 1787085025.78,   "speed_kmh": 1.5}
```

A set speed holds until the next row. This is the ground-truth label:
**the screen speed, not any Bluetooth reading.** Values are on the treadmill's
0.5 km/h grid, held ~30 s each, ramping 0 → 8 km/h (a couple of sessions go to
9–10 km/h; one is a stopped, empty-belt session for the "0 km/h" class).

### `manifest.json`
Session name, start/end epoch timestamps, duration, video geometry + fps, and
the label provenance.

### `cache_side.npz`
The training cache built from the sessions above by `preprocess.py`:

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `X` | `(N, 3, 8, 112, 112)` | uint8 | RGB clips: 8 frames spanning 1.0 s, center-cropped |
| `y` | `(N,)` | float32 | belt speed in km/h |
| `session_id` | `(N,)` | int32 | which session each clip came from (for held-out validation) |

This build: **N = 8679 clips**, speed 0.0 → 10.0 km/h, balanced to ≤ 800 clips
per 0.5 km/h bin.

## Reproduce the cache

```bash
# from the repo root, with opencv-python + numpy installed
python3 1-data-acquisition/preprocess.py \
    --sessions dataset_repro/session_* \
    --out dataset_repro/cache_side.npz
```

Every session reads at 30.00 fps (no `--normalize-fps` needed — the videos are
already at one honest constant rate), labels align by timestamp, and the clip
count / histogram above are reproduced.

Then train:

```bash
python3 1-data-acquisition/train.py \
    --cache dataset_repro/cache_side.npz --out models/speed_cnn.pt
python3 1-data-acquisition/export_onnx.py \
    --ckpt models/speed_cnn.pt --out models/speed_cnn.onnx   # → models/speed_cnn.onnx
```

## How this dataset was prepared

The raw sessions were originally recorded with an earlier tool that logged the
belt speed read over **Bluetooth** and wrote video with a misleading 30 fps
header (real capture was 18–30 fps). Two corrections turned them into the clean,
documented format above — see [`build_repro_dataset.py`](build_repro_dataset.py):

1. **Screen speed, not Bluetooth speed.** The Bluetooth belt trace is noisy and
   captures the belt *ramping* between setpoints. Each reading was snapped to the
   nearest 0.5 km/h and merged into held segments; only segments held long enough
   to be a real dialled-in setpoint were kept, giving the clean step function in
   `speed_manual.jsonl`.
2. **Honest, constant fps.** True per-frame timestamps (recorded alongside each
   video) were used to resample every clip onto a uniform 30 fps grid and write a
   truthful header, so the whole dataset shares one frame rate and
   `preprocess.py`'s time reconstruction is exact.
