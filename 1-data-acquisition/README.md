# Part 1 — Data acquisition

The treadmill has no sensors, so before we can train a model we have to build our
own ground truth: video of the belt paired with the speed it was running at. This
part is the recording rig and the script that turns those recordings into a
training cache.

## The idea

A side camera films the treadmill. While you walk, **you** tell the app what
speed the belt is set to by tapping a button. Every tap is written — with a
timestamp — next to the video. The dataset then assumes the value you picked is
the true belt speed from that moment until you change it again (a step function).

No Bluetooth, no second computer, no network calls. One camera, one web page,
one operator.

```
treadmill (dumb) ──▶ side camera ──▶ acquire.py ──▶ ~/dataset/session_<ts>/
                                          ▲
                          you tap the current speed here
```

## Files

| File | What it does |
|---|---|
| `acquire.py` | Side-camera capture + a manual speed picker (Stopped, 0.5 … 8.0 km/h). Records **raw** video plus a label timeline. |
| `preprocess.py` | Turns recorded sessions into the RGB-clip cache the model trains on. Reads labels from the manual timeline; normalizes FPS. |
| `model.py` | The speed regressor (mc3_18 backbone + regression head). |
| `train.py` | Two-phase fine-tuning (freeze head → finetune). |
| `export_onnx.py` | Exports the trained model to a single ONNX file. |

## Recording a session

```bash
python3 acquire.py --camera /dev/video0 --port 8080 --dataset ~/dataset
```

Open `http://<board-ip>:8080`, then:

1. Set a speed on the treadmill and tap the matching button in the UI.
2. Press **START RECORDING**.
3. Walk. Each time you change the treadmill speed, tap the matching button —
   this keeps the label timeline in sync.
4. Press **STOP RECORDING** when done.

Each session is written to `~/dataset/session_<timestamp>/`:

```
session_20260818_001601/
├── side.mp4            raw side video (no overlays — a clean training signal)
├── speed_manual.jsonl  one row per speed change (+ start/stop):
│                         {"ts": 1723937761.2, "speed_kmh": 3.0}
└── manifest.json       start/end time, frame count, measured fps, note
```

> The live preview draws the current label and a REC dot on screen, but the saved
> `side.mp4` is **raw** — overlays would pollute the training signal.

## Building the training cache

```bash
python3 preprocess.py --sessions dataset/session_* --out dataset/cache_side.npz
# add --normalize-fps to re-encode any outlier video to the common frame rate
```

`preprocess.py` samples short RGB clips. Each clip is **8 frames picked by
timestamp** so it always spans exactly **1.0 second** of real motion (≈ one gait
cycle), independent of the recording FPS. Frames are centre-cropped to the middle
of the frame (keeps the walker + belt, drops room clutter) and resized to
112×112. The clip's label is the belt speed set during that window. Clips are
capped per 0.5 km/h bin so the speed histogram stays flat.

### A note on FPS

Because clips are sampled **by timestamp**, mixed recording frame rates are
handled by construction — a 1-second clip is a 1-second clip whether the camera
ran at 24 or 30 fps. As a safeguard, `preprocess.py` reports each session's fps
and, with `--normalize-fps`, re-encodes any outlier to the dataset's dominant fps
(via `ffmpeg`) so the whole dataset shares one solid, consistent frame rate.

## Camera placement

The model reads gait and belt motion from a **side view**, so placement matters
more than camera quality:

- **Side of the treadmill**, roughly perpendicular to the belt, at about waist
  height. The **whole body and the belt** should be in frame.
- **1280×720** is what we recorded and what the crop constants assume. A plain USB
  webcam is fine.
- Steady framing (a tripod or shelf). The belt should occupy the central band of
  the frame, not the far edge.

`preprocess.py` and the live apps take a **central horizontal crop** of the
1280-wide frame to drop room clutter and keep the walker:

```
CROP_X0 = 240   # keep x ∈ [240, 1040) of a 1280-wide frame → an 800×720 band
CROP_X1 = 1040  # (full body incl. arms, no sofa/shelf on the sides)
```

If your camera frames the scene differently — or isn't 1280 wide — adjust
`CROP_X0`/`CROP_X1` in `preprocess.py` so the crop still contains the walker and
the belt, and set the matching `CNN_CROP_X0`/`CNN_CROP_X1` in the live apps
(`live_speed.py`, `game_server.py`) to the same values. The crop must be identical
between training and inference.

![The recording rig: side camera framing the walker and the belt](../docs/images/rig.jpg)

## How we trained (high level)

- **Protocol:** set each speed on the panel, hold it for **~30 seconds**, then
  step up. We walked **0 → 8 km/h in 0.5 km/h steps** (17 steps, ~8–9 minutes per
  session).
- **Coverage:** multiple sessions across different days, outfits, and people, plus
  an **empty-belt (0 km/h)** session so the model learns what "stopped" looks
  like.
- **Model:** a torchvision **mc3_18** 3D-CNN pretrained on **Kinetics-400** (which
  already contains walking/running actions). We drop its 400-way classifier and
  attach a small regression head that outputs km/h. Trained with **HuberLoss** in
  two phases — 5 epochs with the backbone frozen (head only), then 25 epochs
  fine-tuning everything at a lower learning rate. Kinetics normalization is baked
  into the model, so it takes plain [0..1] RGB clips. Finally exported to a single
  **ONNX** file (opset 17) that the live app loads.

From the repository root:

```bash
# 5 warmup + 25 finetune epochs are the defaults; shown explicitly for the record
python3 1-data-acquisition/train.py \
    --cache dataset_repro/cache_side.npz \
    --out models/speed_cnn.pt \
    --warmup 5 --finetune 25

python3 1-data-acquisition/export_onnx.py \
    --ckpt models/speed_cnn.pt \
    --out models/speed_cnn.onnx
```

The resulting `speed_cnn.onnx` is what Part 2 (live inference) and Part 3
(Bluetooth) load to read speed from the camera.

## What to expect

Building the cache from the 10 sessions in [`dataset_repro/`](../dataset_repro/)
reproduces:

```
X shape: (8679, 3, 8, 112, 112)  dtype: uint8
y range: 0.0 → 10.0 km/h

Samples per speed (km/h):
   0 km/h:   802      5 km/h:   527
   1 km/h:   754      6 km/h:  1420
   2 km/h:  1486      7 km/h:   452
   3 km/h:   506      8 km/h:   957
   4 km/h:  1472      9 km/h:   132
                     10 km/h:   171
```

(Half-step bins are collapsed to whole km/h above; the per-0.5 balancing cap is
800 clips/bin, so the well-covered speeds sit near that ceiling.)

Training `mc3_18` (~11.7 M params) with the command above (5 warmup + 25
fine-tune epochs, one session fully held out for validation) gives a best
validation **MAE ≈ 0.15 km/h** — `train.py` keeps the checkpoint from the
best-scoring epoch. The exported `speed_cnn.onnx` is **≈ 44 MB**.
