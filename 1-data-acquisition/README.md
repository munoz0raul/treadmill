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

## How we trained (high level)

- **Camera position:** side view, the whole body in frame. The central crop keeps
  the walker and the belt and drops the sofa/shelf on the sides.
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

```bash
python3 train.py       --cache dataset/cache_side.npz --out models/speed_cnn.pt
python3 export_onnx.py --ckpt  models/speed_cnn.pt     --out models/speed_cnn.onnx
```

The resulting `speed_cnn.onnx` is what Part 2 (live inference) and Part 3
(Bluetooth) load to read speed from the camera.
