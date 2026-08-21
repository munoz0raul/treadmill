# AI Treadmill — a vision-powered virtual smart treadmill

Turn a **dumb treadmill** (no Bluetooth, no sensors) into a **smart one** using
just a camera and a small AI model running on an **Arduino UNO Q**.

A side camera watches the belt, a 3D convolutional neural network reads the speed
straight from the video, and the board broadcasts that speed over Bluetooth as a
standard fitness machine — so games like Zwift and Rouvy move with your walk,
even though the treadmill itself knows nothing about any of it.

```
dumb treadmill ──▶ camera ──▶ CNN (reads speed from video) ──▶ Bluetooth (FTMS) ──▶ game
```

## The project in four parts

Each folder is a self-contained step with its own README. Read them in order —
each one builds on the last.

| Part | Folder | What it covers |
|---|---|---|
| **1 · Data acquisition** | [`1-data-acquisition/`](1-data-acquisition/) | Film the treadmill and label it by hand, then build a training set. How the model was trained. |
| **2 · Live inference** | [`2-live-inference/`](2-live-inference/) | The app with the model wired in: camera + live AI speed on a web page. No Bluetooth yet. |
| **3 · Bluetooth (FTMS)** | [`3-bluetooth-ftms/`](3-bluetooth-ftms/) | Broadcast the AI speed as a standard smart treadmill so games consume it. The full app. |
| **4 · Demo** | [`4-demo/`](4-demo/) | Final demonstration video and photos *(coming soon)*. |

The full write-up for the Arduino Project Hub lives in
[`docs/PROJECT_HUB.md`](docs/PROJECT_HUB.md).

## How it works, briefly

- **The eye.** A regular USB webcam, placed to the side of the treadmill, sees the
  whole body and the belt.
- **The brain.** A torchvision `mc3_18` 3D-CNN (pretrained on Kinetics-400,
  fine-tuned on our own recordings) looks at a 1-second clip and regresses the
  belt speed in km/h. It runs on the board's CPU via ONNX Runtime.
- **The voice.** The board acts as a Bluetooth **FTMS** (Fitness Machine Service)
  peripheral, streaming the speed the way a real smart treadmill would, so any
  compatible game or app picks it up.

## Hardware

- **Arduino UNO Q** (Qualcomm Dragonwing platform) — runs the model and the BLE
  peripheral.
- A **USB webcam** (we used 1280×720 @ 30 fps).
- A treadmill — any belt treadmill; **no Bluetooth or sensors required**.

## Quick start

```bash
# Part 2 — see the AI speed live (needs a trained models/speed_cnn.onnx)
python3 2-live-inference/live_speed.py --cnn-model ~/models/speed_cnn.onnx

# Part 3 — the full app, broadcasting over Bluetooth to a game
python3 3-bluetooth-ftms/game_server.py --cnn-model ~/models/speed_cnn.onnx
```

Dependencies: Python 3, `opencv-python`, `numpy`, `onnxruntime`, and — for Part 3
— [`bless`](https://github.com/kevincar/bless) (BLE peripheral). Training (Part 1)
also needs `torch` and `torchvision`.

## License

[Mozilla Public License 2.0](LICENSE).
