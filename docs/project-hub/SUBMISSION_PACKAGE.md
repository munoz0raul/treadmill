# Arduino Project Hub submission package

Use this file as the copy/paste checklist for creating the Arduino Project Hub entry.

## Title

AI Treadmill: Turn a Dumb Treadmill into a Bluetooth Smart Treadmill

## Short description / one-liner

Use an Arduino VENTUNO Q and a USB camera to estimate treadmill speed with a local video AI model, optionally accelerate it on the Hexagon HTP NPU, then broadcast it over Bluetooth FTMS so running apps and games can treat a non-smart treadmill like a smart one.

## Difficulty

Intermediate

Rationale: the hardware setup is simple and requires no soldering, but the project uses Linux, Python, ONNX Runtime, Bluetooth Low Energy, and optional model training.

## Suggested category

- Artificial Intelligence / Machine Learning
- Internet of Things
- Sports / Fitness, if available
- Robotics & Machines, if the Hub category list is limited

## Estimated time

- Run with the pretrained model: 1–2 hours after the board is ready.
- Rebuild the dataset cache: about 15–30 minutes depending on machine/storage.
- Train from scratch: several hours on a GPU workstation.

## Cover image

Use a real photo that shows, in one frame:

- the treadmill,
- the side-view USB camera,
- the Arduino VENTUNO Q,
- the browser UI or game connection.

Suggested file in repo:

```text
docs/images/hero.jpg
```

## Components and supplies

### Hardware components

| Quantity | Component | Notes |
|---:|---|---|
| 1 | Arduino VENTUNO Q | Runs the Python app, local ONNX inference, web UI, and BLE FTMS peripheral. |
| 1 | USB webcam | Tested with 1280×720 at 30 fps. Any Linux-compatible UVC webcam should be a good starting point. |
| 1 | Treadmill | No Bluetooth or sensors required. The treadmill is not modified. |
| 1 | Board power supply | Use the power supply recommended for the board and USB peripherals. |
| 1 | Phone/tablet/computer with a fitness app | Zwift, Rouvy, nRF Connect, LightBlue, or another FTMS-compatible app. |
| Optional | GPU workstation | Needed only for training the model from scratch. |

### Software apps and online services

| Tool/service | Purpose |
|---|---|
| Ubuntu/Debian on Arduino VENTUNO Q | Runtime OS for the Python apps. |
| Python 3 | Main project language. |
| OpenCV | Camera capture and video preprocessing. |
| NumPy | Data handling. |
| ONNX Runtime | Default reproducible CPU inference path on the board. |
| Qualcomm AI Runtime (QAIRT) | Optional NPU acceleration path using a prepared A16W8 HTP V75 context binary. |
| PyTorch + torchvision | Optional training workflow. |
| bless | BLE peripheral implementation. |
| bleak | Optional BLE central probe. |
| gdown | Dataset download from Google Drive. |
| GitHub | Source code and trained model release. |

## Repository link

Use the final branch/tag URL when publishing. Current working branch:

```text
https://github.com/munoz0raul/treadmill
```

## External artifacts

### Dataset

```bash
gdown 1V_-AhkDP4gH7HobBH-CnSRrkIpcf5Evp -O dataset_repro.tar.gz
```

Before publishing, verify from a clean machine and optionally add SHA256 here:

```text
dataset_repro.tar.gz SHA256: <TO_FILL>
```

### Trained model

```text
https://github.com/munoz0raul/treadmill/releases/download/model-v1/speed_cnn.onnx
```

Before publishing, optionally add SHA256 here:

```text
speed_cnn.onnx SHA256: <TO_FILL>
```

## Story body

Use [`../PROJECT_HUB.md`](../PROJECT_HUB.md) as the main article body.

When pasting into Arduino Project Hub, replace relative links with absolute links if needed. For example:

```text
../README.md
```

should become:

```text
https://github.com/munoz0raul/treadmill
```

## Schematics / circuit section

This project has no electronic wiring to the treadmill. Use this text in the schematic/circuit area:

```text
There is no electrical connection to the treadmill. The USB webcam connects to the Arduino VENTUNO Q. The VENTUNO Q runs local AI inference and advertises a Bluetooth Low Energy FTMS treadmill service. The treadmill remains mechanically and electrically unchanged.
```

Suggested diagram:

```text
USB webcam → Arduino VENTUNO Q → Bluetooth FTMS → fitness game/app
                 ↑
        local web dashboard

Treadmill: observed by camera only; no modification.
```

## Code section

Use the GitHub repository widget if available:

```text
https://github.com/munoz0raul/treadmill
```

Mention these entry points:

```text
1-data-acquisition/acquire.py
1-data-acquisition/preprocess.py
1-data-acquisition/train.py
1-data-acquisition/export_onnx.py
2-live-inference/live_speed.py
3-bluetooth-ftms/game_server.py
3-bluetooth-ftms/ftms_probe.py
```

## Media checklist

Add these files before final publication:

- [ ] `docs/images/hero.jpg` — treadmill + camera + board + UI/game.
- [ ] `docs/images/rig.jpg` — clear side-camera placement.
- [ ] `docs/images/acquire-ui.jpg` — data acquisition web UI.
- [ ] `docs/images/live-ui.jpg` — live inference UI.
- [ ] `docs/images/ble-panel.jpg` — Bluetooth broadcasting panel.
- [ ] Demo video — 30–60 seconds: start app, walk, game receives speed.
- [ ] Optional NPU proof screenshot/log — `NPU backend ready` and `provider=qnn`.

## Tone and positioning notes

Use a personal maker tone:

- The motivation is making exercise more engaging.
- The treadmill is intentionally not modified.
- The main lesson is that good AI needs good data.
- The board is presented as an edge AI computer that connects local inference to real-world interaction.

Avoid:

- Benchmark comparisons against other boards.
- Claiming the default beginner path uses the NPU. The published, reproducible
  path is CPU ONNX Runtime. The NPU is a **verified optional** path: only claim
  NPU use in the context of running with `--provider qnn` (add `--npu-strict` to
  guarantee no silent CPU fallback) and the QAIRT runtime + A16W8 context binary
  staged on the board. The measured NPU results are in
  `2-live-inference/bench/cpu_vs_npu.md`.
- Medical claims.
- Any unreleased software/package claims not verified in the public image.

## Final pre-publish checklist

- [ ] Replace all placeholder images with real media.
- [ ] Test model download URL.
- [ ] Test dataset download with `gdown` from a clean environment.
- [ ] Add SHA256 checksums for external artifacts.
- [ ] Confirm the final public branch or tag URL.
- [ ] Run `python3 -m py_compile` on all Python scripts.
- [ ] Run `ftms_probe.py` against the board and capture a screenshot/log for the article.
