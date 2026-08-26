# AI Treadmill — a vision-powered virtual smart treadmill

Turn a **dumb treadmill** (no Bluetooth, no sensors) into a **smart one** using
just a camera and a small AI model running on an **Arduino VENTUNO Q**.

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
[`docs/PROJECT_HUB.md`](docs/PROJECT_HUB.md). A copy/paste submission checklist
is in [`docs/project-hub/SUBMISSION_PACKAGE.md`](docs/project-hub/SUBMISSION_PACKAGE.md).

## How it works, briefly

- **The eye.** A regular USB webcam, placed to the side of the treadmill, sees the
  whole body and the belt.
- **The brain.** A torchvision `mc3_18` 3D-CNN (pretrained on Kinetics-400,
  fine-tuned on our own recordings) looks at a 1-second clip and regresses the
  belt speed in km/h. The default reproducible path runs on the board's CPU via
  ONNX Runtime; an optional QAIRT path runs the same model on the Hexagon HTP NPU.
- **The voice.** The board acts as a Bluetooth **FTMS** (Fitness Machine Service)
  peripheral, streaming the speed the way a real smart treadmill would, so any
  compatible game or app picks it up.

## Hardware

- **Arduino VENTUNO Q** — runs the model and the BLE peripheral. Built on the
  **Qualcomm Dragonwing IQ8 (IQ-8275)**: 8-core Kryo CPU, Adreno 623 GPU, Hexagon
  NPU, 16 GB LPDDR5, Wi-Fi 6 + Bluetooth 5.3, running Ubuntu/Debian.
- A **USB webcam** (we used 1280×720 @ 30 fps).
- A treadmill — any belt treadmill; **no Bluetooth or sensors required**.

## Get the dataset and model

The training data and the trained model are too large for git, so they are hosted
externally. Download and unpack them at the repo root:

```bash
# Reproducible dataset — 10 sessions + a prebuilt training cache (~5 GB).
# Hosted on Google Drive; use gdown (plain curl hits Drive's scan interstitial
# and saves an HTML page instead of the tarball for files this large).
pip install gdown
gdown 1V_-AhkDP4gH7HobBH-CnSRrkIpcf5Evp -O dataset_repro.tar.gz
tar -xzf dataset_repro.tar.gz                      # → dataset_repro/

# Trained model (~44 MB ONNX) — skip if you plan to train from scratch
mkdir -p models
curl -L -o models/speed_cnn.onnx \
    "https://github.com/munoz0raul/treadmill/releases/download/model-v1/speed_cnn.onnx"
```

> Both links are live. The model is a
> [GitHub Release asset](https://github.com/munoz0raul/treadmill/releases/tag/model-v1)
> (`model-v1`); the dataset is on Google Drive. See
> [`dataset_repro/README.md`](dataset_repro/README.md) for the dataset format and
> how it was built.

Prefer to build everything yourself? Skip the model download and follow the
train-from-scratch path below — it reproduces `models/speed_cnn.onnx` from the
sessions in `dataset_repro/`.

## Install dependencies

Use a virtual environment. Training and the board have separate requirement sets:

```bash
python3 -m venv venv && source venv/bin/activate

pip install -r requirements-train.txt   # Part 1 — training (GPU workstation)
pip install -r requirements-board.txt   # Parts 2 & 3 — live + Bluetooth (the board)
pip install -r requirements-debug.txt   # optional — ftms_probe.py BLE central
```

## Set up the Arduino VENTUNO Q

The board runs Ubuntu/Debian. Parts 2 and 3 run **on the board**; Part 1 training
runs on a separate GPU workstation.

```bash
# 1. SSH into the board (use your board's address / user)
ssh <user>@<board-ip>

# 2. Get the code on the board
git clone https://github.com/munoz0raul/treadmill.git
cd treadmill

# 3. System packages
sudo apt update
sudo apt install -y python3-venv python3-opencv bluez v4l-utils

# 4. Python deps in a venv
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -r requirements-board.txt

# 5. Find the camera device node
v4l2-ctl --list-devices        # note the /dev/videoN for your USB webcam

# 6. Put the trained model where the apps expect it
mkdir -p ~/models
# either download it straight onto the board …
curl -L -o ~/models/speed_cnn.onnx \
    "https://github.com/munoz0raul/treadmill/releases/download/model-v1/speed_cnn.onnx"
# … or copy it from the workstation where you trained/exported it:
#   scp models/speed_cnn.onnx <user>@<board-ip>:~/models/
```

BlueZ must be running for the FTMS peripheral (`sudo systemctl status bluetooth`).

## Reproduce it end to end

```bash
# ── On the GPU workstation (Part 1) ──────────────────────────────────────────
# 1. Get the dataset (above) and install training deps
pip install -r requirements-train.txt

# 2. Build the training cache from the recorded sessions
python3 1-data-acquisition/preprocess.py \
    --sessions dataset_repro/session_* --out dataset_repro/cache_side.npz

# 3. Fine-tune the model  (holds out one session for validation)
python3 1-data-acquisition/train.py \
    --cache dataset_repro/cache_side.npz --out models/speed_cnn.pt

# 4. Export to ONNX
python3 1-data-acquisition/export_onnx.py \
    --ckpt models/speed_cnn.pt --out models/speed_cnn.onnx

# ── On the Arduino VENTUNO Q (Parts 2 & 3) ───────────────────────────────────
# 5. Copy models/speed_cnn.onnx to the board's ~/models/ (see setup above)

# 6. Part 2 — see the AI speed live on a web page
python3 2-live-inference/live_speed.py --camera /dev/video0 --port 8090 \
    --cnn-model ~/models/speed_cnn.onnx

# 7. Part 3 — the full app: broadcast the speed over Bluetooth (FTMS)
python3 3-bluetooth-ftms/game_server.py --camera /dev/video0 --port 8090 \
    --cnn-model ~/models/speed_cnn.onnx

# Optional: run the live model on the Hexagon HTP NPU instead of CPU
# (requires QAIRT runtime + speed_cnn_a16w8_htpv75.bin staged under ~/models/npu)
python3 3-bluetooth-ftms/game_server.py --camera /dev/video0 --port 8090 \
    --provider qnn --npu-runtime ~/models/npu --npu-strict

# 8. Open http://<board-ip>:8090, press "Start broadcasting", then pair
#    "AI Treadmill" inside Zwift / Rouvy (Run → Run Speed).
```

## What to expect

- **Dataset:** 10 sessions → cache `X (8679, 3, 8, 112, 112)` uint8, labels
  0.0–10.0 km/h (balanced to ≤ 800 clips per 0.5 km/h bin).
- **Training:** `mc3_18` 3D-CNN (~11.7 M params) fine-tuned for 30 epochs
  (5 warmup + 25 fine-tune); best validation **MAE ≈ 0.15 km/h** on a fully
  held-out session.
- **Model:** a single `speed_cnn.onnx`, **≈ 44 MB**, CPU inference on the board by default.
- **Optional NPU path:** A16W8 QAIRT context binary on Hexagon HTP V75; benchmarked at **0.054 km/h** MAE vs the CPU output on identical clips and **23.5×** lower median inference latency (624 ms → 26.6 ms). See [`2-live-inference/bench/cpu_vs_npu.md`](2-live-inference/bench/cpu_vs_npu.md).
- **Live:** speed updates a few times a second; BLE notifies at **≈ 2 Hz**.

## Resources

**Arduino VENTUNO Q**

- [Product page](https://www.arduino.cc/product-ventuno-q/)
- [Store](https://store-usa.arduino.cc/products/ventuno-q)
- [App Lab quickstart](https://docs.arduino.cc/software/app-lab/getting-started/quickstart/)
- [App Lab Bricks](https://docs.arduino.cc/software/app-lab/bricks/about-bricks)

**Qualcomm**

- **Dragonwing IQ8 series / IQ-8275 (QCS8275)** — the SoC the VENTUNO Q is built
  on. See Qualcomm's Dragonwing IQ8-series product and documentation pages
  (qualcomm.com Dragonwing / docs.qualcomm.com).
- **Qualcomm AI Runtime SDK (QAIRT)** — the NPU toolchain used for the optional
  NPU path. Free from the **Qualcomm Software Center** (the *Community* edition
  needs no login); search "Qualcomm AI Runtime SDK" and pick version
  **2.47.0.260601**. Full build recipe:
  [`2-live-inference/bench/npu-build/README.md`](2-live-inference/bench/npu-build/README.md).

## License

[Mozilla Public License 2.0](LICENSE).
