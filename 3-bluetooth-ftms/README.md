# Part 3 — Bluetooth (FTMS)

Part 2 shows the speed on a web page. This part broadcasts that same speed over
Bluetooth as a **standard smart-treadmill signal**, so fitness games and apps
(Zwift, Rouvy, phone apps, BLE scanners) connect to it and move with your walk —
even though the real treadmill has no Bluetooth at all.

```
camera ──▶ CnnSpeedEstimator ──▶ FtmsTreadmill (BLE peripheral) ──▶ game
                 │
                 └──▶ web UI (:8090): live camera + AI speed + Bluetooth panel
```

## Files

| File | What it does |
|---|---|
| `game_server.py` | The full app: Part 2 (camera + CNN + UI) **plus** the FTMS peripheral and a Bluetooth control panel in the UI. |
| `ftms_peripheral.py` | The `FtmsTreadmill` class — a BLE peripheral built on `bless` that advertises the Fitness Machine Service and streams the AI speed. |
| `ftms_probe.py` | A diagnostic BLE **central** that mimics how a game pairs a treadmill and prints PASS/FAIL at each step. Used to verify the peripheral. |

## Running it

```bash
python3 game_server.py --camera /dev/video0 --port 8090 \
                       --cnn-model ~/models/speed_cnn.onnx
# --no-ble  → camera + UI only (this collapses back to Part 2)
# --manual  → no camera; set the broadcast speed by hand (validate the BLE path)
```

Open `http://<board-ip>:8090`, press **Start broadcasting**, then look for
**"AI Treadmill"** inside your game (in Zwift running: **Run → Run Speed**).

> A BLE fitness device does **not** appear in your phone/tablet Bluetooth
> settings — only inside a fitness game or a BLE scanner app (nRF Connect,
> LightBlue).

## Running on the NPU (optional)

By default the speed model runs on the **CPU** via ONNX Runtime — that is the
published, reproducible path. The same `mc3_18` model, quantized to **A16W8**
(16-bit activations, 8-bit weights) and lowered onto the board's **Hexagon HTP
V75 NPU** through the Qualcomm AI Runtime (QAIRT), can drive the inference
instead. Pass `--provider qnn`:

```bash
python3 game_server.py --camera /dev/video0 --port 8090 --provider qnn
# --npu-runtime DIR   → runtime dir with bin/ lib/ dsp/ + the context .bin
#                       (default: ~/models/npu)
# --npu-strict        → do NOT fall back to the CPU if the NPU won't come up
#                       (fail loudly — use when a run must be proven on the NPU)
```

The startup log makes the active engine explicit:

```
INFO NPU backend ready (Hexagon HTP V75) from ~/models/npu
INFO CNN estimator: provider=qnn (Hexagon HTP V75) — clip span 1.00s
```

Under the hood the backend runs the on-device `qnn-net-run` once per clip
(process spawn + context load + infer ≈ 0.3 s here; the inference itself is
~26 ms). At the estimator's ~2 Hz cadence that is plenty, and the whole path is
pure Python — no C++, no compilation. If the NPU runtime is missing or fails its
smoke check, the app **falls back to the CPU** automatically (unless
`--npu-strict`), so it never goes silent.

**Prerequisites (not in git — large binaries, distributed separately, exactly
like the `.onnx` model):** a matched **QAIRT 2.47** aarch64 runtime plus the
A16W8 HTP V75 **context binary** (`speed_cnn_a16w8_htpv75.bin`) staged under
`~/models/npu` (`bin/ lib/ dsp/` + the `.bin`). The context binary is produced on
an x86-64 Linux box by the pipeline *ONNX → DLC → A16W8 quantize → HTP V75
context binary* — the full step-by-step (install the free QAIRT SDK, convert,
quantize, compile, stage for the board) is in
[`../2-live-inference/bench/npu-build/README.md`](../2-live-inference/bench/npu-build/README.md).
One gotcha: the board firmware ships QAIRT 2.46, so you must
ship the matching 2.47 DSP skel and point `ADSP_LIBRARY_PATH` at **only** that
dir — otherwise the firmware's 2.46 skel wins the version race and device
creation fails (error 1008). `NpuSpeedBackend` sets this up for you.

**Why a subprocess and not ONNX Runtime's QNN execution provider?** The model's
5D `Conv3d` ops cannot be placed on the HTP by the QNN EP — they fall back to the
CPU — so the QAIRT context-binary + `qnn-net-run` route is the only way onto the
Hexagon NPU for this model. The full CPU-vs-NPU study (fidelity + speedup, with
the deterministic replay methodology behind it) is in
[`../2-live-inference/bench/`](../2-live-inference/bench/).

## FTMS in a nutshell

The **Fitness Machine Service (FTMS)** is the standard Bluetooth profile for gym
equipment. Normally a smart treadmill is the BLE **peripheral** and the game is
the **central**. Here we invert the usual roles for this project: our board is
the peripheral, impersonating a treadmill, and the game connects to it as a
central and consumes a speed we feed from the camera.

The peripheral exposes these GATT characteristics under the FTMS service
(`0x1826`):

| UUID | Name | Role |
|---|---|---|
| `0x2ACD` | **Treadmill Data** | notify — the speed stream we push ~2×/second |
| `0x2ACC` | Fitness Machine Feature | read — feature bitfield |
| `0x2AD4` | **Supported Speed Range** | read — min/max/step speed |
| `0x2AD9` | Fitness Machine Control Point | write + indicate — the handshake |
| `0x2ADA` | Fitness Machine Status | notify |

A few details that matter in practice:

- **Treadmill Data (`0x2ACD`)** is `flags (uint16 LE)` + `instantaneous speed
  (uint16 LE, in 0.01 km/h units)`. So 3.0 km/h is sent as `300`. We push it at
  about 2 Hz.
- **Supported Speed Range (`0x2AD4`)** is required by running games. Zwift reads
  it during discovery and **silently drops** a treadmill that doesn't expose it —
  this was the single fix that got pairing to stick. We advertise 0–20 km/h in
  0.1 km/h steps.
- **Control Point (`0x2AD9`)** is the handshake. A game writes *Request Control*
  (`0x00`) then *Start* (`0x07`) before trusting the data; we acknowledge each
  with a Success indication (`0x80 <op> 0x01`). There is no belt to actuate — we
  just complete the handshake.

## Verifying pairing with `ftms_probe.py`

Debugging "why won't the game connect?" against a real game is painful because
the game is a black box. `ftms_probe.py` is a small BLE central that reproduces a
game's pairing flow — scan, connect, discover, read Feature + Speed Range,
subscribe to Treadmill Data, run the Control Point handshake, watch the speed —
and prints PASS/FAIL at each step.

```bash
python3 ftms_probe.py --name "AI Treadmill"
```

A healthy run looks like this (every step PASS, then a live speed stream):

```
[1] Scanning 8s for 'AI Treadmill' (or FTMS 0x1826)…
[2] Connecting to XX:XX:XX:XX:XX:XX…
[3] Discovering GATT services…
  SERVICE 00001826-0000-1000-8000-00805f9b34fb   <-- FTMS
  PASS  FTMS service present
[4] Reading discovery characteristics a game checks…
  PASS  Fitness Machine Feature 0x2ACC
  PASS  Supported Speed Range 0x2AD4  (0.0–20.0 km/h, 0.1 step)
[5] Subscribing to Treadmill Data 0x2ACD…
  PASS  subscribed
[6] Control Point handshake (Request Control → Start)…
  PASS  Request Control ack (0x80 00 01)
  PASS  Start ack (0x80 07 01)
[7] Watching speed for 10s…
      DATA 0000 2c01  flags=0x0000  speed=3.00 km/h
      …
=== SUMMARY ===
If steps 3–6 all PASS, a well-behaved FTMS game should pair.
```

If every step passes, a well-behaved FTMS game should pair. Treadmill Data
notifies at **≈ 2 Hz**.
