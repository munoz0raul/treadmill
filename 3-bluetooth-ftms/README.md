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
