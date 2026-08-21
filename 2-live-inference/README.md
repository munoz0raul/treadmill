# Part 2 — Live inference

This is the app with the model wired in, before we add Bluetooth. Point the
camera at the treadmill, run the trained CNN on the video, and show the estimated
belt speed live on a web page: your video on the left, the AI speed on the right.

```
camera ──▶ CnnSpeedEstimator ──▶ web UI (:8090): live camera + AI speed
```

## Files

| File | What it does |
|---|---|
| `live_speed.py` | Camera capture + the ONNX speed estimator + a simple web UI. No Bluetooth. |

## Running it

```bash
python3 live_speed.py --camera /dev/video0 --port 8090 \
                      --cnn-model ~/models/speed_cnn.onnx
```

Open `http://<board-ip>:8090` and you'll see the live camera and a big speed
readout that updates as you walk.

## How it stays accurate

The estimator buffers cropped RGB frames tagged with wall-clock timestamps and,
several times a second, samples **8 frames spanning exactly 1.0 second** — the
same time-based sampling used when building the training set. That is the key
invariant: `CNN_CLIP_SPAN_S = 1.0` here must match `CLIP_SPAN_S` in Part 1's
`preprocess.py`, so a live clip covers the same real motion the model was trained
on regardless of the camera's (variable) frame rate.

Two small touches make it behave well live:

- **Motion gate** — if the clip barely changes frame-to-frame, the belt is
  stopped and we report `0.0` directly (the model never saw a stopped belt and
  can't extrapolate to it).
- **Rolling average** — predictions are smoothed over a short window so the
  readout doesn't jitter.

Inference runs on the CPU via onnxruntime in a background thread, so feeding
frames stays cheap and the web stream never stalls.

Part 3 is exactly this app plus a Bluetooth FTMS peripheral, so a fitness game
can consume the same speed.
