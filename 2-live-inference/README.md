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
  stopped and we report `0.0` directly. The training set *does* include
  stopped/empty-belt clips, so the model has a real "0 km/h" class; the gate is a
  cheap robustness backstop that pins a dead-still belt to zero without waiting on
  the rolling average.
- **Rolling average** — predictions are smoothed over a short window so the
  readout doesn't jitter.

Inference runs on the CPU via onnxruntime in a background thread, so feeding
frames stays cheap and the web stream never stalls.

Part 3 is exactly this app plus a Bluetooth FTMS peripheral, so a fitness game
can consume the same speed.

## Comparing CPU vs NPU (optional)

The VENTUNO Q's Qualcomm Dragonwing IQ8 has a Hexagon NPU (neural processing
unit) in addition to the CPU. This model was verified to convert and lower fully
onto that NPU (via the Qualcomm AI Runtime, quantized to A16W8 — 16-bit
activations, 8-bit weights). The interesting engineering question is then: **does
running on the NPU change the answer, and how much faster is it?**

`live_speed.py` doubles as the reference-capture tool for that comparison. Three
flags — all **off by default**, so the published app is unaffected — turn on
instrumentation:

| Flag | What it records |
|---|---|
| `--log run.jsonl` | one JSON line per inference: the **raw** prediction (before smoothing/gating), per-inference latency `infer_ms`, the motion-gate value, and the operator-dialled panel speed `set_kmh` (ground truth). |
| `--dump-clips DIR` | saves the **exact** `(1,3,8,112,112)` float32 tensor fed to the model as `clip_NNNNNN.npy`. |
| `--dump-every N` | dump only every Nth clip (keeps the folder small). |

### Why replay, not two live runs

Two separate live sessions (walk on CPU, walk again on NPU) mostly measure
*walk-to-walk variation* — different steps, lighting, framing — not the engine.
So the method is **replay**: capture the exact input tensors once on the CPU, then
run those **same tensors** through the NPU offline. The only thing that differs is
the engine, so the diff is a clean measure of the quantization cost.

### The protocol

1. **Record one structured 0→8 km/h ramp on the CPU** (the same ramp used to
   build the dataset — *not* a random walk; the ramp gives ground-truth panel
   speeds and reveals whether the NPU errs differently at different speeds). Hold
   each step ~10–15 s; start with a few seconds of stopped belt for the 0 km/h
   case. Press **Set label** in the UI at each step to stamp the panel speed.

   ```bash
   python3 live_speed.py --camera /dev/video0 --port 8090 \
       --cnn-model ~/models/speed_cnn.onnx \
       --log cpu_run.jsonl --dump-clips cpu_clips/
   ```

2. **Replay the dumped clips through each engine.** The CPU replay is the float
   reference. For this model, ONNX Runtime's QNN execution provider cannot place
   the 5D `Conv3d` ops on the HTP, so the committed NPU report uses the QAIRT
   context-binary path (`qnn-net-run`) described in `bench/README.md`.

   ```bash
   # reference (float ONNX on CPU):
   python3 bench/replay_clips.py --clips cpu_clips/ \
       --model ~/models/speed_cnn.onnx --provider cpu --out cpu_ref.jsonl

   # NPU: run the same clip_*.npy tensors through the QAIRT A16W8 HTP V75
   # context binary with qnn-net-run, emitting JSONL in the same shape as
   # replay_clips.py (see bench/README.md → "Notes on the NPU path").
   ```

3. **Generate the comparison report** (`bench/compare_runs.py`) — joins the runs
   by `clip_id` and emits a Markdown table:

   ```bash
   python3 bench/compare_runs.py --cpu cpu_run.jsonl --npu npu.jsonl \
       --out cpu_vs_npu.md --json cpu_vs_npu.json
   ```

The report has three parts: **numeric fidelity** (MAE and max diff of NPU vs CPU
on identical clips — the quantization cost), **accuracy vs the panel ground
truth** (overall and per speed band, for both engines), and **performance**
(per-inference latency p50/p95 and the speedup). The `.json` is machine-readable
for charts.

> This CPU-vs-NPU comparison is an *extra* engineering study layered on top of the
> core project (which ships CPU-only for reproducibility). The bench tooling in
> `bench/` and the `--log`/`--dump-clips` flags exist only to produce that
> comparison; nothing in the default run path depends on them.
