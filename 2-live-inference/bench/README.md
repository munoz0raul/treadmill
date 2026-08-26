# CPU vs NPU benchmark

An *extra* engineering study on top of the core AI Treadmill project. The shipped
app runs the speed model on the **CPU** (via ONNX Runtime) for reproducibility.
Separately, the same model was verified to convert and lower fully onto the
Ventuno Q's **Hexagon NPU** (neural processing unit) through the Qualcomm AI
Runtime (QAIRT), quantized to **A16W8** (16-bit activations, 8-bit weights). This
folder answers two questions about that NPU path:

1. **Fidelity** — does the NPU (quantized) give the same speed as the float CPU
   model on the *same input*? (i.e. what did quantization cost?)
2. **Performance** — how much faster / lighter is the NPU per inference?

## Why "replay", not two live walks

The naïve approach — walk on the treadmill with the CPU app, then walk again with
the NPU app — is wrong. Two live sessions differ in steps, lighting and framing,
so the comparison would mostly measure *walk-to-walk variation*, not the engine.

Instead we **replay identical inputs**. The CPU session dumps the exact
`(1,3,8,112,112)` tensors it fed the model; those same tensors are then run
through the NPU offline. The clip contents are held fixed, so any difference is
the engine alone.

```
                          ┌── replay through CPU (float)  ─┐
live_speed.py ── dump ────┤                                ├── compare_runs.py ── report
 (--dump-clips)  clips    └── replay through NPU (A16W8)  ─┘
```

## Files

| File | Role |
|---|---|
| `replay_clips.py` | Run a folder of dumped `clip_*.npy` through a model on a chosen execution provider (`cpu` / `qnn`); log the raw prediction + per-inference latency. |
| `compare_runs.py` | Join a CPU live log and an NPU replay by `clip_id`; emit a Markdown + JSON report: fidelity (NPU vs CPU), accuracy vs panel ground truth (per speed band), and latency. |
| `guided_ramp.py` | Optional: auto-stamps the panel-speed label on a fixed cadence (default 0→8 km/h, 10 s each) so you don't touch the browser mid-walk. Run it on the board in a second terminal while `live_speed.py` records; keep the treadmill panel matching the speed it announces. |

## End-to-end

```bash
# 1. Record a structured 0→8 km/h ramp on the CPU, labelling the panel speed
#    at each step (Set label button). Hold each step ~10-15 s.
python3 ../live_speed.py --camera /dev/video0 --port 8090 \
    --cnn-model ~/models/speed_cnn.onnx \
    --log cpu_run.jsonl --dump-clips cpu_clips/

#    …or let guided_ramp.py stamp the labels for you (second terminal on the
#    board) — just keep the treadmill panel matching the speed it announces:
python3 guided_ramp.py --port 8090 --hold 10 --stop 8

# 2. Replay the same clips through each engine.
python3 replay_clips.py --clips cpu_clips/ --model ~/models/speed_cnn.onnx \
    --provider cpu --out cpu_ref.jsonl
python3 replay_clips.py --clips cpu_clips/ --model ~/models/speed_cnn.onnx \
    --provider qnn --out npu.jsonl

# 3. Report.
python3 compare_runs.py --cpu cpu_run.jsonl --npu npu.jsonl \
    --out cpu_vs_npu.md --json cpu_vs_npu.json
```

> In bench mode the motion gate is bypassed for *logging*: every clip is run and
> recorded, including the stopped-belt (0 km/h) segment, so the comparison covers
> the full speed range. The live displayed speed still gates to 0.0 when stopped.

## What the CPU log contains

`live_speed.py --log` writes JSONL. First line is a session header; every
subsequent line is one inference:

```json
{"type":"header","provider":"cpu","model":"speed_cnn.onnx","model_sha256":"…",
 "clip_span_s":1.0,"n_frames":8,"smooth":6,"host":"…","git_commit":"…"}
{"type":"infer","ts":1723937761.23,"clip_id":128,"provider":"cpu",
 "raw_kmh":4.87,"speed_kmh":4.9,"infer_ms":142.3,"mean_mov":0.031,"set_kmh":5.0}
```

- `raw_kmh` — model output **before** smoothing/calibration/motion-gate. This is
  what the fidelity comparison uses; the smoothed `speed_kmh` would hide engine
  differences.
- `infer_ms` — wall-clock of the single `session.run(...)` call.
- `set_kmh` — the panel speed you stamped (ground truth). `null` until first set.
- `clip_id` — ties the line to `cpu_clips/clip_<clip_id>.npy` for replay.

## Notes on the NPU path

The NPU numbers in `cpu_vs_npu.md` were **not** produced by ONNX Runtime's QNN
execution provider. That provider cannot place this model's 5D `Conv3d` ops on
the HTP (they fall back to the CPU), so it is not a real NPU run. Instead the
model was taken through the QAIRT context-binary path:

```
speed_cnn.onnx ──▶ qairt-converter ──▶ .dlc
              ──▶ qairt-quantizer (A16W8) ──▶ a16w8.dlc
              ──▶ qnn-context-binary-generator ──▶ speed_cnn_a16w8_htpv75.bin
```

and the context binary is executed on the board with **`qnn-net-run`** (one clip
per call): each dumped clip is written as a raw `(1,3,8,112,112)` float32 tensor,
fed via an `input_list.txt`, and the fp32 `speed_kmh.raw` output is read back.
Those predictions (plus the per-inference latency parsed from
`qnn-profile-viewer`) were emitted in the same JSONL shape `replay_clips.py`
produces (`{"type":"infer","clip_id":N,"raw_kmh":…,"infer_ms":…}`) and joined
with the CPU log by `compare_runs.py`.

The same `qnn-net-run` wrapper is what `3-bluetooth-ftms/game_server.py --provider
qnn` runs live on the board (see that folder's README, "Running on the NPU").

> The A16W8 `.dlc` / HTP context binary and the matched QAIRT runtime are large
> Qualcomm artifacts prepared on a build server; they are **not** checked into
> git (same treatment as the `.onnx` model). This folder covers the on-device
> measurement and the comparison report — the committed result is
> [`cpu_vs_npu.md`](cpu_vs_npu.md) (machine-readable: [`cpu_vs_npu.json`](cpu_vs_npu.json)).
