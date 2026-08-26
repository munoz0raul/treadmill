# CPU vs NPU — treadmill speed model

- clips compared: **436**
- CPU log: `cpu_run.jsonl` (provider=cpu, model=speed_cnn.onnx)
- NPU log: `npu.jsonl` (provider=qnn_htp_v75)
- model sha256: `f97f3afacd32a701…`

## 1. Numeric fidelity (NPU vs CPU, identical clips)

- **MAE(NPU − CPU) = 0.0536 km/h**
- max |NPU − CPU|  = 0.3345 km/h

_This is the quantization cost: how far A16W8 on the NPU drifts from the float CPU output on the very same input._

## 2. Accuracy vs panel ground truth

| engine | MAE vs panel (km/h) |
|---|---|
| CPU (float) | 2.314 |
| NPU (A16W8) | 2.335 |

### Per speed band (MAE vs panel, km/h)

| band km/h | CPU | NPU |
|---|---|---|
| 0-1 | 0.227 | 0.234 |
| 1-2 | 0.228 | 0.216 |
| 2-3 | 0.358 | 0.345 |
| 3-4 | 0.300 | 0.345 |
| 4-5 | 0.543 | 0.617 |
| 5-6 | 0.368 | 0.447 |
| 6-7 | 0.301 | 0.347 |
| 7-8 | 0.131 | 0.181 |
| 8-9 | 6.374 | 6.368 |

## 3. Performance (per-inference latency)

| engine | mean ms | p50 ms | p95 ms | min | max |
|---|---|---|---|---|---|
| CPU | 629.93 | 623.90 | 692.82 | 567.63 | 808.10 |
| NPU | 26.56 | 26.55 | 27.25 | 25.53 | 28.39 |

- **p50 speedup (CPU/NPU): 23.5×**

## Notes on reading these numbers

- **The headline is fidelity + speedup.** Numeric fidelity (§1) and performance
  (§3) are the clean engine-vs-engine results: A16W8 on the Hexagon NPU drifts
  only **0.054 km/h** on average from the float CPU output on the *identical*
  tensor, while running **23.5× faster** (624 ms → 26.6 ms per inference) with a
  very tight spread (25.5–28.4 ms). Both are ground-truth-independent, so no
  recording artifact can distort them.
- **The `8-9` band and the overall "accuracy vs panel" (2.31 / 2.33) are inflated
  by a stale-label tail**, not an engine effect. At the end of the ramp the
  operator stopped the belt but left the panel label stamped at 8.0 km/h: 97 of
  the 123 `set_kmh=8.0` clips actually show a stopped/slowing belt (the last six
  predict 0.0). Those clips punish *both* engines identically, which is why CPU
  and NPU move together (the CPU−NPU gap stays ~0.05 km/h even there). The
  meaningful accuracy view is the **0-1 … 7-8 bands**, where both engines sit
  under ~0.7 km/h and track each other within 0.05–0.08 km/h.
- **What this demonstrates:** the mc3_18 3D-CNN — which ONNX Runtime's QNN
  execution provider *cannot* place on the HTP (5D Conv3d falls back to CPU) —
  does lower fully onto the Hexagon NPU via the QAIRT context-binary path
  (converter → A16W8 quantizer → HTP V75 context binary → `qnn-net-run`), at
  near-float accuracy and ~24× the throughput.
