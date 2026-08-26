#!/usr/bin/env python3
"""
replay_clips.py — run the clips dumped by `live_speed.py --dump-clips` back
through a model, on any execution provider, and record the prediction + latency
per clip.

This is the heart of the CPU-vs-NPU *numeric* comparison: the CPU session dumps
the exact (1,3,8,112,112) tensors it fed the model; here we replay those SAME
tensors through the NPU (or CPU again) so the only thing that can differ is the
engine — the walk, the lighting and the clip contents are held fixed.

    # CPU reference (float ONNX):
    python3 replay_clips.py --clips run_clips/ --model speed_cnn.onnx \
        --provider cpu  --out cpu.jsonl

    # NPU replay (ONNX Runtime QNN EP, on the board):
    python3 replay_clips.py --clips run_clips/ --model speed_cnn.onnx \
        --provider qnn  --out npu.jsonl

Then diff the two with compare_runs.py.

Providers
---------
  cpu   CPUExecutionProvider (float ONNX) — the reference.
  qnn   QNNExecutionProvider — ONNX Runtime's Qualcomm backend. Note this model's
        5D Conv3d ops fall back to CPU under the QNN EP, so the real NPU numbers
        in cpu_vs_npu.md come from the QAIRT context binary run with qnn-net-run
        (see this folder's README, "Notes on the NPU path"), not from this path.
"""

import argparse
import glob
import json
import os
import time

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

PROVIDER_MAP = {
    "cpu": "CPUExecutionProvider",
    "qnn": "QNNExecutionProvider",
}


def build_session(model_path, provider):
    if ort is None:
        raise SystemExit("onnxruntime not installed")
    ep = PROVIDER_MAP[provider]
    avail = ort.get_available_providers()
    if ep not in avail:
        raise SystemExit(f"{ep} not available in this onnxruntime build "
                         f"(have: {avail}). For --provider qnn you need an "
                         f"onnxruntime with the QNN execution provider.")
    return ort.InferenceSession(model_path, providers=[ep])


def main():
    ap = argparse.ArgumentParser(description="Replay dumped clips through a model")
    ap.add_argument("--clips", required=True, help="dir of clip_*.npy")
    ap.add_argument("--model", required=True, help="ONNX model path")
    ap.add_argument("--provider", default="cpu", choices=list(PROVIDER_MAP))
    ap.add_argument("--out", required=True, help="output JSONL of predictions")
    ap.add_argument("--warmup", type=int, default=3,
                    help="warmup inferences excluded from timing (default 3)")
    args = ap.parse_args()

    clips = sorted(glob.glob(os.path.join(args.clips, "clip_*.npy")))
    if not clips:
        raise SystemExit(f"no clip_*.npy in {args.clips}")
    sess = build_session(args.model, args.provider)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"{len(clips)} clips · provider={args.provider} · in={in_name} out={out_name}")

    # Warmup (first calls include graph finalize / allocation — not representative).
    if clips:
        w = np.load(clips[0]).astype(np.float32)
        for _ in range(max(0, args.warmup)):
            sess.run([out_name], {in_name: w})

    with open(args.out, "w", buffering=1) as fp:
        fp.write(json.dumps({"type": "header", "provider": args.provider,
                             "model": os.path.basename(args.model),
                             "n_clips": len(clips)}) + "\n")
        for path in clips:
            cid = int(os.path.basename(path)[5:11])   # clip_NNNNNN.npy
            clip = np.load(path).astype(np.float32)
            t = time.perf_counter()
            raw = float(sess.run([out_name], {in_name: clip})[0].flat[0])
            infer_ms = (time.perf_counter() - t) * 1000.0
            fp.write(json.dumps({"type": "infer", "clip_id": cid,
                                 "raw_kmh": round(max(0.0, raw), 4),
                                 "infer_ms": round(infer_ms, 3)}) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
