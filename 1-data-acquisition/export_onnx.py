#!/usr/bin/env python3
"""
export_onnx.py — Export trained SpeedCNN to ONNX + TFLite.

Usage:
  python3 export_onnx.py [--ckpt models/speed_cnn.pt] [--out models/speed_cnn.onnx]

Outputs:
  models/speed_cnn.onnx        — ONNX opset 17, input (1, 3, T, 112, 112)
  models/speed_cnn_verify.json — numeric verification vs PyTorch
"""

import argparse
import json
import os

import numpy as np
import torch

from model import SpeedCNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="models/speed_cnn.pt")
    parser.add_argument("--out",  default="models/speed_cnn.onnx")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved_args = ckpt.get("args", {})
    n_frames = saved_args.get("n_frames", 8)

    model = SpeedCNN(n_frames=n_frames, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  val_mae={ckpt['val_mae']:.3f} km/h")

    # dummy input: (1, 3, T, 112, 112)  RGB clip in [0..1]
    dummy = torch.zeros(1, 3, n_frames, 112, 112)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        opset_version=17,
        input_names=["frames"],
        output_names=["speed_kmh"],
        dynamic_axes={"frames": {0: "batch"}, "speed_kmh": {0: "batch"}},
        dynamo=False,   # legacy exporter: single self-contained .onnx (no external
                        #   .onnx.data sidecar) — that's what the board loads.
    )
    print(f"Exported: {args.out}")

    # ── verify with onnxruntime ───────────────────────────────────────────────
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        x_np = np.random.rand(4, 3, n_frames, 112, 112).astype(np.float32)
        x_pt = torch.from_numpy(x_np)
        with torch.no_grad():
            pt_out = model(x_pt).numpy()
        ort_out = sess.run(["speed_kmh"], {"frames": x_np})[0]
        max_diff = float(np.abs(pt_out - ort_out).max())
        print(f"ONNX verify: max_diff={max_diff:.6f}  {'✓ OK' if max_diff < 1e-3 else '✗ LARGE DIFF'}")
        verify = {"max_diff": max_diff, "ok": max_diff < 1e-3,
                  "pt_sample": pt_out[:2].tolist(), "ort_sample": ort_out[:2].tolist()}
        with open(args.out.replace(".onnx", "_verify.json"), "w") as f:
            json.dump(verify, f, indent=2)
    except ImportError:
        print("onnxruntime not installed — skipping numeric verify")

    # ── TFLite conversion (optional, requires ai-edge-torch) ─────────────────
    try:
        import ai_edge_torch
        print("Converting to TFLite via ai_edge_torch …")
        sample = (dummy,)
        edge_model = ai_edge_torch.convert(model.eval(), sample)
        tflite_path = args.out.replace(".onnx", ".tflite")
        edge_model.export(tflite_path)
        print(f"TFLite exported: {tflite_path}")
    except ImportError:
        print("ai_edge_torch not available — skipping TFLite export")
        print("To convert manually:")
        print(f"  pip install onnx2tf && onnx2tf -i {args.out} -o models/")


if __name__ == "__main__":
    main()
