#!/usr/bin/env python3
"""
video_overlay.py — Render a session video with a small speed table overlaid.

Reads a recorded session (side.mp4 + side.jsonl + speed_ble.jsonl +
speed_ai.jsonl) and writes a NEW mp4 with a compact corner table showing three
speeds, aligned per-frame by real capture timestamp:

  Suggested       — the guided AutoSession cue (what we ask for every 30 s)
  Treadmill Real  — the true belt speed (BLE ground truth)
  AI              — the live CNN estimate

The source video is written at a nominal 30 fps header but the real capture
rate was ~18 fps, so we drive the overlay off the per-frame timestamps in
side.jsonl and emit the output at the measured real-time fps. The original
side.mp4 is never modified — output goes to <session>/side_overlay.mp4.

Usage:
  python3 video_overlay.py [--session dataset/session_20260818_202905]
                           [--out side_overlay.mp4]
"""

import argparse
import bisect
import json
import os

import cv2
import numpy as np


# ── Guided AutoSession cue schedule for session_20260818_202905 ──────────────
# (seconds after start_ts, suggested km/h) parsed from the board capture log.
# Before the first cue the belt is in the 8 s "get ready" warmup.
START_TS   = 1787084945.7149
WARMUP_END = 8.019
CUES = [
    (8.019,   0.0), (38.056,  0.5), (68.139,  1.0), (98.204,  1.5),
    (128.254, 2.0), (158.334, 2.5), (188.434, 3.0), (218.512, 3.5),
    (248.573, 4.0), (278.655, 4.5), (308.713, 5.0), (338.765, 5.5),
    (368.819, 6.0), (398.905, 6.5), (428.984, 7.0), (459.059, 7.5),
    (489.195, 8.0),
]
CUE_TS     = [START_TS + d for d, _ in CUES]
CUE_SPEED  = [s for _, s in CUES]


def load_jsonl_speed(path):
    """Return (ts_list, speed_list) sorted by ts from a {ts, speed_kmh} log."""
    ts, sp = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ts.append(d["ts"])
            sp.append(d["speed_kmh"])
    order = sorted(range(len(ts)), key=lambda i: ts[i])
    return [ts[i] for i in order], [sp[i] for i in order]


def load_frame_ts(path):
    """Return per-frame capture timestamps from side.jsonl (one per frame)."""
    ts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts.append(json.loads(line)["ts"])
    return ts


def nearest_hold(ts_arr, sp_arr, t):
    """Most recent sample at or before t (step/hold). None if before first."""
    if not ts_arr or t < ts_arr[0]:
        return None
    i = bisect.bisect_right(ts_arr, t) - 1
    return sp_arr[i]


def suggested_at(t):
    """Step-function suggested speed; None during the pre-first-cue warmup."""
    if t < CUE_TS[0]:
        return None
    i = bisect.bisect_right(CUE_TS, t) - 1
    return CUE_SPEED[i]


def draw_table(frame, suggested, real, ai):
    """Draw a small semi-transparent speed table in the top-left corner."""
    rows = [
        ("Suggested",      suggested, (120, 200, 255)),   # amber-ish
        ("Treadmill Real", real,      (255, 207,   0)),   # cyan (BGR)
        ("AI",             ai,        ( 80, 235, 120)),   # green
    ]

    pad, line_h = 14, 34
    label_x, value_x = pad + 8, pad + 250
    title_h = 30
    w = 340
    h = title_h + line_h * len(rows) + pad
    x0, y0 = 18, 18

    # translucent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "SPEED  (km/h)", (x0 + 8, y0 + 22),
                font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    y = y0 + title_h + 24
    for label, val, color in rows:
        cv2.putText(frame, label, (x0 + label_x - pad, y),
                    font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        txt = "--" if val is None else f"{val:.1f}"
        cv2.putText(frame, txt, (x0 + value_x - pad, y),
                    font, 0.75, color, 2, cv2.LINE_AA)
        y += line_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="dataset/session_20260818_202905")
    ap.add_argument("--out", default="side_overlay.mp4")
    args = ap.parse_args()

    sess = args.session
    mp4_in   = os.path.join(sess, "side.mp4")
    out_path = os.path.join(sess, args.out)

    frame_ts = load_frame_ts(os.path.join(sess, "side.jsonl"))
    ble_ts, ble_sp = load_jsonl_speed(os.path.join(sess, "speed_ble.jsonl"))
    ai_ts,  ai_sp  = load_jsonl_speed(os.path.join(sess, "speed_ai.jsonl"))
    print(f"frames(jsonl)={len(frame_ts)}  ble={len(ble_sp)}  ai={len(ai_sp)}")

    cap = cv2.VideoCapture(mp4_in)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {mp4_in}")
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Real-time fps from the actual capture span (nominal 30 fps header lies).
    span = frame_ts[-1] - frame_ts[0] if len(frame_ts) > 1 else 1.0
    n = min(n_vid, len(frame_ts))
    fps_out = round((n - 1) / span, 2) if span > 0 else 30.0
    print(f"video frames={n_vid}  {W}x{H}  real span={span:.1f}s  "
          f"out fps={fps_out}  -> {out_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps_out, (W, H))
    if not vw.isOpened():
        raise SystemExit("VideoWriter failed to open (mp4v)")

    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_ts[i]
        draw_table(frame,
                   suggested_at(t),
                   nearest_hold(ble_ts, ble_sp, t),
                   nearest_hold(ai_ts,  ai_sp,  t))
        vw.write(frame)
        i += 1
        if i % 500 == 0:
            print(f"  {i}/{n} frames")

    cap.release()
    vw.release()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Done: {out_path}  ({i} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
