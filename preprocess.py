#!/usr/bin/env python3
"""
preprocess.py — Build RGB-clip cache from side.mp4 + speed_ble.jsonl

For each session we sample short RGB clips (T frames sampled by TIMESTAMP so one
clip always spans exactly CLIP_SPAN_S seconds ≈ one gait cycle, regardless of the
session's recording FPS). Each frame is cropped to a central horizontal window
(drops the sofa on the left and the shelf/plants on the right — pure background
noise) and resized to CLIP x CLIP. Label = mean belt speed over the clip's span.

Output stored as uint8 to keep the cache small; train.py converts to
float [0..1] in the DataLoader.

  dataset/cache_side.npz   keys:
    X          (N, 3, T, CLIP, CLIP) uint8   RGB clips
    y          (N,)                   float32 km/h
    session_id (N,)                   int32

Usage:
  python3 preprocess.py [--sessions s1 s2 ...] [--out dataset/cache_side.npz]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# ── hyperparameters ───────────────────────────────────────────────────────────
CLIP         = 112   # square input expected by mc3_18
T            = 8     # frames per clip
CLIP_SPAN_S  = 1.0   # real-time span of one clip, in SECONDS (≈ one gait cycle).
                     #   Frames are sampled by TIMESTAMP over this window, so every
                     #   clip covers the same real motion regardless of the session's
                     #   recording FPS. MUST equal CNN_CLIP_SPAN_S in pose_server.py.
CLIP_HOP_S   = 0.5   # seconds between successive clip start positions
CROP_X0      = 240   # central horizontal crop: keep x∈[240,1040) of 1280
CROP_X1      = 1040  #   → 800x720 region (full body incl. arms, no side clutter)
MAX_LAG      = 1.5   # max seconds between frame ts and nearest BLE sample
MIN_SPEED    = 0.0   # include stopped (0 km/h) — model must learn "stopped"
MAX_CLIPS_PER_BIN = 800   # balance: cap clips per 0.5 km/h bin to flatten histogram

DEFAULT_SESSIONS = [
    "dataset/session_20260805_222647",   # 0
    "dataset/session_20260805_230844",   # 1
    "dataset/session_20260805_232711",   # 2  ← different person
    "dataset/session_20260814_215452",   # 3
    "dataset/session_20260814_220628",   # 4
    "dataset/session_20260817_234515",   # 5  ← black outfit
    "dataset/session_20260818_000345",   # 6  ← orange outfit
    "dataset/session_20260818_001601",   # 7  ← adidas shorts  ← val
    "dataset/session_20260818_002822",   # 8  ← empty belt, 0 km/h
]


def load_ble(path):
    """Return sorted list of (ts, speed_kmh)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append((r["ts"], r["speed_kmh"]))
    return sorted(rows)


def nearest_speed(ble_rows, ts):
    """Binary-search nearest BLE sample.  Returns None if gap > MAX_LAG."""
    if not ble_rows:
        return None
    lo, hi = 0, len(ble_rows) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ble_rows[mid][0] < ts:
            lo = mid + 1
        else:
            hi = mid
    best = lo
    if lo > 0 and abs(ble_rows[lo-1][0] - ts) < abs(ble_rows[lo][0] - ts):
        best = lo - 1
    if abs(ble_rows[best][0] - ts) > MAX_LAG:
        return None
    return ble_rows[best][1]


def process_session(session_dir, verbose=True):
    video_path = os.path.join(session_dir, "side.mp4")
    ble_path   = os.path.join(session_dir, "speed_ble.jsonl")
    pose_path  = os.path.join(session_dir, "side.jsonl")   # for timestamps

    if not os.path.exists(video_path):
        print(f"  skip {session_dir}: no side.mp4"); return [], []
    if not os.path.exists(ble_path):
        print(f"  skip {session_dir}: no speed_ble.jsonl"); return [], []
    if not os.path.exists(pose_path):
        print(f"  skip {session_dir}: no side.jsonl"); return [], []

    ble_rows = load_ble(ble_path)

    frame_ts = []
    with open(pose_path) as f:
        for line in f:
            if line.strip():
                frame_ts.append(json.loads(line)["ts"])

    frame_speed = [nearest_speed(ble_rows, ts) for ts in frame_ts]

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f"  {os.path.basename(session_dir)}: {total} frames, "
              f"{len(frame_ts)} timestamps, {len(ble_rows)} BLE samples")

    # Read all frames: central crop → RGB → resize CLIP x CLIP
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        crop = bgr[:, CROP_X0:CROP_X1]                       # (720, 800, 3) BGR
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (CLIP, CLIP), interpolation=cv2.INTER_AREA)
        frames.append(small)
    cap.release()

    n = min(len(frames), len(frame_speed), len(frame_ts))
    frames = frames[:n]
    frame_speed = frame_speed[:n]
    frame_ts = frame_ts[:n]

    # Time-based clip sampling: for each window starting at t0 (stepping every
    # CLIP_HOP_S), pick T frames whose timestamps are nearest to T targets spread
    # uniformly across a CLIP_SPAN_S-second window. This mirrors pose_server's
    # _sample_clip exactly, so a clip covers the same real motion regardless of the
    # session's recording FPS (the bimodal-24/29.5-fps skew is what broke live).
    def nearest_frame_idx(t):
        """Index of the frame whose ts is closest to t (frame_ts is sorted)."""
        import bisect
        j = bisect.bisect_left(frame_ts, t)
        if j <= 0:
            return 0
        if j >= n:
            return n - 1
        return j if (frame_ts[j] - t) < (t - frame_ts[j - 1]) else j - 1

    X_list, y_list = [], []
    if n >= T:
        t_first, t_last = frame_ts[0], frame_ts[-1]
        t0 = t_first
        while t0 + CLIP_SPAN_S <= t_last + 1e-6:
            targets = [t0 + i * CLIP_SPAN_S / (T - 1) for i in range(T)]
            idxs = [nearest_frame_idx(t) for t in targets]
            speeds = [frame_speed[i] for i in idxs]
            if any(s is None for s in speeds):
                t0 += CLIP_HOP_S
                continue
            mean_speed = sum(speeds) / len(speeds)
            if mean_speed < MIN_SPEED:
                t0 += CLIP_HOP_S
                continue
            clip = np.stack([frames[i] for i in idxs], axis=0)   # (T, CLIP, CLIP, 3)
            clip = clip.transpose(3, 0, 1, 2)                     # (3, T, CLIP, CLIP)
            X_list.append(clip)
            y_list.append(mean_speed)
            t0 += CLIP_HOP_S

    if verbose and y_list:
        print(f"    → {len(X_list)} clips, speed {min(y_list):.1f}→{max(y_list):.1f} km/h")
    elif verbose:
        print(f"    → 0 clips")
    return X_list, y_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    parser.add_argument("--out", default="dataset/cache_side.npz")
    args = parser.parse_args()

    all_X, all_y, all_sid = [], [], []
    for sid, s in enumerate(args.sessions):
        X, y = process_session(s)
        all_X.extend(X)
        all_y.extend(y)
        all_sid.extend([sid] * len(X))

    if not all_X:
        print("No samples generated — check session paths.")
        sys.exit(1)

    # Balance: cap clips per 0.5 km/h bin so the histogram is flat.
    # This reduces shrinkage-to-mean (slope < 1) caused by bin imbalance.
    BIN = 0.5
    from collections import defaultdict
    import random as _rnd
    _rnd.seed(42)
    bins = defaultdict(list)
    for i, spd in enumerate(all_y):
        b = int(spd / BIN)   # floor bin index — each 0.5 km/h slot is separate
        bins[b].append(i)
    keep = []
    for b, idxs in sorted(bins.items()):
        if len(idxs) > MAX_CLIPS_PER_BIN:
            idxs = _rnd.sample(idxs, MAX_CLIPS_PER_BIN)
        keep.extend(idxs)
    keep.sort()
    all_X   = [all_X[i]   for i in keep]
    all_y   = [all_y[i]   for i in keep]
    all_sid = [all_sid[i] for i in keep]
    print(f"\nAfter balancing (cap {MAX_CLIPS_PER_BIN}/bin): {len(all_X)} clips kept")

    X   = np.stack(all_X, axis=0).astype(np.uint8)      # (N, 3, T, CLIP, CLIP)
    y   = np.array(all_y,  dtype=np.float32)
    sid = np.array(all_sid, dtype=np.int32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, session_id=sid)

    print(f"\nSaved {args.out}")
    print(f"  X shape: {X.shape}  dtype: {X.dtype}  ({X.nbytes/1e9:.2f} GB uncompressed)")
    print(f"  y shape: {y.shape}  range: {y.min():.1f}→{y.max():.1f} km/h")
    print(f"  mean={y.mean():.2f}  std={y.std():.2f}")

    from collections import Counter
    c = Counter(round(v) for v in y)
    print("\nSamples per speed (km/h):")
    for k in sorted(c):
        bar = "█" * (c[k] // 10)
        print(f"  {k:2d} km/h: {c[k]:5d}  {bar}")


if __name__ == "__main__":
    main()
