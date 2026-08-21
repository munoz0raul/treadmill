#!/usr/bin/env python3
"""
preprocess.py — build the RGB-clip training cache from acquired sessions.

Each session recorded by acquire.py is:
  side.mp4            raw side video
  speed_manual.jsonl  the hand-set belt speed timeline (step function)
  manifest.json       metadata (measured fps, etc.)

For each session we sample short RGB clips. A clip is T frames picked by
TIMESTAMP so it always spans exactly CLIP_SPAN_S seconds (≈ one gait cycle),
regardless of the session's recording FPS. Each frame is cropped to a central
horizontal window (drops room clutter on the sides, keeps the walker + belt) and
resized to CLIP×CLIP. The clip's label is the belt speed that was set during that
window, read from speed_manual.jsonl.

Two things make the labels trustworthy across sessions:
  1. Frames are sampled by timestamp, so mixed recording FPS is handled by
     construction — a 1-second clip is a 1-second clip whether the camera ran at
     24 or 30 fps.
  2. As a belt-and-braces measure we detect the dominant FPS across all sessions
     and, if a video's fps differs by more than FPS_TOLERANCE, we re-encode it to
     the common fps with ffmpeg before sampling (see --normalize-fps). This keeps
     one solid, consistent sampling rate across the whole dataset.

Output:
  dataset/cache_side.npz
    X          (N, 3, T, CLIP, CLIP) uint8   RGB clips
    y          (N,)                   float32 km/h
    session_id (N,)                   int32

Usage:
  python3 preprocess.py [--sessions dataset/session_* ...]
                        [--out dataset/cache_side.npz]
                        [--normalize-fps]
"""

import argparse
import bisect
import glob
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

# ── hyperparameters ───────────────────────────────────────────────────────────
CLIP         = 112   # square input expected by mc3_18
T            = 8     # frames per clip
CLIP_SPAN_S  = 1.0   # real-time span of one clip, in SECONDS (≈ one gait cycle).
                     #   Frames are sampled by TIMESTAMP over this window, so every
                     #   clip covers the same real motion regardless of the session's
                     #   recording FPS. MUST equal CNN_CLIP_SPAN_S in the live app.
CLIP_HOP_S   = 0.5   # seconds between successive clip start positions
CROP_X0      = 240   # central horizontal crop: keep x∈[240,1040) of 1280
CROP_X1      = 1040  #   → 800×720 region (full body incl. arms, no side clutter)
MIN_SPEED    = 0.0   # include stopped (0 km/h) — the model must learn "stopped"
MAX_CLIPS_PER_BIN = 800   # balance: cap clips per 0.5 km/h bin to flatten histogram
FPS_TOLERANCE = 0.5  # a video whose fps differs from the mode by more than this is
                     #   re-encoded to the common fps when --normalize-fps is set


def load_labels(path):
    """Return sorted list of (ts, speed_kmh) from a speed_manual.jsonl file."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append((r["ts"], r["speed_kmh"]))
    return sorted(rows)


def speed_at(label_rows, ts):
    """Step-function lookup: the speed set at or before `ts` (a set speed holds
    until the next change). Before the first row → the first row's speed."""
    if not label_rows:
        return None
    # rightmost row with row_ts <= ts
    lo, hi = 0, len(label_rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if label_rows[mid][0] <= ts:
            lo = mid + 1
        else:
            hi = mid
    idx = max(0, lo - 1)
    return label_rows[idx][1]


def video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 1e-3 else 0.0


def common_fps(sessions):
    """The dominant (mode, rounded to 0.5) fps across all sessions' side.mp4."""
    counts = Counter()
    for s in sessions:
        vp = os.path.join(s, "side.mp4")
        if os.path.exists(vp):
            fps = video_fps(vp)
            if fps:
                counts[round(fps * 2) / 2.0] += 1   # round to nearest 0.5
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def normalize_fps(video_path, target_fps):
    """Re-encode a video to `target_fps` with ffmpeg, in place (via a temp file).
    Frames are sampled by timestamp anyway, so this is a consistency safeguard —
    it keeps one solid frame rate across the dataset. Requires ffmpeg on PATH."""
    tmp = video_path + ".norm.mp4"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-r", str(target_fps),
           "-c:v", "mpeg4", "-q:v", "3", tmp]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        os.replace(tmp, video_path)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"    ! fps normalize skipped ({e.__class__.__name__}); "
              f"timestamp sampling still handles mixed fps")
        return False


def process_session(session_dir, verbose=True):
    video_path = os.path.join(session_dir, "side.mp4")
    label_path = os.path.join(session_dir, "speed_manual.jsonl")

    if not os.path.exists(video_path):
        print(f"  skip {session_dir}: no side.mp4"); return [], []
    if not os.path.exists(label_path):
        print(f"  skip {session_dir}: no speed_manual.jsonl"); return [], []

    label_rows = load_labels(label_path)
    if not label_rows:
        print(f"  skip {session_dir}: empty label timeline"); return [], []

    # The label timeline is anchored in wall-clock epoch time (acquire.py stamps
    # each speed change and start/stop). Frame timestamps are reconstructed from
    # the session start (first label row) plus frame_index / fps — a uniform grid
    # that shares the same clock as the labels.
    fps = video_fps(video_path) or 30.0
    t_session_start = label_rows[0][0]

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(f"  {os.path.basename(session_dir)}: {total} frames @ {fps:.2f} fps, "
              f"{len(label_rows)} label rows")

    frames, frame_ts = [], []
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        crop = bgr[:, CROP_X0:CROP_X1]
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (CLIP, CLIP), interpolation=cv2.INTER_AREA)
        frames.append(small)
        frame_ts.append(t_session_start + idx / fps)
        idx += 1
    cap.release()

    n = len(frames)
    frame_speed = [speed_at(label_rows, t) for t in frame_ts]

    def nearest_frame_idx(t):
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
    parser.add_argument("--sessions", nargs="+",
                        default=sorted(glob.glob("dataset/session_*")),
                        help="session dirs (default: dataset/session_*)")
    parser.add_argument("--out", default="dataset/cache_side.npz")
    parser.add_argument("--normalize-fps", action="store_true",
                        help="re-encode any outlier video to the dataset's common "
                             "fps before sampling (needs ffmpeg on PATH)")
    args = parser.parse_args()

    sessions = args.sessions
    if not sessions:
        print("No sessions found — pass --sessions or record some with acquire.py.")
        sys.exit(1)

    # ── FPS report + optional normalization ──────────────────────────────────
    target = common_fps(sessions)
    print(f"Common (dominant) fps across sessions: {target}")
    for s in sessions:
        vp = os.path.join(s, "side.mp4")
        if not os.path.exists(vp):
            continue
        fps = video_fps(vp)
        flag = ""
        if target and abs(fps - target) > FPS_TOLERANCE:
            flag = "  <-- outlier"
            if args.normalize_fps:
                print(f"  {os.path.basename(s)}: {fps:.2f} fps → normalizing to {target}")
                normalize_fps(vp, target)
                flag = "  (normalized)"
        print(f"  {os.path.basename(s)}: {fps:.2f} fps{flag}")

    all_X, all_y, all_sid = [], [], []
    for sid, s in enumerate(sessions):
        X, y = process_session(s)
        all_X.extend(X)
        all_y.extend(y)
        all_sid.extend([sid] * len(X))

    if not all_X:
        print("No samples generated — check session paths.")
        sys.exit(1)

    # Balance: cap clips per 0.5 km/h bin so the histogram is flat. This reduces
    # shrinkage-to-mean (slope < 1) caused by bin imbalance.
    BIN = 0.5
    import random as _rnd
    _rnd.seed(42)
    bins = defaultdict(list)
    for i, spd in enumerate(all_y):
        b = int(spd / BIN)
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

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, session_id=sid)

    print(f"\nSaved {args.out}")
    print(f"  X shape: {X.shape}  dtype: {X.dtype}  ({X.nbytes/1e9:.2f} GB uncompressed)")
    print(f"  y shape: {y.shape}  range: {y.min():.1f}→{y.max():.1f} km/h")
    print(f"  mean={y.mean():.2f}  std={y.std():.2f}")

    c = Counter(round(v) for v in y)
    print("\nSamples per speed (km/h):")
    for k in sorted(c):
        bar = "#" * (c[k] // 10)
        print(f"  {k:2d} km/h: {c[k]:5d}  {bar}")


if __name__ == "__main__":
    main()
