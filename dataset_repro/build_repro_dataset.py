#!/usr/bin/env python3
"""
build_repro_dataset.py — rebuild the training dataset in the DOCUMENTED format.

The raw sessions in dataset/ were recorded with the *old* method:
  side.mp4          (header fps lies: always 30, real capture 18-30 fps)
  side.jsonl        (pose keypoints WITH true per-frame wall-clock timestamps)
  speed_ble.jsonl   (noisy belt speed read over Bluetooth from the treadmill)

The documented, reproducible format (what acquire.py would produce today) is:
  side.mp4            raw side video at an HONEST constant fps
  speed_manual.jsonl  the SCREEN / setpoint speed as a clean step function
  manifest.json       metadata (measured fps == header fps)

This script converts every session into that format, writing to OUT_ROOT:

  1. LABEL (screen speed, not Bluetooth speed):
     Snap each BLE reading to the nearest 0.5 km/h, merge into held segments,
     keep only segments that hold >= MIN_HOLD_S (the plateaus the user dialled
     in) plus any 0 km/h rests. Emit one speed_manual.jsonl row at each kept
     segment's start. This reconstructs the value shown on the treadmill's
     screen and drops the brief ramp transients between setpoints.

  2. FPS (honest, constant, common across the dataset):
     Frame times come from side.jsonl (the real clock). Resample each video
     onto a uniform TARGET_FPS grid (nearest source frame by timestamp) and
     write it with a truthful header. After this, preprocess.py's
     `t_session_start + idx / fps` reconstruction is exactly correct, and every
     session shares one frame rate.

Usage:
  python3 build_repro_dataset.py                 # all dataset/session_*
  python3 build_repro_dataset.py --target-fps 30 --out dataset_repro
"""

import argparse
import glob
import json
import os
import sys

import cv2

# ── reconstruction parameters ──────────────────────────────────────────────
TARGET_FPS   = 30.0   # one honest constant frame rate for the whole dataset
MIN_HOLD_S   = 6.0    # a snapped speed must hold this long to count as a
                      #   dialled-in setpoint (drops ramp transients)
MIN_REST_S   = 2.0    # keep 0 km/h rests at least this long (the "stopped" label)


def snap_half(v):
    """Snap a speed to the nearest 0.5 km/h (the screen's resolution)."""
    return round(v * 2) / 2.0


def load_frame_timestamps(side_jsonl):
    """True per-frame wall-clock timestamps, in capture order."""
    ts = []
    with open(side_jsonl) as f:
        for line in f:
            if line.strip():
                ts.append(json.loads(line)["ts"])
    return ts


def build_manual_labels(ble_jsonl, session_start):
    """Reconstruct the screen-speed step function from the noisy BLE trace.

    Returns a list of (ts, speed_kmh) rows: one per kept segment start, always
    anchored with a row at session_start so preprocess.py can label every frame.
    """
    rows = []
    with open(ble_jsonl) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append((r["ts"], snap_half(r["speed_kmh"])))
    if not rows:
        return []

    # merge consecutive equal snapped values into held segments
    segs = []  # [value, start_ts, end_ts]
    cv, st, lt = rows[0][1], rows[0][0], rows[0][0]
    for ts, v in rows[1:]:
        if v != cv:
            segs.append([cv, st, lt])
            cv, st = v, ts
        lt = ts
    segs.append([cv, st, lt])

    # keep dialled-in plateaus (and real rests); drop brief ramp transients
    kept = []
    for v, s, e in segs:
        dur = e - s
        if v == 0.0:
            if dur >= MIN_REST_S:
                kept.append((s, v))
        elif dur >= MIN_HOLD_S:
            kept.append((s, v))

    if not kept:
        return []

    # anchor a row at the true session start so no frame is left unlabeled
    labels = []
    first_val = kept[0][1] if kept[0][0] <= session_start + 1e-6 else 0.0
    labels.append((round(session_start, 4), first_val))
    for s, v in kept:
        if s > session_start + 1e-6:
            labels.append((round(s, 4), v))

    # collapse any now-adjacent duplicates
    dedup = [labels[0]]
    for ts, v in labels[1:]:
        if v != dedup[-1][1]:
            dedup.append((ts, v))
    return dedup


def resample_video(src_mp4, frame_ts, dst_mp4, target_fps):
    """Re-encode src_mp4 onto a uniform target_fps grid using the real frame
    timestamps, so the output plays at an honest constant fps over the same
    real duration. Returns (out_frames, real_duration_s, in_frames).

    Streams the source one frame at a time (never holds the whole video in RAM).
    Target times and source frames are both monotonic in time, so a two-pointer
    walk keeps only the current source frame resident: for each uniform target
    time we advance the source cursor while the *next* source frame is closer,
    then emit the current one (re-emitting it when up-sampling)."""
    n_in = int(cv2.VideoCapture(src_mp4).get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(len(frame_ts), n_in) if n_in > 0 else len(frame_ts)
    if n < 2:
        return 0, 0.0, n_in
    ts = frame_ts[:n]
    t0, t1 = ts[0], ts[-1]
    real_dur = t1 - t0
    n_out = max(2, int(round(real_dur * target_fps)))

    cap = cv2.VideoCapture(src_mp4)
    ok, cur = cap.read()          # current source frame (index j)
    if not ok:
        cap.release()
        return 0, 0.0, n_in
    j = 0
    nxt_ok, nxt = cap.read()      # look-ahead frame (index j+1), read on demand

    h, w = cur.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(dst_mp4, fourcc, target_fps, (w, h))

    n_written = 0
    for k in range(n_out):
        target = t0 + k / target_fps
        # advance while the next source frame is at least as close to target
        while (nxt_ok and j + 1 < n
               and abs(ts[j + 1] - target) <= abs(ts[j] - target)):
            cur = nxt
            j += 1
            nxt_ok, nxt = cap.read()
        vw.write(cur)
        n_written += 1
    vw.release()
    cap.release()
    return n_written, real_dur, n_in


def process(session_dir, out_root, target_fps):
    name = os.path.basename(session_dir)
    side_mp4   = os.path.join(session_dir, "side.mp4")
    side_jsonl = os.path.join(session_dir, "side.jsonl")
    ble_jsonl  = os.path.join(session_dir, "speed_ble.jsonl")

    for p in (side_mp4, side_jsonl, ble_jsonl):
        if not os.path.exists(p):
            print(f"  skip {name}: missing {os.path.basename(p)}")
            return False

    frame_ts = load_frame_timestamps(side_jsonl)
    if len(frame_ts) < 2:
        print(f"  skip {name}: no frame timestamps")
        return False
    session_start = frame_ts[0]

    labels = build_manual_labels(ble_jsonl, session_start)
    if not labels:
        print(f"  skip {name}: no stable plateaus in BLE trace")
        return False

    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    dst_mp4 = os.path.join(out_dir, "side.mp4")
    n_out, real_dur, n_in = resample_video(side_mp4, frame_ts, dst_mp4, target_fps)
    if n_out == 0:
        print(f"  skip {name}: video too short")
        return False

    # speed_manual.jsonl — the documented screen-speed step function
    with open(os.path.join(out_dir, "speed_manual.jsonl"), "w") as f:
        for ts, v in labels:
            f.write(json.dumps({"ts": ts, "speed_kmh": v}) + "\n")

    speeds = sorted({v for _, v in labels})
    manifest = {
        "session": name,
        "start_ts": round(session_start, 4),
        "end_ts": round(session_start + real_dur, 4),
        "duration_s": round(real_dur, 2),
        "camera": {
            "label": "side",
            "video_frames": n_out,
            "video_size": [1280, 720],
            "video_fps": target_fps,
        },
        "label_file": "speed_manual.jsonl",
        "label_source": "screen setpoint (0.5 km/h steps)",
        "label_rows": len(labels),
        "speeds_kmh": speeds,
        "clock": "time.time()",
        "note": ("side.mp4 re-encoded to an honest constant fps from the real "
                 "per-frame timestamps; speed_manual.jsonl is the screen speed "
                 "(plateau-snapped from the belt trace), not the Bluetooth reading"),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    in_fps = n_in / real_dur if real_dur else 0
    print(f"  {name}: {n_in} frames @ {in_fps:.1f} fps  ->  {n_out} @ {target_fps:.0f} fps "
          f"({real_dur:.0f}s) | {len(labels)} labels {speeds[0]:.1f}-{speeds[-1]:.1f} km/h")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+",
                    default=sorted(glob.glob("dataset/session_*")))
    ap.add_argument("--out", default="dataset_repro")
    ap.add_argument("--target-fps", type=float, default=TARGET_FPS)
    args = ap.parse_args()

    if not args.sessions:
        print("No sessions found.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    print(f"Rebuilding {len(args.sessions)} sessions -> {args.out}/  "
          f"@ {args.target_fps:.0f} fps\n")
    ok = 0
    for s in args.sessions:
        if process(s, args.out, args.target_fps):
            ok += 1
    print(f"\nDone: {ok}/{len(args.sessions)} sessions written to {args.out}/")


if __name__ == "__main__":
    main()
