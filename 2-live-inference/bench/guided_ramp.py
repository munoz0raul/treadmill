#!/usr/bin/env python3
"""
guided_ramp.py — auto-stamps the panel-speed label on a fixed cadence so you
don't have to touch the browser during a bench recording. Run this ON THE BOARD
in a second terminal while live_speed.py is recording (--log/--dump-clips).

It POSTs each target speed to live_speed.py's /set_speed endpoint every HOLD_S
seconds and prints a live countdown. YOU keep the treadmill panel matching the
"NOW" speed it announces, and walk. That's the only manual part.

    0 → 8 km/h in 0.5 steps, 10 s each  (start with the belt stopped for 0 km/h)
"""
import argparse
import time
import urllib.request

def stamp(port, kmh):
    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{port}/set_speed?kmh={kmh}", timeout=3).read()
        return True
    except Exception as e:
        print(f"  ! could not stamp {kmh}: {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Guided speed-label ramp")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--hold", type=float, default=10.0, help="seconds per step")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--countdown", type=float, default=5.0,
                    help="seconds of lead-in before the ramp starts")
    args = ap.parse_args()

    speeds = []
    v = args.start
    while v <= args.stop + 1e-6:
        speeds.append(round(v, 1))
        v += args.step

    print(f"Guided ramp: {speeds[0]} → {speeds[-1]} km/h, "
          f"{args.hold:.0f}s each, {len(speeds)} steps "
          f"(~{len(speeds)*args.hold/60:.1f} min).")
    print("Keep the treadmill PANEL matching the NOW speed. Ctrl-C to abort.\n")
    for c in range(int(args.countdown), 0, -1):
        print(f"  starting in {c}s ... (step onto the belt, hold the rails)", end="\r")
        time.sleep(1)
    print(" " * 60, end="\r")

    for i, s in enumerate(speeds):
        stamp(args.port, s)
        nxt = speeds[i + 1] if i + 1 < len(speeds) else None
        for r in range(int(args.hold), 0, -1):
            tag = f"→ next {nxt}" if nxt is not None else "→ last step"
            print(f"  NOW {s:>4.1f} km/h   {r:>2}s left   {tag}     ", end="\r")
            time.sleep(1)
        print()
    print("\nRamp done. Stop the belt. Then Ctrl-C the live_speed.py recorder.")

if __name__ == "__main__":
    main()
