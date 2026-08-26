#!/usr/bin/env python3
"""
ftms_probe.py — a diagnostic FTMS *central* that mimics what a running game
(Zwift, Rouvy) does when it pairs a treadmill, and logs every GATT step so we
can see exactly where a peripheral (our board's ftms_peripheral.py) is rejected.

This is the mirror of the board side: the board is the FTMS peripheral, this is
the central that connects to it. Runs on the Mac via bleak (CoreBluetooth).

Sequence (this is roughly Zwift's treadmill pairing flow):
  scan → connect → discover services/chars/descriptors
       → read Fitness Machine Feature   (0x2ACC)
       → read Supported Speed Range     (0x2AD4)   [often missing → drop]
       → subscribe Treadmill Data       (0x2ACD, notify)
       → subscribe Fitness Machine Status(0x2ADA, notify)
       → enable indications on Control Pt(0x2AD9)
       → write Request Control (0x00)   → expect indication 80 00 01
       → write Start/Resume    (0x07)   → expect indication 80 07 01
       → watch Treadmill Data notifications, decode instantaneous speed

Every step prints PASS/FAIL. A missing characteristic or a bad handshake reply
is called out explicitly. Run:  python3 ftms_probe.py [--name "AI Treadmill"]
"""

import argparse
import asyncio
import struct
import sys
import time

from bleak import BleakScanner, BleakClient

FTMS_SERVICE = "00001826-0000-1000-8000-00805f9b34fb"
UUID_DATA    = "00002acd-0000-1000-8000-00805f9b34fb"   # Treadmill Data (notify)
UUID_FEATURE = "00002acc-0000-1000-8000-00805f9b34fb"   # Fitness Machine Feature (read)
UUID_SPEEDRG = "00002ad4-0000-1000-8000-00805f9b34fb"   # Supported Speed Range (read)
UUID_CP      = "00002ad9-0000-1000-8000-00805f9b34fb"   # Control Point (write+indicate)
UUID_STATUS  = "00002ada-0000-1000-8000-00805f9b34fb"   # Fitness Machine Status (notify)


def _ok(msg):   print(f"  \033[32mPASS\033[0m  {msg}")
def _bad(msg):  print(f"  \033[31mFAIL\033[0m  {msg}")
def _info(msg): print(f"        {msg}")


def decode_treadmill_data(data: bytes):
    """flags uint16 LE; if bit0 (More Data) clear, instantaneous speed is the
    first field, uint16 LE in 0.01 km/h. Mirrors the board's encoder."""
    if len(data) < 4:
        return None
    flags = int.from_bytes(data[0:2], "little")
    speed = int.from_bytes(data[2:4], "little") / 100.0
    return flags, speed


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="AI Treadmill", help="advertised device name to match")
    ap.add_argument("--scan", type=float, default=10.0, help="scan timeout seconds")
    ap.add_argument("--watch", type=float, default=15.0, help="seconds to watch speed notifications")
    args = ap.parse_args()

    print(f"[1] Scanning {args.scan}s for '{args.name}' (or FTMS 0x1826)…")
    target = None
    found = await BleakScanner.discover(timeout=args.scan, return_adv=True)
    for d, adv in found.values():
        uuids = [u.lower() for u in (adv.service_uuids or [])]
        if d.name == args.name or FTMS_SERVICE in uuids:
            _ok(f"found {d.address}  name={d.name!r}  rssi={adv.rssi}  adv_uuids={uuids}")
            target = d
            break
    if not target:
        _bad("device not found. Names seen: " +
             ", ".join(sorted({(d.name or '?') for d, _ in found.values()})))
        return 1

    print(f"[2] Connecting to {target.address}…")
    disconnected = asyncio.Event()

    def on_disconnect(_):
        print("  \033[33m!! peripheral disconnected us\033[0m")
        disconnected.set()

    async with BleakClient(target, disconnected_callback=on_disconnect) as c:
        _ok(f"connected = {c.is_connected}")

        # [3] Service / characteristic discovery -----------------------------
        print("[3] Discovering GATT services…")
        chars = {}
        ftms_present = False
        for s in c.services:
            is_ftms = s.uuid.lower() == FTMS_SERVICE
            ftms_present = ftms_present or is_ftms
            print(f"  SERVICE {s.uuid}{'   <-- FTMS' if is_ftms else ''}")
            for ch in s.characteristics:
                chars[ch.uuid.lower()] = ch
                descs = [d.uuid for d in ch.descriptors]
                cccd = any(d.uuid.lower().startswith("00002902") for d in ch.descriptors)
                print(f"    CHAR {ch.uuid}  props={ch.properties}"
                      f"{'  +CCCD' if cccd else ''}")
        if ftms_present:
            _ok("FTMS service 0x1826 present")
        else:
            _bad("FTMS service 0x1826 NOT exposed after connect (Zwift would drop here)")

        # [4] Mandatory-ish reads a game does during discovery ---------------
        print("[4] Reading discovery characteristics a game checks…")
        feat = chars.get(UUID_FEATURE)
        if feat and "read" in feat.properties:
            val = await c.read_gatt_char(feat)
            if len(val) >= 8:
                f1, f2 = struct.unpack("<II", val[:8])
                _ok(f"Feature 0x2ACC = {val.hex()}  (machine=0x{f1:08x} target=0x{f2:08x})")
                if f1 == 0 and f2 == 0:
                    _info("note: all-zero features — valid, but some games gate on bits here")
            else:
                _bad(f"Feature 0x2ACC too short: {val.hex()}")
        else:
            _bad("Feature 0x2ACC missing/not readable")

        rng = chars.get(UUID_SPEEDRG)
        if rng and "read" in rng.properties:
            val = await c.read_gatt_char(rng)
            if len(val) >= 6:
                lo, hi, step = struct.unpack("<HHH", val[:6])
                _ok(f"Supported Speed Range 0x2AD4 = min={lo/100} max={hi/100} step={step/100} km/h")
            else:
                _bad(f"Speed Range 0x2AD4 too short: {val.hex()}")
        else:
            _bad("Supported Speed Range 0x2AD4 MISSING  <-- prime suspect for Zwift drop")

        # [5] Subscribe to Treadmill Data ------------------------------------
        print("[5] Subscribing to Treadmill Data 0x2ACD…")
        last = {"t": 0.0, "n": 0}

        def on_data(_, data: bytearray):
            dec = decode_treadmill_data(bytes(data))
            last["n"] += 1
            now = time.time()
            if now - last["t"] > 0.9:  # throttle prints to ~1 Hz
                last["t"] = now
                if dec:
                    print(f"      DATA {bytes(data).hex()}  flags=0x{dec[0]:04x}  speed={dec[1]:.2f} km/h")
                else:
                    print(f"      DATA {bytes(data).hex()}  (too short)")

        data_ch = chars.get(UUID_DATA)
        if data_ch and ("notify" in data_ch.properties or "indicate" in data_ch.properties):
            await c.start_notify(data_ch, on_data)
            _ok("subscribed to Treadmill Data notifications")
        else:
            _bad("Treadmill Data 0x2ACD missing or not notifiable")

        status_ch = chars.get(UUID_STATUS)
        if status_ch and "notify" in status_ch.properties:
            await c.start_notify(status_ch, lambda _, d: print(f"      STATUS {bytes(d).hex()}"))
            _ok("subscribed to Fitness Machine Status notifications")
        else:
            _info("Fitness Machine Status 0x2ADA not subscribable (non-fatal)")

        # [6] Control Point handshake ----------------------------------------
        print("[6] Control Point handshake (Request Control → Start)…")
        cp = chars.get(UUID_CP)
        cp_resp = []

        def on_cp(_, data: bytearray):
            b = bytes(data)
            cp_resp.append(b)
            ok = len(b) >= 3 and b[0] == 0x80 and b[2] == 0x01
            tag = "OK (0x80 …01 Success)" if ok else "unexpected"
            print(f"      CP-IND {b.hex()}  -> {tag}")

        if not cp:
            _bad("Control Point 0x2AD9 missing — game cannot take control")
        else:
            can_ind = "indicate" in cp.properties
            can_wr  = "write" in cp.properties or "write-without-response" in cp.properties
            _info(f"CP props={cp.properties}")
            if can_ind:
                try:
                    await c.start_notify(cp, on_cp)   # bleak handles indicate too
                    _ok("enabled indications on Control Point")
                except Exception as e:
                    _bad(f"could not enable CP indications: {e}")
            else:
                _bad("Control Point has no 'indicate' property (game expects it)")

            if can_wr:
                async def cp_write(opcode, label):
                    cp_resp.clear()
                    await c.write_gatt_char(cp, bytes([opcode]), response=True)
                    await asyncio.sleep(1.0)
                    if cp_resp:
                        r = cp_resp[-1]
                        if len(r) >= 3 and r[0] == 0x80 and r[1] == opcode and r[2] == 0x01:
                            _ok(f"{label}: got Success indication {r.hex()}")
                        else:
                            _bad(f"{label}: reply {r.hex()} not 0x80 {opcode:02x} 01")
                    else:
                        _bad(f"{label}: NO indication received (game would drop here)")
                await cp_write(0x00, "Request Control (0x00)")
                await cp_write(0x07, "Start/Resume (0x07)")
            else:
                _bad("Control Point not writable")

        # [7] Watch the speed stream -----------------------------------------
        print(f"[7] Watching speed for {args.watch}s…")
        n0 = last["n"]
        try:
            await asyncio.wait_for(disconnected.wait(), timeout=args.watch)
            _bad("peripheral dropped the link during watch window")
        except asyncio.TimeoutError:
            pass
        got = last["n"] - n0
        if got > 0:
            _ok(f"received {got} speed notifications in {args.watch}s "
                f"(~{got/args.watch:.1f} Hz)")
        else:
            _bad("no speed notifications arrived")

        print("\n=== SUMMARY ===")
        print("If steps 3–6 all PASS, a well-behaved FTMS game should pair.")
        print("Most likely Zwift-drop cause = FAIL on Speed Range 0x2AD4 or CP indication.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        pass
