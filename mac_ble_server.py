#!/usr/bin/env python3
"""
mac_ble_server.py — Treadmill BLE bridge + speed control.

Usage:
    ~/.local/bin/uv --system-certs run --with bleak python3 mac_ble_server.py

Endpoints:
  GET  /            → web control UI
  GET  /speed       → {"speed_kmh":3.0,"ts":...,"connected":true,"samples":42}
  GET  /status      → full status + control + ramp state
  POST /speed/set   → {"speed_kmh":3.0}
  POST /ramp/start  → {"start_kmh":1.0,"end_kmh":9.0,"step_kmh":0.5,"dwell_s":10}
  POST /ramp/stop   → cancel auto-ramp
  GET  /ramp/status → ramp state JSON
"""

import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from bleak import BleakClient, BleakScanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FTMS_SERVICE   = "00001826-0000-1000-8000-00805f9b34fb"
UUID_DATA      = "00002acd-0000-1000-8000-00805f9b34fb"
UUID_CP        = "00002ad9-0000-1000-8000-00805f9b34fb"
TREADMILL_NAME = "TRMZU"
PORT           = 8765

RESULT = {0x01: "Success", 0x02: "Op Code not supported",
          0x03: "Invalid Parameter", 0x04: "Operation Failed",
          0x05: "Control Not Permitted"}

# ── shared state ──────────────────────────────────────────────────────────────

_lock = threading.Lock()

_state = {
    "speed_kmh": None, "ts": None,
    "connected": False, "samples": 0,
    "control_granted": False,
}

_ramp = {
    "running": False,
    "current_kmh": None, "end_kmh": None,
    "step_index": 0, "total_steps": 0,
    "dwell_s": 10,
}

_ble_loop:      asyncio.AbstractEventLoop = None
_ble_client:    BleakClient               = None
_control_granted = False
_cp_future:     asyncio.Future            = None
_cp_lock:       asyncio.Lock              = None
_ramp_handle    = None
_keepalive_handle = None
_target_kmh     = None   # current commanded speed (None = not controlling)


def _upd(**kw):
    with _lock: _state.update(kw)

def _upd_ramp(**kw):
    with _lock: _ramp.update(kw)

def _get_state():
    with _lock: return dict(_state)

def _get_ramp():
    with _lock: return dict(_ramp)


# ── BLE write helpers (asyncio context) ──────────────────────────────────────

async def _write_cp_wait(payload: bytes, timeout=5.0):
    global _cp_future
    loop = asyncio.get_event_loop()
    _cp_future = loop.create_future()
    await _ble_client.write_gatt_char(UUID_CP, payload, response=True)
    try:
        _req, code = await asyncio.wait_for(asyncio.shield(_cp_future), timeout)
        return code == 0x01, RESULT.get(code, f"0x{code:02x}")
    except asyncio.TimeoutError:
        return False, "timeout"


async def _ensure_control():
    global _control_granted
    if _control_granted:
        return True, "already granted"
    ok, msg = await _write_cp_wait(bytes([0x00]))
    if ok:
        _control_granted = True
        _upd(control_granted=True)
        log.info("Control granted")
    else:
        log.warning("Control request failed: %s", msg)
    return ok, msg


async def _keepalive_loop():
    """Re-assert Set Target Speed periodically to HOLD the setpoint.

    This treadmill drifts back to its 1.0 km/h minimum a few seconds after
    reaching a commanded speed if left alone (the official app holds speed by
    continuously re-asserting the target). Cadence 2.0s: slow enough to let the
    belt reach target, fast enough to counter the decay. Encoding ×10 (0.1 km/h),
    the confirmed-correct SET unit. Never re-requests control (0x00) — that
    resets the setpoint toward 1.0."""
    global _target_kmh
    while True:
        await asyncio.sleep(2.0)
        with _lock:
            spd = _target_kmh
        if spd is None or _ble_client is None or not _ble_client.is_connected:
            continue
        if _cp_lock is None:
            continue
        try:
            async with _cp_lock:
                speed_int = round(max(0.0, min(spd, 25.0)) * 10)
                payload = bytes([0x02]) + speed_int.to_bytes(2, "little")
                await _ble_client.write_gatt_char(UUID_CP, payload, response=True)
                log.info("Keepalive re-assert: %.1f km/h", spd)
        except Exception as e:
            log.warning("Keepalive error: %s", e)


async def set_speed_async(speed_kmh: float):
    global _cp_lock, _target_kmh
    if _ble_client is None or not _ble_client.is_connected:
        return False, "not connected"
    if _cp_lock is None:
        _cp_lock = asyncio.Lock()
    async with _cp_lock:
        ok, msg = await _ensure_control()
        if not ok:
            return False, f"control failed: {msg}"
        # ×10 (0.1 km/h units) for SET Target Speed. CONFIRMED LIVE 2026-08-14:
        # commanding 2 with ×100 (value 200) drove the belt to 20 km/h on both
        # the panel and the BLE readback — 10× too high. So SET unit = 0.1 km/h.
        # NOTE the asymmetry: SET is ×10, but READBACK (0x2ACD) is ×100 (÷100 to decode).
        speed_int = round(max(0.0, min(speed_kmh, 25.0)) * 10)
        payload = bytes([0x02]) + speed_int.to_bytes(2, "little")
        ok, msg = await _write_cp_wait(payload)
        if ok:
            log.info("Speed set → %.2f km/h (0x%04x, ×10)", speed_kmh, speed_int)
            with _lock:
                _target_kmh = speed_kmh if speed_kmh > 0 else None
        return ok, msg


def set_speed_sync(speed_kmh: float):
    """Call from HTTP handler thread."""
    if _ble_loop is None:
        return False, "BLE not ready"
    try:
        fut = asyncio.run_coroutine_threadsafe(set_speed_async(speed_kmh), _ble_loop)
        return fut.result(timeout=8.0)
    except Exception as e:
        return False, str(e)


# ── Auto-ramp ─────────────────────────────────────────────────────────────────

async def _ramp_task(start_kmh, end_kmh, step_kmh, dwell_s):
    global _ramp_handle
    speeds = []
    s = start_kmh
    while s <= end_kmh + 0.001:
        speeds.append(round(s, 1))
        s = round(s + step_kmh, 1)

    _upd_ramp(running=True, end_kmh=end_kmh, dwell_s=dwell_s,
              step_index=0, total_steps=len(speeds), current_kmh=None)
    try:
        for i, spd in enumerate(speeds):
            _upd_ramp(step_index=i, current_kmh=spd)
            ok, msg = await set_speed_async(spd)
            log.info("Ramp %d/%d → %.1f km/h: %s", i + 1, len(speeds), spd, msg)
            await asyncio.sleep(dwell_s)
    except asyncio.CancelledError:
        log.info("Ramp cancelled")
    finally:
        _upd_ramp(running=False)
        _ramp_handle = None


async def start_treadmill_async():
    if _ble_client is None or not _ble_client.is_connected:
        return False, "not connected"
    if _cp_lock is None:
        return False, "cp not ready"
    async with _cp_lock:
        ok, msg = await _ensure_control()
        if not ok:
            return False, f"control failed: {msg}"
        ok, msg = await _write_cp_wait(bytes([0x07]))
        if ok:
            log.info("Treadmill started (0x07)")
        return ok, msg


def start_treadmill_sync():
    if _ble_loop is None:
        return False, "BLE not ready"
    try:
        fut = asyncio.run_coroutine_threadsafe(start_treadmill_async(), _ble_loop)
        return fut.result(timeout=8.0)
    except Exception as e:
        return False, str(e)


async def stop_treadmill_async():
    """Stop belt by setting speed to 0 (0x08 not supported by this treadmill)."""
    return await set_speed_async(0.0)


def stop_treadmill_sync():
    if _ble_loop is None:
        return False, "BLE not ready"
    try:
        fut = asyncio.run_coroutine_threadsafe(stop_treadmill_async(), _ble_loop)
        return fut.result(timeout=8.0)
    except Exception as e:
        return False, str(e)



    global _ramp_handle
    if _ble_loop is None:
        return False, "BLE not ready"

    def _go():
        global _ramp_handle
        if _ramp_handle and not _ramp_handle.done():
            _ramp_handle.cancel()
        _ramp_handle = _ble_loop.create_task(
            _ramp_task(start_kmh, end_kmh, step_kmh, dwell_s))

    _ble_loop.call_soon_threadsafe(_go)
    return True, "ramp started"


def stop_ramp_sync():
    global _ramp_handle
    if _ble_loop is None:
        return False, "BLE not ready"

    def _stop():
        global _ramp_handle
        if _ramp_handle and not _ramp_handle.done():
            _ramp_handle.cancel()

    _ble_loop.call_soon_threadsafe(_stop)
    _upd_ramp(running=False)
    return True, "ramp stopped"


# ── BLE loop ──────────────────────────────────────────────────────────────────

async def find_treadmill():
    log.info("Scanning for '%s'...", TREADMILL_NAME)
    found = None

    def cb(dev, adv):
        nonlocal found
        if found: return
        name = dev.name or ""
        uuids = [str(u).lower() for u in (adv.service_uuids or [])]
        if TREADMILL_NAME.lower() in name.lower() or FTMS_SERVICE in uuids:
            log.info("Found: %s  rssi=%d", name, adv.rssi)
            found = dev

    scanner = BleakScanner(detection_callback=cb)
    await scanner.start()
    for _ in range(150):
        await asyncio.sleep(0.1)
        if found: break
    await scanner.stop()
    return found


async def session(device):
    global _ble_client, _control_granted, _cp_future, _cp_lock

    def on_data(_h, raw):
        raw = bytes(raw)
        flags = int.from_bytes(raw[0:2], "little")
        if flags & 0x0001: return
        speed = int.from_bytes(raw[2:4], "little") / 100.0
        with _lock:
            _state["speed_kmh"] = speed
            _state["ts"]        = round(time.time(), 4)
            _state["samples"]  += 1

    def on_cp(_h, raw):
        raw = bytes(raw)
        if raw[0] == 0x80 and _cp_future and not _cp_future.done():
            _cp_future.set_result((raw[1], raw[2] if len(raw) > 2 else 0))

    def on_disconnect(_):
        global _control_granted
        log.warning("BLE disconnected")
        _control_granted = False
        _upd(connected=False, speed_kmh=None, control_granted=False)

    _cp_lock = asyncio.Lock()

    async with BleakClient(device, timeout=20.0,
                           disconnected_callback=on_disconnect) as client:
        _ble_client = client
        _upd(connected=True)
        log.info("Connected — subscribing DATA + CP")
        await client.start_notify(UUID_DATA, on_data)
        await client.start_notify(UUID_CP, on_cp)

        # NO keepalive. CONFIRMED 2026-08-14: this treadmill's BLE is READ-ONLY
        # for speed — even the official ZiYou app only reads speed, it cannot set
        # it. Set Target (0x02) returns "Success" for FTMS spec-compliance but the
        # firmware ignores it and reverts the belt to 1.0 km/h. Re-asserting the
        # setpoint (keepalive) cannot beat this and just spams the CP. Speed is set
        # on the PHYSICAL PANEL; BLE is used only as the ground-truth speed LABEL.
        try:
            while client.is_connected:
                await asyncio.sleep(1.0)
                with _lock:
                    spd = _state["speed_kmh"]
                    n   = _state["samples"]
                if spd is not None:
                    log.info("  %.2f km/h  (%d samples)", spd, n)
        finally:
            pass

    _ble_client = None
    _control_granted = False
    _upd(connected=False, speed_kmh=None, control_granted=False)


async def ble_main():
    global _ble_loop
    _ble_loop = asyncio.get_event_loop()
    while True:
        device = await find_treadmill()
        if not device:
            log.warning("Not found — retry in 5s")
            _upd(connected=False, speed_kmh=None)
            await asyncio.sleep(5.0)
            continue
        try:
            await session(device)
        except Exception as e:
            log.warning("Session error: %s", e)
        _upd(connected=False, speed_kmh=None)
        log.info("Reconnecting in 3s...")
        await asyncio.sleep(3.0)


# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path == "/":
            self._send_bytes(INDEX_HTML, "text/html; charset=utf-8")
        elif self.path in ("/speed", "/status"):
            st = _get_state()
            st["ramp"] = _get_ramp()
            self._send_json(st)
        elif self.path == "/ramp/status":
            self._send_json(_get_ramp())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        if self.path == "/treadmill/start":
            ok, msg = start_treadmill_sync()
            self._send_json({"ok": ok, "msg": msg})

        elif self.path == "/treadmill/stop":
            ok, msg = stop_treadmill_sync()
            self._send_json({"ok": ok, "msg": msg})

        elif self.path == "/speed/set":
            spd = data.get("speed_kmh")
            if spd is None:
                self._send_json({"ok": False, "error": "missing speed_kmh"}); return
            ok, msg = set_speed_sync(float(spd))
            self._send_json({"ok": ok, "msg": msg, "speed_kmh": spd})

        elif self.path == "/ramp/start":
            start = float(data.get("start_kmh", 1.0))
            end   = float(data.get("end_kmh",   9.0))
            step  = float(data.get("step_kmh",  0.5))
            dwell = float(data.get("dwell_s",  10.0))
            ok, msg = start_ramp_sync(start, end, step, dwell)
            self._send_json({"ok": ok, "msg": msg})

        elif self.path == "/ramp/stop":
            ok, msg = stop_ramp_sync()
            self._send_json({"ok": ok, "msg": msg})

        else:
            self.send_error(404)

    def _send_json(self, obj):
        data = json.dumps(obj).encode()
        self._send_bytes(data, "application/json")

    def _send_bytes(self, data, ct):
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)


# ── Web UI ────────────────────────────────────────────────────────────────────

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Treadmill Control</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }
header {
  display: flex; align-items: center; justify-content: space-between;
  background: #1a1a2e; padding: 12px 20px; border-bottom: 1px solid #333;
}
header h1 { font-size: 1.05em; color: #00cfff; font-weight: 600; }
#conn-badge {
  font-size: 0.8em; font-weight: 700; padding: 4px 12px; border-radius: 12px;
  background: #2a0f0f; color: #f99;
}
#conn-badge.ok { background: #0b2a0b; color: #7f7; }

.layout { display: flex; flex-direction: column; gap: 14px; padding: 14px; max-width: 480px; margin: 0 auto; }

.card { background: #1c1c2e; border: 1px solid #2a2a4a; border-radius: 10px; padding: 18px; }
.card h2 { font-size: 0.78em; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 14px; }

/* live speed */
#live-speed { font-size: 4.5rem; font-weight: 900; font-family: monospace; color: #00cfff; text-align: center; line-height: 1; }
#ctrl-status { font-size: 0.78em; color: #888; text-align: center; margin-top: 6px; }
#ctrl-status.ok { color: #7f7; }

/* belt start/stop */
.belt-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 6px; }
.belt-btn {
  padding: 16px; border-radius: 8px; border: none; cursor: pointer;
  font-size: 1em; font-weight: 700; letter-spacing: 1px;
}
#belt-start { background: #00aa44; color: #fff; }
#belt-start:hover { filter: brightness(1.15); }
#belt-stop  { background: #cc2222; color: #fff; }
#belt-stop:hover  { filter: brightness(1.15); }
.belt-btn:disabled { opacity: 0.45; cursor: default; filter: none !important; }
#belt-result { font-size: 0.78em; text-align: center; color: #888; min-height: 1.2em; margin-bottom: 10px; }
#belt-result.ok  { color: #7f7; }
#belt-result.err { color: #f77; }
.adj-btn {
  flex: 0 0 64px; height: 56px; border-radius: 8px; border: none; cursor: pointer;
  font-size: 1.3em; font-weight: 700; background: #2a2a4a; color: #e0e0e0;
}
.adj-btn:hover { background: #3a3a6a; }
#target-display {
  flex: 1; text-align: center; font-size: 2rem; font-weight: 900;
  font-family: monospace; color: #ffcc00;
}
#set-btn {
  width: 100%; padding: 14px; border-radius: 8px; border: none; cursor: pointer;
  font-size: 1em; font-weight: 700; letter-spacing: 1px;
  background: #0a84ff; color: #fff; margin-bottom: 14px;
}
#set-btn:hover { filter: brightness(1.15); }
#set-btn:disabled { opacity: 0.5; cursor: default; }
#set-result { font-size: 0.8em; text-align: center; color: #888; min-height: 1.2em; }
#set-result.ok  { color: #7f7; }
#set-result.err { color: #f77; }

.presets { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 10px; }
.preset-btn {
  padding: 10px 4px; border-radius: 6px; border: 1px solid #2a2a4a;
  background: #111; color: #ccc; cursor: pointer; font-size: 0.82em; font-weight: 600;
  text-align: center;
}
.preset-btn:hover { background: #2a2a4a; color: #fff; }
.preset-btn.active { background: #0a84ff; border-color: #0a84ff; color: #fff; }

/* ramp */
.ramp-cfg { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
.cfg-item { display: flex; flex-direction: column; gap: 4px; }
.cfg-item label { font-size: 0.72em; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
.cfg-item input {
  background: #111; border: 1px solid #333; border-radius: 6px; color: #fff;
  padding: 8px 10px; font-size: 1em; font-family: monospace; width: 100%;
}
#ramp-btn {
  width: 100%; padding: 14px; border-radius: 8px; border: none; cursor: pointer;
  font-size: 1em; font-weight: 700; letter-spacing: 1px;
  background: #00aa44; color: #fff;
}
#ramp-btn:hover { filter: brightness(1.15); }
#ramp-btn.stop { background: #cc2222; animation: pulse 1.2s infinite; }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:.7} }
#ramp-progress {
  margin-top: 12px; font-size: 0.82em; color: #888; text-align: center; min-height: 1.4em;
}
.prog-bar-wrap { height: 6px; background: #222; border-radius: 3px; margin-top: 8px; overflow: hidden; }
.prog-bar { height: 100%; background: #00aa44; border-radius: 3px; transition: width 0.5s; width: 0; }
</style>
</head>
<body>

<header>
  <h1>Treadmill Control</h1>
  <div id="conn-badge">● disconnected</div>
</header>

<div class="layout">

  <!-- Live speed -->
  <div class="card">
    <h2>Live Speed (BLE)</h2>
    <div id="live-speed">--</div>
    <div id="ctrl-status">control: not granted</div>
  </div>

  <!-- Belt start / stop -->
  <div class="card">
    <h2>Belt Control</h2>
    <div class="belt-row">
      <button id="belt-start" class="belt-btn" onclick="beltCmd('start')">&#9654; START BELT</button>
      <button id="belt-stop"  class="belt-btn" onclick="beltCmd('stop')">&#9646;&#9646; STOP BELT</button>
    </div>
    <div id="belt-result"></div>
  </div>

  <!-- Manual control -->
  <div class="card">
    <h2>Set Speed</h2>
    <div class="adj-row">
      <button class="adj-btn" onclick="adjust(-0.5)">&#8722;0.5</button>
      <div id="target-display">3.0 km/h</div>
      <button class="adj-btn" onclick="adjust(+0.5)">+0.5</button>
    </div>
    <button id="set-btn" onclick="sendSpeed()">SET SPEED</button>
    <div id="set-result"></div>
    <div class="presets" id="presets"></div>
  </div>

  <!-- Auto ramp -->
  <div class="card">
    <h2>Auto Ramp</h2>
    <div class="ramp-cfg">
      <div class="cfg-item">
        <label>Start (km/h)</label>
        <input id="r-start" type="number" value="1.0" step="0.5" min="0.5" max="9.0">
      </div>
      <div class="cfg-item">
        <label>End (km/h)</label>
        <input id="r-end" type="number" value="9.0" step="0.5" min="1.0" max="9.0">
      </div>
      <div class="cfg-item">
        <label>Step (km/h)</label>
        <input id="r-step" type="number" value="0.5" step="0.5" min="0.5" max="2.0">
      </div>
      <div class="cfg-item">
        <label>Dwell (s)</label>
        <input id="r-dwell" type="number" value="10" step="1" min="5" max="120">
      </div>
    </div>
    <button id="ramp-btn" onclick="toggleRamp()">&#9654; START AUTO RAMP</button>
    <div id="ramp-progress"></div>
    <div class="prog-bar-wrap"><div class="prog-bar" id="prog-bar"></div></div>
  </div>

</div>

<script>
let targetSpeed = 3.0;
let rampRunning = false;

// Build preset buttons
const PRESETS = [];
for (let s = 1.0; s <= 9.05; s = Math.round((s + 0.5) * 10) / 10) PRESETS.push(s);
const presetsEl = document.getElementById("presets");
presetsEl.innerHTML = PRESETS.map(s =>
  `<button class="preset-btn" id="p${s.toFixed(1).replace('.','_')}"
     onclick="setPreset(${s})">${s.toFixed(1)}</button>`
).join("");

function setPreset(s) {
  targetSpeed = s;
  updateTargetDisplay();
  sendSpeed();
}

function adjust(delta) {
  targetSpeed = Math.round((targetSpeed + delta) * 10) / 10;
  targetSpeed = Math.max(0.5, Math.min(9.5, targetSpeed));
  updateTargetDisplay();
}

function updateTargetDisplay() {
  document.getElementById("target-display").textContent = targetSpeed.toFixed(1) + " km/h";
  PRESETS.forEach(s => {
    const el = document.getElementById("p" + s.toFixed(1).replace('.','_'));
    if (el) el.className = "preset-btn" + (Math.abs(s - targetSpeed) < 0.05 ? " active" : "");
  });
}
updateTargetDisplay();

async function beltCmd(action) {
  const res = document.getElementById("belt-result");
  const startBtn = document.getElementById("belt-start");
  const stopBtn  = document.getElementById("belt-stop");
  startBtn.disabled = true; stopBtn.disabled = true;
  res.className = ""; res.textContent = action === "start" ? "starting…" : "stopping…";
  try {
    const r = await fetch("/treadmill/" + action, {method: "POST"});
    const d = await r.json();
    res.className = d.ok ? "ok" : "err";
    res.textContent = d.ok ? "✔ " + d.msg : "✗ " + d.msg;
  } catch(e) {
    res.className = "err"; res.textContent = "network error";
  } finally {
    startBtn.disabled = false; stopBtn.disabled = false;
  }
}

async function sendSpeed() {
  const btn = document.getElementById("set-btn");
  const res = document.getElementById("set-result");
  btn.disabled = true; res.className = ""; res.textContent = "sending…";
  try {
    const r = await fetch("/speed/set", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({speed_kmh: targetSpeed})
    });
    const d = await r.json();
    res.className = d.ok ? "ok" : "err";
    res.textContent = d.ok ? "✔ " + d.speed_kmh + " km/h set" : "✗ " + d.msg;
  } catch(e) {
    res.className = "err"; res.textContent = "network error";
  } finally {
    btn.disabled = false;
  }
}

async function toggleRamp() {
  if (rampRunning) {
    await fetch("/ramp/stop", {method: "POST"});
  } else {
    const body = {
      start_kmh: parseFloat(document.getElementById("r-start").value),
      end_kmh:   parseFloat(document.getElementById("r-end").value),
      step_kmh:  parseFloat(document.getElementById("r-step").value),
      dwell_s:   parseFloat(document.getElementById("r-dwell").value),
    };
    await fetch("/ramp/start", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)
    });
  }
}

// Status poll
async function pollStatus() {
  try {
    const r = await fetch("/status");
    const d = await r.json();

    // connection badge
    const badge = document.getElementById("conn-badge");
    badge.className = d.connected ? "ok" : "";
    badge.textContent = d.connected ? "● connected" : "● disconnected";

    // live speed
    const spd = document.getElementById("live-speed");
    spd.textContent = (d.speed_kmh !== null && d.speed_kmh !== undefined)
      ? d.speed_kmh.toFixed(1) : "--";

    // control status
    const ctrl = document.getElementById("ctrl-status");
    ctrl.className = d.control_granted ? "ok" : "";
    ctrl.textContent = d.control_granted ? "✔ control granted" : "control: not granted";

    // ramp
    const ramp = d.ramp || {};
    rampRunning = ramp.running || false;
    const rampBtn = document.getElementById("ramp-btn");
    rampBtn.className = rampRunning ? "stop" : "";
    rampBtn.innerHTML = rampRunning ? "&#9646;&#9646; STOP RAMP" : "&#9654; START AUTO RAMP";

    const prog = document.getElementById("ramp-progress");
    const bar  = document.getElementById("prog-bar");
    if (rampRunning && ramp.total_steps > 0) {
      const pct = Math.round(ramp.step_index / ramp.total_steps * 100);
      prog.textContent = `step ${ramp.step_index + 1}/${ramp.total_steps} — ${ramp.current_kmh} km/h → ${ramp.end_kmh} km/h`;
      bar.style.width = pct + "%";
      // sync target display with ramp
      if (ramp.current_kmh) { targetSpeed = ramp.current_kmh; updateTargetDisplay(); }
    } else {
      if (!rampRunning) { prog.textContent = ""; bar.style.width = "0"; }
    }
  } catch(e) {
    document.getElementById("conn-badge").className = "";
    document.getElementById("conn-badge").textContent = "● server unreachable";
  }
  setTimeout(pollStatus, 800);
}
pollStatus();
</script>
</body>
</html>
""".encode("utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def _run_ble():
    global _ble_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _ble_loop = loop
    loop.run_until_complete(ble_main())


if __name__ == "__main__":
    threading.Thread(target=_run_ble, daemon=True).start()
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    log.info("Control UI  → http://localhost:%d/", PORT)
    log.info("Speed API   → http://localhost:%d/speed", PORT)
    log.info("Set speed   → POST http://localhost:%d/speed/set", PORT)
    log.info("Auto ramp   → POST http://localhost:%d/ramp/start", PORT)
    server.serve_forever()
