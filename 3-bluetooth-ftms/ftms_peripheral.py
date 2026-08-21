#!/usr/bin/env python3
"""
ftms_peripheral.py — a vision-powered *virtual* FTMS treadmill.

This is the mirror image of `mac_ble_server.py`. There, the Mac was a BLE
**central** that read the treadmill's own FTMS speed. Here the edge board is the
BLE **peripheral**: it impersonates a smart treadmill, advertising the standard
Fitness Machine Service (FTMS, 0x1826) so any fitness *game* (Zwift, Rouvy, the
ftmsemu verifier, a phone app, …) connects to it as a central and consumes a
speed we feed it — the speed the camera + CNN read off a dumb, Bluetooth-less
treadmill.

    dumb treadmill ──▶ camera+CNN ──▶ speed_provider() ──▶ FtmsTreadmill ──BLE──▶ game

GATT layout (all under FTMS 0x1826):
  0x2ACD  Treadmill Data       notify + read   ← the speed stream we push @2Hz
  0x2ACC  Fitness Machine Feature   read        ← feature bitfield (we expose none)
  0x2AD9  Fitness Machine Control Point  write+indicate ← handshake stub
  0x2ADA  Fitness Machine Status    notify       ← handshake stub

Treadmill Data packet is byte-compatible with this repo's own decoder in
`mac_ble_server.py` (`speed = int.from_bytes(raw[2:4],"little")/100.0`): flags
uint16 LE + instantaneous speed uint16 LE in 0.01 km/h units.

bless is asyncio-based; this class owns a private event loop in a daemon thread,
so the (threaded) HTTP server and camera can drive it with plain method calls.
"""

import asyncio
import logging
import struct
import threading
import time

log = logging.getLogger(__name__)

try:
    from bless import (BlessServer, GATTCharacteristicProperties,
                       GATTAttributePermissions)
    _BLESS_AVAILABLE = True
    # bless spells the "writable" permission differently across versions
    # (0.2.x: `writable`, 0.3.x: `writeable`). Resolve whichever exists so the
    # code runs on both.
    _PERM_WRITABLE = getattr(GATTAttributePermissions, "writable", None) \
        or getattr(GATTAttributePermissions, "writeable")
except ImportError:
    _BLESS_AVAILABLE = False

# ── FTMS UUIDs (copied verbatim from mac_ble_server.py so the two sides agree) ──
FTMS_SERVICE = "00001826-0000-1000-8000-00805f9b34fb"
UUID_DATA    = "00002acd-0000-1000-8000-00805f9b34fb"   # Treadmill Data (notify)
UUID_FEATURE = "00002acc-0000-1000-8000-00805f9b34fb"   # Fitness Machine Feature
UUID_SPEEDRG = "00002ad4-0000-1000-8000-00805f9b34fb"   # Supported Speed Range (read)
UUID_CP      = "00002ad9-0000-1000-8000-00805f9b34fb"   # Control Point
UUID_STATUS  = "00002ada-0000-1000-8000-00805f9b34fb"   # Fitness Machine Status

# Supported Speed Range (0x2AD4): min, max, min-increment — all uint16 LE in
# 0.01 km/h. Running games (Zwift) read this during discovery and silently drop
# a treadmill that doesn't expose it. 0..20 km/h, 0.1 km/h step covers walk→run.
SPEED_MIN_KMH = 0.0
SPEED_MAX_KMH = 20.0
SPEED_STEP_KMH = 0.1

NOTIFY_HZ = 2.0   # speed push rate to the game (FTMS spec minimum is ~1 Hz)


class FtmsTreadmill:
    """A BLE peripheral that advertises FTMS and streams a caller-supplied speed.

    `speed_provider` is a zero-arg callable returning the current speed in km/h
    (or None). We pass `CnnSpeedEstimator.latest_speed` so the belt speed the AI
    reads from the camera flows straight to the game.
    """

    def __init__(self, name="AI Treadmill", speed_provider=None, notify_hz=NOTIFY_HZ):
        self._name = name
        self._speed_provider = speed_provider or (lambda: None)
        self._notify_hz = notify_hz

        self._loop = None
        self._server = None
        self._thread = None
        self._notify_handle = None

        self._advertising = False
        self._connected = False
        self._last_sent = None
        self._last_cp_write = None   # wall-clock of the last Control-Point write —
                                     # the real "a game is driving us" signal, since
                                     # bless's is_connected() is adapter-wide and
                                     # also fires for a stray laptop/phone probe.
        self._lock = threading.Lock()

    # ── byte layout ───────────────────────────────────────────────────────────
    @staticmethod
    def _treadmill_data(kmh) -> bytes:
        """Build a Treadmill Data (0x2ACD) payload.

        flags=0x0000 → bit0 ("More Data") clear means Instantaneous Speed is the
        first field: uint16 LE in 0.01 km/h units. This is exactly what
        mac_ble_server.py decodes on the way in, so a round-trip is lossless.
        """
        kmh = max(0.0, float(kmh))
        return struct.pack("<HH", 0x0000, round(kmh * 100))

    @property
    def available(self) -> bool:
        return _BLESS_AVAILABLE

    # ── public, thread-safe control (called from HTTP handler threads) ──────────
    def start(self):
        """Start advertising. Spins up the BLE thread on first call; re-starts
        the server on later calls after a stop()."""
        if not _BLESS_AVAILABLE:
            return {"ok": False, "error": "bless not installed"}
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._thread_main, daemon=True)
                self._thread.start()
                return {"ok": True, "started": True}
            loop, server = self._loop, self._server
        if loop is not None and not self._advertising:
            fut = asyncio.run_coroutine_threadsafe(self._start_server(), loop)
            try:
                fut.result(timeout=10)
            except Exception as e:
                log.warning("FTMS start failed: %s", e)
                return {"ok": False, "error": str(e)}
        return {"ok": True, "advertising": self._advertising}

    def stop(self):
        """Stop advertising (the BLE thread/loop stays alive for a later start)."""
        with self._lock:
            loop = self._loop
        if loop is not None and self._advertising:
            fut = asyncio.run_coroutine_threadsafe(self._stop_server(), loop)
            try:
                fut.result(timeout=5)
            except Exception as e:
                log.warning("FTMS stop failed: %s", e)
                return {"ok": False, "error": str(e)}
        return {"ok": True, "advertising": self._advertising}

    def restart(self):
        """Stop then re-advertise. A BLE peripheral goes silent once *any* central
        connects (single-link), so if a stray device (e.g. a laptop that probed the
        radio) grabbed the link, phones/games can no longer find us. This kicks the
        current link and puts us back on the air — the UI's "Reset connection"."""
        r_stop = self.stop()
        time.sleep(0.5)
        r_start = self.start()
        return {"ok": r_start.get("ok", False), "stopped": r_stop, "started": r_start}

    def set_name(self, name: str):
        """Rename the advertised device. Only takes effect while not advertising
        (the name is baked into the advert when the server starts)."""
        if self._advertising:
            return {"ok": False, "error": "stop broadcasting before renaming"}
        self._name = name
        return {"ok": True, "name": name}

    def status(self):
        with self._lock:
            # "link_up" = some central is connected to the adapter (bless's view).
            # "game_active" = we've seen an FTMS Control-Point write recently, i.e.
            # a real fitness client did the request-control/start handshake — this
            # is what actually distinguishes a game from a stray probe.
            game_active = (self._last_cp_write is not None
                           and (time.time() - self._last_cp_write) < 30.0)
            return {
                "available": _BLESS_AVAILABLE,
                "name": self._name,
                "advertising": self._advertising,
                "link_up": self._connected,
                "connected": self._connected,   # back-compat alias
                "game_active": game_active,
                "speed_kmh": self._last_sent,
            }

    # ── asyncio internals (run inside the private loop thread) ──────────────────
    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._start_server())
            self._notify_handle = self._loop.create_task(self._notify_task())
            self._loop.run_forever()
        except Exception as e:
            log.error("FTMS BLE loop crashed: %s", e)
        finally:
            self._loop.close()

    async def _start_server(self):
        # A fresh server per (re)start — bless has no re-advertise-after-stop API.
        self._server = BlessServer(name=self._name, loop=self._loop)
        self._server.read_request_func = self._on_read
        self._server.write_request_func = self._on_write

        await self._server.add_new_service(FTMS_SERVICE)

        # Fitness Machine Feature: 8 bytes (features + target-setting features).
        # All-zero is valid — instantaneous speed on Treadmill Data is present
        # regardless of this field; some centrals just read it during discovery.
        await self._server.add_new_characteristic(
            FTMS_SERVICE, UUID_FEATURE,
            GATTCharacteristicProperties.read,
            bytearray(struct.pack("<II", 0x00000000, 0x00000000)),
            GATTAttributePermissions.readable)

        # Supported Speed Range (0x2AD4): min, max, min-increment — uint16 LE in
        # 0.01 km/h. Zwift/Rouvy running read this during discovery; a treadmill
        # that omits it is silently dropped (confirmed with ftms_probe.py).
        await self._server.add_new_characteristic(
            FTMS_SERVICE, UUID_SPEEDRG,
            GATTCharacteristicProperties.read,
            bytearray(struct.pack("<HHH",
                                  round(SPEED_MIN_KMH * 100),
                                  round(SPEED_MAX_KMH * 100),
                                  round(SPEED_STEP_KMH * 100))),
            GATTAttributePermissions.readable)

        # Treadmill Data: the actual speed stream (notify), seeded at 0 km/h.
        await self._server.add_new_characteristic(
            FTMS_SERVICE, UUID_DATA,
            GATTCharacteristicProperties.notify | GATTCharacteristicProperties.read,
            bytearray(self._treadmill_data(0.0)),
            GATTAttributePermissions.readable)

        # Control Point + Status: a compatibility stub. Games (e.g. Zwift) send
        # 0x00 "Request Control" then 0x07 "Start" before trusting the data.
        # There is no belt to move — we just ACK so the handshake completes.
        await self._server.add_new_characteristic(
            FTMS_SERVICE, UUID_CP,
            GATTCharacteristicProperties.write | GATTCharacteristicProperties.indicate,
            None,
            _PERM_WRITABLE)
        await self._server.add_new_characteristic(
            FTMS_SERVICE, UUID_STATUS,
            GATTCharacteristicProperties.notify,
            bytearray(),
            GATTAttributePermissions.readable)

        await self._server.start()
        with self._lock:
            self._advertising = True
        log.info("FTMS peripheral advertising as %r (FTMS service 0x1826)", self._name)

    async def _stop_server(self):
        try:
            if self._server is not None:
                await self._server.stop()
        finally:
            with self._lock:
                self._advertising = False
                self._connected = False
            self._server = None
            log.info("FTMS peripheral stopped advertising")

    async def _notify_task(self):
        """Every 1/notify_hz s, push the current speed to any subscribed central."""
        period = 1.0 / self._notify_hz
        while True:
            await asyncio.sleep(period)
            server = self._server
            if not self._advertising or server is None:
                continue
            try:
                connected = await server.is_connected()
            except Exception:
                connected = False
            spd = self._speed_provider()
            if spd is None:
                spd = 0.0
            try:
                char = server.get_characteristic(UUID_DATA)
                if char is not None:
                    char.value = bytearray(self._treadmill_data(spd))
                    server.update_value(FTMS_SERVICE, UUID_DATA)
            except Exception as e:
                log.debug("notify failed: %s", e)
            with self._lock:
                self._connected = connected
                self._last_sent = float(spd)

    # ── GATT request callbacks ──────────────────────────────────────────────────
    def _on_read(self, characteristic, **kwargs):
        return characteristic.value

    def _on_write(self, characteristic, value, **kwargs):
        """Control-Point stub: ACK any op-code with Success so the game's
        request-control / start handshake completes. No belt is actuated."""
        if str(characteristic.uuid).lower() != UUID_CP.lower():
            characteristic.value = value
            return
        op = value[0] if value else 0xFF
        log.info("FTMS control point write op=0x%02x → ACK Success", op)
        with self._lock:
            self._last_cp_write = time.time()   # a real game is driving us
        # Response Code (0x80), Request Op Code, Result Code (0x01 = Success).
        resp = bytes([0x80, op, 0x01])
        try:
            cp = self._server.get_characteristic(UUID_CP)
            if cp is not None:
                cp.value = bytearray(resp)
                self._server.update_value(FTMS_SERVICE, UUID_CP)
        except Exception as e:
            log.debug("control-point ACK failed: %s", e)


# ── standalone smoke test: advertise a synthetic 0→8→0 km/h ramp, no camera ─────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _t0 = time.time()

    def _fake_ramp():
        # Triangular 0→8→0 over 32 s, so a scanner sees the value move.
        phase = (time.time() - _t0) % 32.0
        return phase / 2.0 if phase < 16.0 else (32.0 - phase) / 2.0

    tm = FtmsTreadmill(name="AI Treadmill", speed_provider=_fake_ramp)
    if not tm.available:
        raise SystemExit("bless not installed — pip install bless")
    print("Starting FTMS peripheral (synthetic ramp). Ctrl-C to stop.")
    tm.start()
    try:
        while True:
            time.sleep(2)
            print("status:", tm.status())
    except KeyboardInterrupt:
        tm.stop()
        print("stopped.")
