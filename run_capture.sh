#!/bin/bash
# Launch the treadmill capture rig on the edge board.
#
# Run this ON THE BOARD (the machine with the camera attached).
#
# ── PREREQUISITES ────────────────────────────────────────────────────────────
#   A separate host (e.g. a Mac) must be running mac_ble_server.py on port 8765
#   (it does the BLE handshake with the treadmill and serves the speed as JSON).
#   Point the board at it with the MAC_BLE_URL env var, e.g.:
#     export MAC_BLE_URL="http://192.168.1.50:8765"
#   The board polls $MAC_BLE_URL/speed — no board-side Bluetooth needed.
#
# ── CAMERA ───────────────────────────────────────────────────────────────────
#   Single camera, labeled "side" (the CNN uses the side view).
#   List devices:  v4l2-ctl --list-devices

set -e
cd "$(dirname "$0")"

SIDE_CAM="${SIDE_CAM:-/dev/video0}"
DATASET_DIR="${DATASET_DIR:-$HOME/dataset}"

echo "Side camera: $SIDE_CAM"

python3 pose_server.py \
    --camera "$SIDE_CAM" \
    --labels side \
    --dataset "$DATASET_DIR" \
    "$@"
