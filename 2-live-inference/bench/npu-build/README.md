# Building the NPU context binary (`speed_cnn_a16w8_htpv75.bin`)

The shipped app runs the speed model on the **CPU** with ONNX Runtime — that is
the reproducible default and needs nothing from this folder. This guide is the
**optional** path that produced the NPU artifacts used in the CPU-vs-NPU study
([`../cpu_vs_npu.md`](../cpu_vs_npu.md)): how to take the trained `speed_cnn.onnx`
and compile it into the **Hexagon HTP V75** context binary
`speed_cnn_a16w8_htpv75.bin` that `3-bluetooth-ftms/game_server.py --provider qnn`
runs on the board.

It requires Qualcomm's free **QAIRT** (Qualcomm AI Runtime SDK), which you obtain
yourself — the SDK and the compiled `.bin` are **not** in this repo (large binaries,
Qualcomm's to distribute; same treatment as the `.onnx`). What *is* here is the
full recipe, so with your own QAIRT install you can regenerate every artifact.

> **This switches computers.** Training (Part 1) and the live app (Parts 2–3) run
> on a Mac and on the board. **This conversion runs on an x86-64 Linux machine** —
> the QAIRT NPU compiler only exists for x86 Linux. The board itself cannot compile
> the model; it only *runs* the finished `.bin`.

Known-good versions from a full reproduction: **QAIRT 2.47.0.260601**, Hexagon
target **V75** (`"htp_arch": "v75"`), Python 3.12, `numpy==1.26.4`, `onnx==1.16.1`,
`onnxruntime==1.18.1`.

---

## 1. What you need

- An **x86-64 Linux box** (Ubuntu 22.04/24.04 tested) with ~10 GB free disk. Not a
  quota-limited `$HOME` — the SDK unzips to several GB.
- The **QAIRT SDK** (free — see step 2). It provides `qairt-converter`,
  `qairt-quantizer`, `qnn-context-binary-generator`, and the runtime `.so` libs.
- **`speed_cnn.onnx`** — the trained model (input `frames [1,3,8,112,112]` float32,
  output `speed_kmh [1,1]`). Download it from the repo's model release, or export
  your own with `1-data-acquisition/export_onnx.py`.
- **Calibration clips** — a few dozen real preprocessed `(1,3,8,112,112)` float32
  tensors (step 3). Quantization learns each tensor's value range from real data.

Pick a **workspace directory** and keep everything under it. In this guide,
`$WORK` is the parent folder for the whole NPU reproduction, not the git checkout
itself. The repository will be cloned inside it as `$WORK/treadmill`.

For example, if you want the whole project under `~/prjs/treadmill`, use:

```bash
export WORK="$HOME/prjs/treadmill"     # change this to any writable path
mkdir -p "$WORK" && cd "$WORK"
```

After step 3, the layout will look like this:

```text
$WORK/
├── treadmill/                  # git clone of this repository
├── qairt/2.47.0.260601/         # QAIRT SDK
├── .venv/                       # Python environment for QAIRT tools
├── llvm-libs/                   # local libc++ runtime extracted from .debs
├── speed_cnn.onnx               # model to convert
├── calib/                       # calibration .raw tensors + input_list.txt
├── ctx/                         # generated context binary output
└── npu-stage/                   # files copied to the VENTUNO Q
```

---

## 2. Install the QAIRT SDK (once)

**Download the Community edition** (free, no login). Match the SDK version to your
board's runtime — mine reported `2.47.0.260601`, so I used the exact same SDK:

```bash
cd "$WORK"
# From the Qualcomm Software Center — "Qualcomm AI Runtime SDK", Community edition.
# The versioned download path (no login for Community):
wget "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.47.0.260601/v2.47.0.260601.zip"
unzip v2.47.0.260601.zip          # -> ./qairt/2.47.0.260601/  (this is your SDK dir)
export SDK="$WORK/qairt/2.47.0.260601"
```

> If the direct link 403s (the gateway can require a browser session), open the
> **Qualcomm Software Center** in a browser, search **"Qualcomm AI Runtime SDK"**,
> pick the **Community** edition at version **2.47.0.260601**, and download the zip
> — it's the same file. The `qpm-cli` package manager (also on the Software Center)
> can fetch it headlessly too.

**Create a Python venv.** On stock Ubuntu the built-in `python3 -m venv` is often
broken (`ensurepip is not available`) and needs sudo; `virtualenv` avoids that:

```bash
pip install --user --break-system-packages virtualenv
python3 -m virtualenv "$WORK/.venv"
source "$WORK/.venv/bin/activate"
```

**Install the Python deps — these exact versions.** The SDK's
`check-python-dependency` verifies its support packages but does **not** install
`onnx`, and recent `onnx`/`numpy` break the SDK's native code:

```bash
python3 "$SDK/bin/check-python-dependency"                 # installs ~30 support packages
pip install "numpy==1.26.4" "onnx==1.16.1" "onnxruntime==1.18.1"
```

> Why the pins (each is a real failure otherwise): `numpy` 2.x breaks the SDK's
> native `.so` files; `onnx` 1.22 drops the `onnx.version.version` attribute the
> converter reads (`AttributeError: module 'onnx' has no attribute 'version'`);
> `onnxruntime==1.18.1` matches the pinned `onnx`.

**Stage the LLVM libc++ runtime.** The SDK's native tools are built against LLVM's
`libc++`, which a clean Ubuntu box does not ship and the SDK does not bundle — so
you'd otherwise get `libc++.so.1: cannot open shared object file`. Fetch them
without sudo:

```bash
cd /tmp
apt-get download libc++1-18 libc++abi1-18 libunwind-18      # just fetch the .debs
for d in libc++1-18_*.deb libc++abi1-18_*.deb libunwind-18_*.deb; do
  dpkg-deb -x "$d" "$WORK/llvm-libs"                        # extract whole deb (symlinks need it)
done
cd "$WORK"
export LLVM_LIBS="$WORK/llvm-libs/usr/lib/llvm-18/lib"      # holds the three .so.1 files
```

**Put the tools on PATH** and verify:

```bash
source "$SDK/bin/envsetup.sh"                               # SDK sets PATH / LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$LLVM_LIBS:$LD_LIBRARY_PATH"
qairt-converter --version                                  # should print 2.47.0.260601
```

---

## 3. Get the model and calibration tensors

The quantizer needs **calibration tensors**. These are not labels and they are not
used to train the model again. They are just representative model inputs that let
QAIRT measure activation ranges before it converts the network to A16W8
quantization. For this project the cleanest source is the public training cache:
it already contains the exact `(3,8,112,112)` clips produced by the documented
preprocessing pipeline.

So this step does three things:

1. clone the repo,
2. download the trained `speed_cnn.onnx`,
3. download the public `dataset_repro.tar.gz` archive so `dataset_repro/cache_side.npz`
   exists, then convert 64 representative clips from that cache into QAIRT `.raw`
   calibration tensors.

> The git checkout alone does **not** contain `dataset_repro/cache_side.npz` or the
> videos. Those large files live in the external dataset archive. If the Python
> script below says `dataset_repro/cache_side.npz` is missing, run the `gdown` +
> `tar` commands in this section first.

```bash
cd "$WORK"

# Get the public repo under the workspace. This creates $WORK/treadmill.
# If it already exists, reuse it.
if [ ! -d "$WORK/treadmill/.git" ]; then
  git clone https://github.com/munoz0raul/treadmill.git "$WORK/treadmill"
fi
export REPO="$WORK/treadmill"
cd "$REPO"

# Get the trained ONNX model used by the article/benchmark.
curl -L -o "$WORK/speed_cnn.onnx" \
  "https://github.com/munoz0raul/treadmill/releases/download/model-v1/speed_cnn.onnx"

# Get the reproducible dataset archive. This archive contains the large side.mp4
# files and the prebuilt dataset_repro/cache_side.npz. Plain curl does not work
# reliably for this large Google Drive file, so use gdown.
pip install gdown
gdown 1V_-AhkDP4gH7HobBH-CnSRrkIpcf5Evp -O dataset_repro.tar.gz
tar -xzf dataset_repro.tar.gz

# Sanity check: this must exist before calibration.
test -f dataset_repro/cache_side.npz
```

Now create the quantizer input list from the public cache:

```bash
cd "$REPO"
python3 - <<'PY'
import os
import numpy as np

WORK = os.environ["WORK"]
cache = "dataset_repro/cache_side.npz"
if not os.path.exists(cache):
    raise SystemExit(
        f"Missing {cache}. Download dataset_repro.tar.gz with gdown and unpack it first."
    )

d = np.load(cache)
X = d["X"]  # uint8, shape (N,3,8,112,112)
os.makedirs(f"{WORK}/calib", exist_ok=True)

# Pick 64 clips spread across the cache so the quantizer sees the full dataset.
# Labels are not used for quantization; only representative input values matter.
idxs = np.linspace(0, len(X) - 1, 64, dtype=int)
for j, i in enumerate(idxs):
    clip = (X[i:i+1].astype(np.float32) / 255.0)  # add batch: (1,3,8,112,112)
    clip.tofile(f"{WORK}/calib/clip_{j:03d}.raw")
print(f"wrote {len(idxs)} calibration tensors to {WORK}/calib")
PY

find "$WORK/calib" -name 'clip_*.raw' | sort > "$WORK/calib/input_list.txt"
wc -l "$WORK/calib/input_list.txt"   # expected: 64
```

If you do not want to download the prebuilt cache, rebuild it from the downloaded
videos first:

```bash
cd "$REPO"
python3 1-data-acquisition/preprocess.py \
  --sessions dataset_repro/session_* \
  --out dataset_repro/cache_side.npz
```

> Advanced note: you *can* calibrate from live clips dumped with
> `2-live-inference/live_speed.py --dump-clips`, but that is only useful if you
> intentionally want the NPU binary calibrated to a different camera/framing. For
> reproducing this project, use the public cache path above.

---

## 4. Convert → quantize (A16W8) → context binary

Three tools, run from `$WORK`. The graph name QAIRT derives from the model is
`speed_a16w8` after quantization — the JSON configs below must match it.

**4.1 — ONNX → DLC (float).** Translate the graph into Qualcomm's `.dlc`, still
floating point:

```bash
cd "$WORK"
qairt-converter \
  --input_network speed_cnn.onnx \
  --source_model_input_shape frames 1,3,8,112,112 \
  --output_path speed_fp.dlc
```

The released ONNX model has a dynamic batch axis on the input named `frames`.
QAIRT needs the concrete shape, so the converter command pins it to the live
inference shape: batch 1, RGB channels 3, 8 frames, 112×112 pixels.

**4.2 — Quantize to A16W8 (16-bit activations, 8-bit weights).** This is the key
choice. The model outputs a single continuous value — `speed_kmh` — with a small
dynamic range. In plain INT8 (8-bit activations) that delicate output loses too
much precision; **16-bit activations preserve it** while 8-bit weights keep the
model small and fast. This is what kept the quantized NPU output within
**0.054 km/h MAE** of the float CPU model (see [`../cpu_vs_npu.md`](../cpu_vs_npu.md)):

```bash
qairt-quantizer \
  --input_dlc speed_fp.dlc \
  --input_list calib/input_list.txt \
  --act_bitwidth 16 --weights_bitwidth 8 \
  --output_dlc speed_a16w8.dlc
```

This step can take a while. The quantizer runs the graph once for each
calibration clip on the host CPU backend to collect activation ranges. With 64
video clips, a slow host can take tens of minutes. Seeing `QNN_CPU` in this log
is expected; the NPU execution happens later on the board.

**4.3 — Compile the HTP V75 context binary.** `qnn-context-binary-generator` needs
two small JSON configs — one naming the graph + target arch, one pointing the
backend extension at it:

```bash
cat > htp_config.json <<'JSON'
{
  "graphs": [ { "graph_names": ["speed_a16w8"], "vtcm_mb": 0, "O": 3 } ],
  "devices": [ { "htp_arch": "v75" } ]
}
JSON

cat > backend_ext.json <<JSON
{
  "backend_extensions": {
    "shared_library_path": "$SDK/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so",
    "config_file_path": "$WORK/htp_config.json"
  }
}
JSON

mkdir -p ctx
qnn-context-binary-generator \
  --dlc_path speed_a16w8.dlc \
  --backend "$SDK/lib/x86_64-linux-clang/libQnnHtp.so" \
  --config_file backend_ext.json \
  --output_dir ctx \
  --binary_file speed_cnn_a16w8_htpv75
# -> ctx/speed_cnn_a16w8_htpv75.bin
```

That `.bin` is the artifact the board runs.

---

## 5. Stage the runtime for the board

This step has two sides:

- **HOST**: the x86-64 Linux machine where you installed QAIRT and ran steps 1–4.
- **BOARD**: the Arduino VENTUNO Q, which will only run the finished artifact.

The board does **not** need the full QAIRT SDK. It only needs:

```text
$BOARD_WORK/models/npu/
├── bin/qnn-net-run
├── lib/*.so
├── dsp/libQnnHtpV75Skel.so
└── speed_cnn_a16w8_htpv75.bin
```

`$BOARD_WORK` is the workspace path **on the VENTUNO Q**, not necessarily the same
literal home directory as the host. If your board user is different from your host
user, write the board path explicitly, for example
`/home/ubuntu/prjs/treadmill` or `/home/arduino/prjs/treadmill`.

### 5.1 — On the HOST: collect board-runtime files into `npu-stage/`

Run this on the x86-64 Linux host from `$WORK`.

First, choose **one** aarch64 runtime target from the SDK. Do not copy from
`aarch64-*` with a wildcard: the SDK contains several aarch64 variants with the
same filenames, and `cp` will print messages such as “will not overwrite
just-created ...” when they all try to land in the same destination.

List the available targets:

```bash
cd "$WORK"
find "$SDK/bin" -maxdepth 1 -type d -name 'aarch64-*' -printf '%f\n' | sort
find "$SDK/lib" -maxdepth 1 -type d -name 'aarch64-*' -printf '%f\n' | sort
```

For the VENTUNO Q image used in this project, the known-good target from the
full reproduction is the OpenEmbedded aarch64 build:

```bash
export QNN_TARGET="aarch64-oe-linux-gcc11.2"
```

Do not assume `aarch64-ubuntu-gcc9.4` is the right choice just because the board
runs Linux: in QAIRT 2.47 that target may be missing HTP-specific files such as
`libQnnHtpV75Stub.so`. Pick one target that contains all required files, and keep
all `bin/` and `lib/` files from the **same** `$QNN_TARGET`. Do not mix files from
multiple target folders unless you are deliberately debugging ABI compatibility.

Now create a clean staging folder and copy exactly that target's files:

```bash
cd "$WORK"
rm -rf "$WORK/npu-stage"
mkdir -p "$WORK/npu-stage/bin" "$WORK/npu-stage/lib" "$WORK/npu-stage/dsp"

# Sanity check: fail early if the chosen target is not present in this SDK
# or if it lacks the HTP V75 runtime stub.
test -x "$SDK/bin/$QNN_TARGET/qnn-net-run"
test -f "$SDK/lib/$QNN_TARGET/libQnnHtp.so"
test -f "$SDK/lib/$QNN_TARGET/libQnnHtpV75Stub.so"

# qnn-net-run for the board CPU architecture. You copy it here on the host;
# you do not run this aarch64 binary on the host.
cp "$SDK/bin/$QNN_TARGET/qnn-net-run"                        "$WORK/npu-stage/bin/"

# CPU-side aarch64 runtime libraries for the board.
cp "$SDK/lib/$QNN_TARGET/libQnnHtp.so"                       "$WORK/npu-stage/lib/"
cp "$SDK/lib/$QNN_TARGET/libQnnSystem.so"                    "$WORK/npu-stage/lib/"
cp "$SDK/lib/$QNN_TARGET/libQnnHtpNetRunExtensions.so"       "$WORK/npu-stage/lib/"
cp "$SDK/lib/$QNN_TARGET/libQnnHtpPrepare.so"                "$WORK/npu-stage/lib/"
cp "$SDK/lib/$QNN_TARGET/libQnnHtpV75Stub.so"                "$WORK/npu-stage/lib/"

# On-NPU kernel: the DSP skel. This is the version-sensitive file.
cp "$SDK/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so"       "$WORK/npu-stage/dsp/"

# The context binary generated in step 4.3.
cp "$WORK/ctx/speed_cnn_a16w8_htpv75.bin"                    "$WORK/npu-stage/"
```

After this, the **host** should have:

```text
$WORK/npu-stage/
├── bin/qnn-net-run
├── lib/libQnnHtp.so
├── lib/libQnnSystem.so
├── lib/libQnnHtpNetRunExtensions.so
├── lib/libQnnHtpPrepare.so
├── lib/libQnnHtpV75Stub.so
├── dsp/libQnnHtpV75Skel.so
└── speed_cnn_a16w8_htpv75.bin
```

Optional sanity check on the host:

```bash
find "$WORK/npu-stage" -maxdepth 2 -type f | sort
```

### 5.2 — Still on the HOST: copy `npu-stage/` directly to the BOARD

Stay in the same terminal on the x86-64 Linux host. You do **not** SSH into the
host again. Just set the VENTUNO Q connection details and use `scp` from the host
to the board:

```bash
# Replace these two values with your board login and IP address.
# Do not leave the angle-bracket placeholders in the command.
export BOARD_USER="ubuntu"                # example only: use your VENTUNO Q username
export BOARD_IP="192.168.1.50"            # example only: use your VENTUNO Q IP address

# Path on the VENTUNO Q. This common case assumes the user's home is /home/$BOARD_USER.
# If your board uses a different home path, set BOARD_WORK to that absolute path instead.
export BOARD_WORK="/home/$BOARD_USER/prjs/treadmill"
```

Create the destination directory on the board, then copy the staged files:

```bash
ssh "$BOARD_USER@$BOARD_IP" "mkdir -p '$BOARD_WORK/models/npu'"
scp -r "$WORK/npu-stage/"* "$BOARD_USER@$BOARD_IP:$BOARD_WORK/models/npu/"
```

If you see an error like `remote mkdir "/home/<board-user>/...": No such file or
directory`, it means the placeholder was copied literally. Set `BOARD_USER` to the
real board username and make `BOARD_WORK` match the actual path on the board.

After the copy, the **board** should have this layout:

```text
$BOARD_WORK/models/npu/
├── bin/qnn-net-run
├── lib/libQnnHtp.so
├── lib/libQnnSystem.so
├── lib/libQnnHtpNetRunExtensions.so
├── lib/libQnnHtpPrepare.so
├── lib/libQnnHtpV75Stub.so
├── dsp/libQnnHtpV75Skel.so
└── speed_cnn_a16w8_htpv75.bin
```

You can verify from the host with:

```bash
ssh "$BOARD_USER@$BOARD_IP" "find '$BOARD_WORK/models/npu' -maxdepth 2 -type f | sort"
```

> **Version-match gotcha.** The board firmware ships **QAIRT 2.46**; this `.bin` was
> built with **2.47**. Ship the matching **2.47** DSP skel (above) and let
> `game_server.py` point `ADSP_LIBRARY_PATH` at **only** that dir — otherwise the
> firmware's 2.46 skel wins the version race and device creation fails with **error
> 1008**. (`NpuSpeedBackend` in `game_server.py` handles this.)

---

## 6. Run it live

On the VENTUNO Q, use the same `$WORK` / `$REPO` convention from
`docs/PROJECT_HUB.md`, with the runtime staged under `$WORK/models/npu`.

First make sure the **board checkout** is on the branch that contains the NPU
backend. If `game_server.py --help` does not show `--provider`, you are running an
older copy of the code.

```bash
export WORK="$HOME/prjs/treadmill"     # or the workspace path you chose on the board
export REPO="$WORK/treadmill"
cd "$REPO"
python3 3-bluetooth-ftms/game_server.py --help | grep -- '--provider'
```

Then run the app:

```bash
cd "$REPO/3-bluetooth-ftms"
python3 game_server.py --camera /dev/video0 \
  --provider qnn --npu-runtime "$WORK/models/npu" --npu-strict
# INFO NPU backend ready (Hexagon HTP V75) from $WORK/models/npu
# INFO CNN estimator: provider=qnn (Hexagon HTP V75) — clip span 1.00s
```

If you get `error: unrecognized arguments: --provider qnn --npu-runtime ...`, the
NPU runtime files may be staged correctly, but the board is running an old
`game_server.py`. Pull the branch above on the board and retry.

The payoff — fidelity vs the float CPU model and the latency win — is measured in
[`../cpu_vs_npu.md`](../cpu_vs_npu.md).

---

## Why this route (and not ONNX Runtime's QNN provider)?

ONNX Runtime has a QNN execution provider, but it **cannot place this model's 5D
`Conv3d` ops on the HTP** — they fall back to the CPU, so it isn't a real NPU run.
The QAIRT context-binary + `qnn-net-run` path in this guide is the only way to get
this 3D-CNN onto the Hexagon NPU. That's why the live backend shells out to
`qnn-net-run` per clip rather than using the QNN EP.
