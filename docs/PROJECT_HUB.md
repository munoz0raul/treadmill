# AI Treadmill: teaching a camera to read a treadmill's speed

*A vision-powered virtual smart treadmill on the Arduino UNO Q.*

> **Media placeholders** — replace the `![...]()` lines and the video link below
> with the real photos and demo clip. Suggested shots are listed in Part 4.

![Hero shot: the treadmill, the camera, and the live speed on screen](docs/images/hero.jpg)

## What this project does

Most treadmills at home are "dumb": they have a belt and a speed dial, and that's
it. No Bluetooth, no way to connect them to a running game like Zwift or Rouvy.
The smart ones that do connect are expensive.

This project makes a dumb treadmill act like a smart one — with nothing but a
**camera** and a small **AI model** running on an **Arduino UNO Q**. A webcam
watches the belt from the side, a neural network reads the speed straight from
the video, and the board broadcasts that speed over Bluetooth using the same
standard a real smart treadmill uses. Open your favorite fitness game, and your
avatar moves in time with your walk.

```
dumb treadmill ──▶ camera ──▶ CNN (reads speed from video) ──▶ Bluetooth (FTMS) ──▶ game
```

No modification to the treadmill. No sensors taped to the belt. Just a camera
pointed at it and a bit of AI.

## Devices and components

- **Arduino UNO Q** (Qualcomm Dragonwing platform) — the brain. It runs the AI
  model and acts as the Bluetooth device.
- **USB webcam** — a regular 720p webcam, placed to the side of the treadmill.
- **A treadmill** — any belt treadmill. No Bluetooth or sensors required.
- *(For training only)* a computer with a GPU to fine-tune the model.

## Software and tools

- **Python 3** with **OpenCV** (camera + video), **NumPy**, and **ONNX Runtime**
  (running the model on the board).
- **PyTorch / torchvision** — for training the model (`mc3_18` video network).
- **bless** — a Python library that lets the board act as a Bluetooth Low Energy
  peripheral.
- A **web browser** — the app's interface is a simple local web page.

## How it was built

The project came together in three practical steps, plus a final demo. Each is a
self-contained folder in the repository.

### Part 1 — Teaching by example (data acquisition)

An AI model is only as good as the examples it learns from, and we needed examples
of "this is what the belt looks like at 3 km/h, at 4 km/h…". The treadmill can't
tell us its speed, so we made our own ground truth.

The `acquire.py` script opens a web page showing the camera and a row of speed
buttons: **Stopped, 0.5, 1.0, … up to 8.0 km/h**. The routine is simple: set a
speed on the treadmill, tap the matching button, and walk. Every tap is saved
with a timestamp next to the raw video. The dataset then assumes the value you
picked is the true belt speed until you change it.

![The acquisition web UI: camera preview and speed buttons](docs/images/acquire-ui.jpg)

We recorded several sessions across different days, with different people and
outfits, holding each speed for about 30 seconds before stepping up — plus one
session of an empty, stopped belt so the model learns what "stopped" looks like.

A second script, `preprocess.py`, turns those recordings into training clips.
Each clip is **8 frames spanning exactly 1 second** of motion — about one walking
stride — cropped to the center of the frame so the model sees the walker and the
belt, not the furniture. Sampling by *time* (rather than by frame count) means it
doesn't matter if one video recorded at 24 fps and another at 30 — a one-second
clip is always a one-second clip.

### Part 2 — Reading the speed live (inference)

With a trained model in hand, `live_speed.py` puts it to work. It shows the camera
on one side of the page and a big speed number on the other, updated several times
a second as you walk.

![Live inference: camera on the left, AI speed on the right](docs/images/live-ui.jpg)

Under the hood, the model looks at that rolling 1-second clip and predicts a
speed. Two small touches keep it honest: a **motion gate** that reports 0 km/h
when the belt clearly isn't moving, and a short **rolling average** so the number
doesn't jitter. This is already a complete, useful app — it just doesn't talk to
anything else yet.

### Part 3 — Speaking the treadmill language (Bluetooth / FTMS)

The last step is to make games believe our camera-and-AI setup *is* a treadmill.
Fitness equipment speaks a standard Bluetooth dialect called **FTMS** (Fitness
Machine Service). Normally the treadmill is the Bluetooth device and the game
connects to it; we simply play the part of the treadmill.

The board advertises itself as **"AI Treadmill"** and streams the AI speed the way
a real smart treadmill would. `game_server.py` is Part 2's app plus this Bluetooth
layer and a small connection panel in the UI (Advertising → Linked → Game active).

![The Bluetooth panel connecting to a game](docs/images/ble-panel.jpg)

Getting a real game to pair reliably took some care — for example, running games
expect the treadmill to advertise its supported speed range, and will silently
ignore one that doesn't. To debug this without fighting a black-box game, the repo
includes `ftms_probe.py`: a little tool that pretends to be a game, walks through
the whole pairing sequence, and prints PASS/FAIL at each step. That made the
difference between "it sometimes works" and "it works".

### Part 4 — The demo

The final piece is a demonstration video of the whole thing working end to end —
walking on the treadmill while a game responds. *(Coming soon.)*

## Try it yourself

Everything is on GitHub, split into the four folders above with a README in each.

```bash
# See the AI speed live (needs a trained models/speed_cnn.onnx)
python3 2-live-inference/live_speed.py --cnn-model ~/models/speed_cnn.onnx

# The full app, broadcasting over Bluetooth to a game
python3 3-bluetooth-ftms/game_server.py --cnn-model ~/models/speed_cnn.onnx
```

**Repository:** https://github.com/munoz0raul/treadmill

## Notes and lessons

- **Sample by time, not by frames.** Different cameras and even the same camera
  under different lighting record at different frame rates. Defining a clip as "1
  second of motion" instead of "8 frames" is what let the model trained on
  recordings stay accurate live.
- **Bluetooth on combo radios.** On boards that share one radio for Wi-Fi and
  Bluetooth, heavy Wi-Fi use can weaken the Bluetooth signal. Using a wired
  network for the board — and checking the antenna — gave the most reliable
  pairing.
- **Build a probe.** When integrating with someone else's black-box app, a small
  tool that imitates that app and reports each step is worth its weight in gold.

## License

Released under the [Mozilla Public License 2.0](LICENSE).
