# The pipeline, in detail

The README has the commands. This is the reasoning behind each stage, and the things that
cost an afternoon once and are written down so they do not cost another.

## Why a duck needs to see ducks

Everything social in the robot's behaviour stack currently keys on Bluetooth: a duck knows another
duck is *nearby*, never *where*. Seeing one is what turns "somebody is in the room" into
approaching, following, facing, and reacting — and the chorale into ducks that look at each other
while they sing.

The target is one class (`duck`) at a frame rate the RK3566's small NPU can hold while `robotd`
keeps its 50 Hz control loop, on a camera mounted 20 cm off the floor behind a wide lens.

## Stages

| stage | what it does | state |
|---|---|---|
| **capture** | subscribe to a robot's own camera stream, one session at a time | works |
| **review** | triage, pre-label, correct in Label Studio, get YOLO labels back — one command | works |
| **train** | fine-tune a small detector, split by session, export ONNX | works, needs data |
| **export** | ONNX → RKNN, INT8, and measure it on the board | works |

Every stage reads and writes the Hub: `capture --push`, `review --push`, `train --push`, and
`dataset`/`model` `push`/`pull` for doing it by hand.

**One thing about the extras before anything else:** they are not additive. `uv sync --extra
train` uninstalls what `--extra label` put there, and a bare `uv sync` uninstalls both — so unless
you only ever capture, the line to use is:

```bash
uv sync --all-extras
hf auth login        # once; the Hub, and the rendezvous that reaches the robot, use this token
```

### capture

```bash
uv run capture --tag kitchen-afternoon --seconds 120 --push
```

**Nothing on the robot is stopped, and nothing is installed on it.** `mediad` holds the camera and
keeps holding it. The tool asks it for `media.stream` — the same call the vision-demo Space makes
— and the robot dials a WebSocket the tool opens on the laptop and pushes JPEG frames down it,
upright, at 2 Hz, straight off the tee that already feeds the console's video and the robot's own
duck detector. It is `duckctl open` with a program at the far end instead of a browser:

```
laptop ──media.stream {url: "ws://192.168.10.42:8765/frames"}──► rendezvous ──► robot
robot  ═══════════════ JPEG frames, LAN, direct ═══════════════════════════► laptop
```

The instruction crosses the Hugging Face rendezvous, which is how the robot is found (no address
to type: the duck online on your account is the duck) and why `hf auth login` is a prerequisite.
The pixels do not: the robot has to be able to reach the laptop, so same LAN, and `--advertise
<ip>` when the laptop's guess at its own address is wrong (two LANs, a VPN). `--dry-run` finds the
duck and streams nothing. `-v` logs the rendezvous traffic.

Two rules the transport imposes, both said plainly by the tool when hit: **one consumer at a
time** — a console open on the robot (`duckctl open`) makes it busy — and the robot needs a
`mediad` with the control lane over the rendezvous (the robot repo's current `main`; an older one
accepts the session and answers nothing, which the tool says in as many words).

Frames arrive **already upright** (`mediad` turns them by the mount angle before its tee, and the
hello says `rotate: 0`), so what is captured is what a model will be handed at inference, by
construction. `session.json` records that, the robot's name, serial and release, and the robot's
own hello verbatim.

Two ducks make this easy: one walks around, the other watches. What to shoot, roughly in order of
how much it buys:

- **A duck in another duck's view**, at 20 cm to 3 m, from every angle — head on, from behind,
  half-hidden behind a chair leg.
- **Rooms with no duck in them at all.** A detector trained only on frames containing a duck finds
  ducks in curtains. Tag these `empty-<room>`.
- **The awkward cases**: backlit against a window, a duck on a dark floor, two ducks overlapping,
  a duck lying down after a fall, wheels on.
- **Other robots and other toys.** A Reachy Mini turned up in the first session's background for
  free; a stool and a pair of legs were the pre-labeller's two false positives. These are the hard
  negatives, and they are what stop the model learning "small thing on a floor".

Keep sessions short and many rather than one long one: a session is the unit the train/val split
uses, because two frames half a second apart are the same picture and mixing them across the split
reports a score the model has not earned.

### review — the whole middle of the pipeline, in one command

```bash
uv run review datasets/raw/<session> --push
```

That does all of it: ranks the frames, pre-labels them, starts Label Studio, logs in, creates the
project, imports the tasks with the boxes already drawn, and opens a browser at them. Correct the
boxes, press **Ctrl-C in the terminal**, and the corrections come back through the API as YOLO
labels in `datasets/reviewed/<session>/` — and, with `--push`, onto the Hub. Nothing is exported by
hand, and no token is copied out of a settings page.

Re-running it on a session you are halfway through costs a second: the triage and the pre-labels are
already on disk, and a project that already has its tasks is left alone.

It opens the **labelling stream**, not the task table: in the table a frame opens in a modal where
submitting does not move on and half the shortcuts are unbound, which is maddening across fifty
frames. In the stream, `1` selects the label, `Ctrl+Enter` submits and advances, `Delete` removes the
selected box, `Ctrl+Z` undoes. A frame with nothing in it should be submitted empty — that is a
negative, and it is worth having.

Label Studio runs out of `.label-studio/` in the repo, with a user this tool invents and a password
it writes down there — so it cannot collide with another Label Studio, and `rm -rf .label-studio` is
a clean slate. The port is 8080, or the next one free.

What the two halves are doing, for when one of them misbehaves:

- **Triage.** A duck walking around films the floor: of the first session's 99 frames, most are a
  soft grey blur. Two cheap numbers per frame — variance of a Laplacian for sharpness, mean edge
  energy for content — pick the 40 worth looking at and keep 10 empty ones as negatives, because a
  detector that has never seen an empty room finds ducks in curtains. `triage.json` records the
  numbers so the thresholds can be argued with rather than guessed at.
- **Pre-labelling.** Grounding DINO (tiny), prompted with noun phrases. On the first session it
  found a duck in 48 of 50 frames, 0.31–0.67, with two false positives — a blue stool and a pair of
  legs. That is the point: deleting two boxes is a different job from drawing fifty. Two things
  learned the hard way, both in the code as comments: **every noun in the prompt is a query** (`a
  small robot standing on the floor` asked it to find the floor, and it did, frame-sized, in a
  third of them), and **colour is not a cue** — these robots come in blue, white and grey, so
  `yellow duck` finds the cushion. The threshold stays low on purpose: a missing box costs more
  human time than a wrong one.

`uv run autolabel <session> --sheet` renders the boxes onto a contact sheet if you want to see what
the pre-labeller thinks before opening the editor, and `uv run review --import <export>.json` reads
a manual export if the API round trip ever does not happen.

### Is the pre-labeller worth it

```bash
uv run agreement datasets/raw/<session>
```

Compares what it drew with what survived the correction pass, which is the number that decides
whether to keep using it. On the first session:

```
  frames reviewed      50  (6 deliberately empty)
  frames untouched     44  (88%)
  boxes accepted       50
  boxes nudged         0
  boxes drawn anew     1   ← what the pre-labeller missed
  boxes deleted        6   ← what it invented
  precision 89%  recall 98%
```

88% of frames needed no touch at all, and nothing had to be nudged — so the boxes are not merely
in the right place, they are tight enough to accept. That is the case for pointing it at hundreds
of frames and keeping the human job to "glance, Ctrl+Enter". It is also the case for the low
threshold: the single box it missed cost more attention than the six it invented.

### train

```bash
uv run dataset pull           # every session anyone has captured and corrected
uv run dataset build          # or --smoke, for one session, plumbing only
uv run train --export --push  # tagged with the run's name; main points at it
```

`dataset build` makes **one dataset out of every reviewed session** — all of them by default, or
a choice: `--tag kitchen --tag hall` takes every session shot with those tags, `--sessions a b`
names them, `--exclude c` leaves one out. Which sessions went in, and which side of the split each
landed on, is written to `build.json`, and `train` copies that into the run so the model on the Hub
says what it was trained on.

It **refuses to split a single session**, because splitting one by frame puts near-copies on both
sides and the val score becomes memorisation. It holds out whole sessions, the newest by default
(`--val` names others), and symlinks rather than copies so the frames stay the one copy in `raw/`.

`yolo11n` at 320×320 from COCO weights. The augmentation follows the camera rather than a
photo set: ±12° of roll because the camera rolls with the gait, modest translation and scale because
it is one camera at one height, no vertical flip because the daemon already turned the picture the
right way up. The motion blur is left to the data — half of every session has it, which teaches
better than a blur transform would.

`--export` writes ONNX with static shapes at opset 12, which is what `rknn-toolkit2` will take.
`--push` uploads `best.pt` and `best.onnx` as `duck_detect.pt`/`.onnx` on the model repo, with the
run's `summary.json`, `results.csv` and `args.yaml` under `runs/<name>/`, and tags the commit with
the run's name.

### export

```bash
uv run --isolated --python 3.12 --with rknn-toolkit2 --with pillow --with onnxruntime \
    --with "setuptools<81" --with "onnx==1.16.1" scripts/to_rknn.py \
    runs/detect/duck-v1/weights/best.onnx
uv run model push runs/detect/duck-v1        # adds duck_detect.rknn under the same tag
```

Its own interpreter and its own pins, because `rknn-toolkit2` publishes wheels for cp310–cp312
(this repo runs 3.13), still imports `pkg_resources` (gone in setuptools 84) and still calls
`onnx.mapping` (gone after onnx 1.16). None of that is negotiable, so it is written down here
rather than rediscovered.

10 MB of float ONNX becomes **3.9 MB of INT8 RKNN**, quantised against 120 letterboxed frames drawn
from the real sessions. The script then runs both models on a frame the corrections say has a duck
in it and matches the detections as *sets*:

```
  float onnx: 2 box(es) after nms
  int8 rknn : 2 box(es) after nms
  2 of 2 kept, mean overlap 95%
  scores: float ['0.92', '0.91']  int8 ['1.38', '1.38']
```

Two things that cost an hour and are now in the code as comments. The head emits 2100 candidates,
so **one duck is twenty overlapping boxes** until something suppresses them — comparing "the best
box" of each model compared two arbitrary members of the same cluster and reported 8% overlap on a
model that was working perfectly. And **the int8 output tensor carries its own scale**, so scores
land outside 0..1: a confidence threshold has to be set against the quantised model rather than
inherited from the float one.

The robot reads `models/duck_detect.rknn` out of its release (`deploy/robotd.toml`, `[detect]`).
`uv run model pull` puts the current one — or `--revision duck-v1` a named one — in `weights/`,
which is what to hand the release build.

## The Hub, in one place

```bash
uv run dataset push [session ...]    # frames once, corrections whenever they change
uv run dataset pull [session ...]    # into datasets/, same layout
uv run model push runs/detect/<name> [--tag <tag>]
uv run model pull [--revision <tag>]
```

The dataset is `pollen-robotics/microduck-duck-detector-dataset`, the model `pollen-robotics/microduck-duck-detector`, both created
private on first push (`--public` if that is wanted). `DUCK_DATASET_REPO` and `DUCK_MODEL_REPO`
point the tools at a fork or a scratch account. Sessions are pushed as one commit each and never
rewritten — a frame on the Hub is a frame somebody may have labelled — while `reviewed/<session>/`
is replaced as a unit, so re-pushing after a correction pass changes the labels and nothing else.
Each push regenerates the dataset card's session table from `session.json`, and each model push
writes a card with the run's metrics.

## One local wrinkle

This machine's CUDA wheel (`torch 2.13+cu130`) ships cuDNN sublibraries that fail each other's
version check, and it is not the loader path — a scrubbed `LD_LIBRARY_PATH` fails identically. Every
tool here probes a convolution at startup and turns cuDNN off if that is what it takes, which costs
a little speed and keeps everything working. A torch build whose cuDNN is consistent is the real
fix, whenever somebody picks one.

## Layout

```
datasets/raw/<session>/       frames + session.json + triage.json   ─┐ gitignored,
datasets/labelled/<session>/  the pre-labeller's boxes, and a contact sheet ├ mirrored on the Hub
datasets/reviewed/<session>/  what a person corrected — the training labels ─┘
datasets/yolo/                what `dataset build` assembles, symlinks to raw
weights/                      what `model pull` fetches
src/duck_detector/            the tools; robot.py is the rendezvous client, hub.py the Hub
docs/                         notes worth keeping
```
