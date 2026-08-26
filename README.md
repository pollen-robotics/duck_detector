# duck detector

Finding Microducks in a Microduck's camera — the dataset, the training, and the export that ends up
on the robot's NPU.

Deployed by [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck), the way
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) policies are: trained here,
exported, and loaded there. The robot repo stays Rust and stays fast; the datasets, the CUDA wheels
and the checkpoints live here.

## Why a duck needs to see ducks

Everything social in the robot's behaviour stack currently keys on Bluetooth: a duck knows another
duck is *nearby*, never *where*. Seeing one is what turns "somebody is in the room" into
approaching, following, facing, and reacting — and the chorale into ducks that look at each other
while they sing.

The target is one class (`duck`) at a frame rate the RK3566's small NPU can hold while `robotd`
keeps its 50 Hz control loop, on a camera mounted 20 cm off the floor behind a wide lens.

## The pipeline

| stage | what it does | state |
|---|---|---|
| **capture** | pull stills from a robot's own camera, one session at a time | works |
| **triage** | rank a session's frames, because most of them are blurred floor | works |
| **label** | pre-label with an open-vocabulary detector, then correct by hand | works, needs the correction pass |
| **review** | correct those boxes in Label Studio, and get YOLO labels back | works |
| **train** | fine-tune a small detector, split by session, export ONNX | works, needs data |
| **export** | ONNX → RKNN, INT8, and measure it on the board | after that |

### capture

```bash
uv sync
uv run capture --host microduck@192.168.10.124 --tag kitchen-afternoon --seconds 120
```

It stops `mediad` (V4L2 capture is exclusive, and `mediad` holds the camera), captures at 2 Hz,
pulls the frames into `datasets/raw/<session>/`, starts `mediad` again, and writes a
`session.json` beside the frames. `--dry-run` prints what it would run.

**Frames are captured through the same quarter turn the robot applies at inference**
(`mediad --rotate`, default 90°). A dataset captured sideways trains a detector for a camera nobody
has, and it is invisible in the numbers.

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

### triage

```bash
uv run triage datasets/raw/<session>
```

A duck walking around films the floor: of the first session's 99 frames, most are a soft grey blur
and a handful have a room and another duck in them. Two cheap numbers per frame — variance of a
Laplacian for sharpness, mean edge energy for content — rank them, and `triage.json` records the
numbers so the thresholds can be argued with rather than guessed at. It keeps a deliberate sample of
the empty frames too, because a detector that has never seen an empty room finds ducks in curtains.

### label

```bash
uv sync --extra label
uv run autolabel datasets/raw/<session> --sheet
```

Grounding DINO (tiny) prompted with noun phrases, then a person corrects it. On the first session
it found a duck in 48 of 50 frames, scores 0.31–0.67, with two false positives — a blue stool and a
pair of legs. That is the point: deleting two boxes is a different job from drawing fifty.

Two things learned the hard way, both in the code as comments:

- **Every noun in the prompt is a query.** `a small robot standing on the floor` asked it to find
  the floor, and it did, in a third of the frames, with a box the size of the frame.
- **Colour is not a cue.** These robots come in blue, white and grey shells, so `yellow duck` finds
  the cushion.

`--sheet` renders the boxes onto a contact sheet. Look at it before correcting anything; it is how
you find out the prompt is describing the sofa. The threshold stays low on purpose — a missing box
costs more human time than a wrong one.

### review

```bash
uv sync --extra review
uv run review prepare datasets/raw/<session>   # tasks with the boxes already drawn
uv run review serve                            # Label Studio, pointed at this repo
#   … correct, then Export → JSON
uv run review import <export>.json             # corrections back as YOLO labels
```

`serve` sets `LOCAL_FILES_SERVING_ENABLED` and the document root, and `prepare` writes tasks that
point at `/data/local-files/?d=…` — so the frames load with nothing configured in the UI. The
pre-labeller's boxes arrive as *predictions*, which is the difference between accepting a box and
drawing one.

A frame somebody opened and left empty is a negative and is kept. A frame nobody opened is skipped:
"there is nothing here" and "nobody looked" are different, and only one of them is training data.

### train

```bash
uv sync --extra train
uv run dataset build          # or --smoke, for one session, plumbing only
uv run train --export
```

`dataset build` **refuses to split a single session**, because splitting one by frame puts
near-copies on both sides and the val score becomes memorisation. It holds out whole sessions, the
newest by default, and symlinks rather than copies so the frames stay the one copy in `raw/`.

`yolo11n` at 320×320 from COCO weights. The augmentation follows the camera rather than a
photo set: ±12° of roll because the camera rolls with the gait, modest translation and scale because
it is one camera at one height, no vertical flip because the daemon already turned the picture the
right way up. The motion blur is left to the data — half of every session has it, which teaches
better than a blur transform would.

`--export` writes ONNX with static shapes at opset 12, which is what `rknn-toolkit2` will take.

### export — after that

The ONNX exists; the RKNN conversion does not yet. INT8 with a calibration set drawn from the
robot's own frames, then measured on the board — quantisation is where a detector that worked on the
laptop stops working, so it is a measurement rather than an assumption.

## One local wrinkle

This machine's CUDA wheel (`torch 2.13+cu130`) ships cuDNN sublibraries that fail each other's
version check, and it is not the loader path — a scrubbed `LD_LIBRARY_PATH` fails identically. Every
tool here probes a convolution at startup and turns cuDNN off if that is what it takes, which costs
a little speed and keeps everything working. A torch build whose cuDNN is consistent is the real
fix, whenever somebody picks one.

## Layout

```
datasets/raw/<session>/       frames + session.json   (gitignored)
src/duck_detector/            the tools
docs/                         notes worth keeping
```
