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
| **label** | pre-label with an open-vocabulary detector, then correct by hand | next |
| **train** | fine-tune a small detector, split by session | next |
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
- **Other yellow things.** A rubber duck, a banana, a cushion. This is what stops the model
  learning "yellow blob".

Keep sessions short and many rather than one long one: a session is the unit the train/val split
uses, because two frames half a second apart are the same picture and mixing them across the split
reports a score the model has not earned.

### label — next

The plan: pre-label with an open-vocabulary detector (OWLv2 or Grounding DINO, prompted
`a small yellow duck robot`) and correct by hand, rather than draw thousands of boxes from nothing.
An 8 GB laptop GPU runs either at a few frames a second, which is plenty for a few thousand stills,
and the correction pass is the only part that costs a human.

### train — next

A small YOLO (`yolo11n`/`yolov8n`) at 320×320, fine-tuned from COCO weights: one class, a few
thousand frames, heavy augmentation on brightness and blur because the ISP and the shutter are what
change most between rooms. Split by session, never by frame.

### export — after that

ONNX with static shapes and an opset `rknn-toolkit2` accepts, then INT8 quantisation with a
calibration set drawn from the robot's own frames — the quantisation is where a detector that
worked on the laptop stops working on the board, so it is measured there, not assumed.

## Layout

```
datasets/raw/<session>/       frames + session.json   (gitignored)
src/duck_detector/            the tools
docs/                         notes worth keeping
```
