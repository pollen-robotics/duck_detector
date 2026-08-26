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
