# The dataset

## Sessions, not frames

A capture is one **session**: a stretch of time, one robot looking, one room, one lighting
condition, recorded in `session.json` beside the frames.

Everything downstream splits by session. Two frames half a second apart are the same picture — the
duck has moved a centimetre — so a random frame split puts near-copies in both train and val, and
the val score then measures memorisation. It is the most common way a detector looks good on a
laptop and useless on a robot, and it is invisible unless the split is built to prevent it.

So: **many short sessions beat one long one.** Change something between them — the room, the light,
where the watching duck stands — and tag what changed.

## Where a session comes from

The robot's own stream. `capture` asks `mediad` for `media.stream` — the same call the vision-demo
Space makes — and the robot dials a WebSocket on the laptop and pushes JPEG frames down it, off the
tee that already feeds the console's video and the robot's own duck detector. `mediad` is never
stopped and nothing is installed on the board; the frames are, by construction, the picture the
detector will be handed at inference.

## What a session records

```json
{
  "session": "20260910T140000Z_kitchen_graphite",
  "tag": "kitchen",
  "robot": "graphite",
  "serial": "cec2b3808a7238ff",
  "frames": 240,
  "hz": 2.0,
  "seconds": 120,
  "width": 720, "height": 1280,
  "rotate": 0,
  "mount_rotate": 90,
  "release": "0.9.4",
  "note": "two ducks, one walking",
  "transport": "media.stream",
  "longest": 1280, "quality": 90,
  "hello": { "...the robot's own description of the stream..." }
}
```

`rotate` and `mount_rotate` are the ones that would cause an afternoon of confusion if they were
ever wrong. The robot turns the picture by the mount angle (`mediad --rotate`, default 90°) before
its tee, so the frames arrive **upright** and the hello says `rotate: 0` about them — a session with
anything else there is a different domain, and mixing the two silently trains a detector for
neither. `hello` is kept whole because it is the robot's own account of what it sent.

`robot` and `serial` matter because the cameras are not identical: same module, different
mounting, and eventually a different board. A model that only ever saw one robot's camera is worth
knowing about.

Sessions from before September 2026 were captured over ssh with `mediad` stopped and carry `flip:
"90r"` instead — the same upright picture by a different route, and the same domain.

## What to capture

Roughly in order of what it buys:

1. **A duck in another duck's view**, 20 cm to 3 m, every angle, half-occluded as often as not.
2. **Rooms with no duck at all** — a detector trained only on frames containing a duck finds ducks
   in curtains. Tag `empty-<room>`.
3. **The awkward ones**: backlit against a window, dark floor, two ducks overlapping, a duck lying
   down after a fall, wheels on.
4. **Other robots and other toys**: a Reachy Mini, a stool, a bag on the floor. These are the hard
   negatives — the pre-labeller's mistakes on session one were a blue stool and a pair of legs,
   which is exactly what wants correcting rather than avoiding.

The shells come in **blue, white and grey**, so colour is not a cue and nothing in this pipeline
treats it as one.

## Layout, here and on the Hub

```
datasets/
  raw/<session>/frame_00000.jpg …   session.json     ─┐
  labelled/<session>/frame_*.txt    labels.json       ├─ mirrored on the Hub
  reviewed/<session>/frame_*.txt                     ─┘
  review/<session>/                 Label Studio's tasks — scratch
  yolo/                             what `dataset build` assembles — scratch
```

None of it is in git. The three data directories are the tree of the dataset repo
`pollen-robotics/microduck-duck-detector-dataset`: `uv run dataset push` sends what the Hub does not have
(frames once, corrections every time they change) and `uv run dataset pull` brings it all down.
