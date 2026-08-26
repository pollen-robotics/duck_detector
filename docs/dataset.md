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

## What a session records

```json
{
  "session": "20260826T120000Z_kitchen_duck-c51b",
  "tag": "kitchen",
  "robot": "duck-c51b",
  "serial": "bb7b734a7717ac41",
  "frames": 240,
  "hz": 2.0,
  "width": 1280, "height": 720,
  "flip": "90r",
  "release": "0.9.3",
  "note": "two ducks, one walking"
}
```

`flip` is the one that will cause an afternoon of confusion if it is ever wrong: the robot turns
the picture a quarter turn before anything sees it (`mediad --rotate`, default 90°), so the dataset
is captured through the same turn. A session captured with `--flip identity` is not wrong, it is a
different domain — and mixing the two silently trains a detector for neither.

`robot` and `serial` matter because the cameras are not identical: same module, different
mounting, and eventually a different board. A model that only ever saw one robot's camera is worth
knowing about.

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

## Layout

```
datasets/
  raw/<session>/frame_00001.jpg …   session.json
  labelled/<session>/                YOLO-format .txt beside each frame
```

Neither is in git — a session is tens of megabytes of JPEG. The repo carries the recipe.
