# duck detector

A detector that finds Microducks in a Microduck's camera, for the robot's NPU. One class, `duck`,
320×320, INT8 RKNN. Deployed by [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck).

| | |
|---|---|
| dataset | [`pollen-robotics/microduck-duck-detector-dataset`](https://huggingface.co/datasets/pollen-robotics/microduck-duck-detector-dataset) |
| model | [`pollen-robotics/microduck-duck-detector`](https://huggingface.co/pollen-robotics/microduck-duck-detector) |

## Setup

```bash
uv sync --all-extras     # the extras are not additive; always sync them all
hf auth login            # once — the Hub, and the rendezvous that finds the robot, use this token
```

## The whole sequence

```bash
# 1. capture a session from a duck that is on and online (nothing is stopped or installed on it)
uv run capture --tag kitchen-afternoon --seconds 120 --push

# 2. correct the pre-labels in Label Studio, Ctrl-C when done; the labels go to the Hub
uv run review datasets/raw/<session> --push

# 3. one dataset from every reviewed session on the Hub, then train
uv run dataset pull
uv run dataset build                  # or --tag kitchen, --sessions a b, --exclude c
uv run train --export --push          # runs/detect/duck/, tagged "duck" on the Hub

# 4. quantise for the NPU and push the .rknn under the same tag
uv run --isolated --python 3.12 --with rknn-toolkit2 --with pillow --with onnxruntime \
    --with "setuptools<81" --with "onnx==1.16.1" scripts/to_rknn.py runs/detect/duck/weights/best.onnx
uv run model push runs/detect/duck

# 5. fetch the current detector for the robot's release build
uv run model pull                     # weights/duck_detect.{pt,onnx,rknn}
```

## The short version of each step

- **capture** asks the robot for `media.stream`; the robot dials a WebSocket on the laptop and
  pushes upright JPEG frames. The robot is found through the Hugging Face rendezvous. It must reach
  the laptop (same LAN; `--advertise <ip>` if the guess is wrong), and a console open on it makes it
  busy. Many short tagged sessions beat one long one; capture rooms with no duck too.
- **review** triages, pre-labels with Grounding DINO, and opens Label Studio with the boxes drawn.
  `1` selects the label, `Ctrl+Enter` submits and advances, `Delete` removes a box. Submit empty
  frames empty.
- **dataset build** merges sessions into one YOLO dataset and holds out whole sessions, never
  frames. It refuses a single session unless `--smoke`.
- **train** fine-tunes `yolo11n` at 320 and exports ONNX. **to_rknn.py** quantises against real
  frames and checks the INT8 model still sees a duck; its odd `uv run` line is what `rknn-toolkit2`
  requires.

## Hub commands

```bash
uv run dataset push [session ...]     # frames once, corrections whenever they change
uv run dataset pull [session ...]
uv run model push runs/detect/<name> [--tag <tag>]
uv run model pull [--revision <tag>]
```

## More

- [docs/pipeline.md](docs/pipeline.md) — why each stage is the way it is, and what went wrong once
- [docs/dataset.md](docs/dataset.md) — sessions, `session.json`, what to capture
