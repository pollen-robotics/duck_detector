"""ONNX → RKNN, quantised to INT8, checked on the simulator before it ever reaches a board.

Run it with its own interpreter, because `rknn-toolkit2` publishes wheels for cp310–cp312 and this
repo runs on 3.13:

    uv run --isolated --python 3.12 --with rknn-toolkit2 --with pillow \
        scripts/to_rknn.py runs/detect/duck-v1/weights/best.onnx

**Quantisation is the step where a detector that worked on a laptop stops working**, so this does
three things rather than one: builds the INT8 model, runs it on the toolkit's simulator, and reports
what it found on a frame that is known to contain a duck. A conversion that succeeds and detects
nothing is the failure worth catching here, on a desk, rather than on a robot.

The calibration images are letterboxed to the model's own input size and drawn from real sessions —
quantisation ranges come from what the camera actually produces, not from a resize of something
else.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

from PIL import Image

# What `mediad` will hand the model, and what `dataset build`/`train` used: RGB, letterboxed into a
# square, values 0..1. The mean/std below are how RKNN is told about that last part.
INPUT = 320
MEAN = [[0.0, 0.0, 0.0]]
STD = [[255.0, 255.0, 255.0]]

# The SoC in the robot. `rk3566` and `rk3568` share the NPU; the toolkit wants one named.
TARGET = "rk3566"

# Letterbox grey, matching ultralytics' padding so calibration sees the same borders inference will.
PAD = (114, 114, 114)


def letterbox(image: Image.Image, size: int = INPUT) -> Image.Image:
    """Scale to fit, pad to square — the same shape the training pipeline fed the model."""
    scale = min(size / image.width, size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height))
    canvas = Image.new("RGB", (size, size), PAD)
    canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
    return canvas


def calibration_set(frames: list[Path], directory: Path, count: int) -> Path:
    """Letterboxed JPEGs, and the list file RKNN reads them from."""
    random.seed(0)  # a reproducible build matters more than a fresh sample every time
    chosen = random.sample(frames, min(count, len(frames)))
    listing = directory / "calibration.txt"
    lines = []
    for index, frame in enumerate(chosen):
        path = directory / f"calib_{index:04d}.jpg"
        letterbox(Image.open(frame).convert("RGB")).save(path, quality=95)
        lines.append(str(path))
    listing.write_text("\n".join(lines) + "\n")
    sessions = len({f.parent.name for f in chosen})
    print(f"calibration: {len(chosen)} frames from {sessions} session(s)")
    return listing


Box = tuple[float, float, float, float]


def decode(output, threshold: float = 0.35) -> list[tuple[float, Box]]:
    """YOLO11's single-class head: `(1, 5, N)` of cx, cy, w, h, score — in input pixels.

    Written here as well as in the robot's own code, because this is what says whether the quantised
    model still sees anything: a conversion that runs and finds nothing looks identical to one that
    worked, right up until the robot ignores a duck.
    """
    import numpy as np

    array = np.array(output).reshape(5, -1)
    boxes = []
    for index in range(array.shape[1]):
        score = float(array[4, index])
        if score < threshold:
            continue
        cx, cy, w, h = (float(v) for v in array[:4, index])
        boxes.append((score, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    boxes.sort(key=lambda box: -box[0])

    # **Suppressed here, because the head is not.** 2100 candidates per frame means one duck comes
    # back as twenty overlapping boxes, and comparing "the best raw candidate" between two models
    # compares two arbitrary members of the same cluster. The robot will have to do this too.
    kept: list[tuple[float, tuple[float, float, float, float]]] = []
    for score, box in boxes:
        if all(iou(box, other) < 0.5 for _, other in kept):
            kept.append((score, box))
    return kept


def with_a_duck(reviewed: Path, raw: Path) -> Path | None:
    """A frame the corrections say contains a duck, newest session first.

    The point of the check is to distinguish "quantisation broke it" from "there was nothing there",
    and only the reviewed labels know which frames have something in them.
    """
    for session in sorted(reviewed.glob("*"), reverse=True):
        for label in sorted(session.glob("frame_*.txt")):
            if not label.read_text().strip():
                continue
            frame = raw / session.name / f"{label.stem}.jpg"
            if frame.exists():
                return frame
    return None


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - overlap
    return overlap / union if union > 0 else 0.0


def float_inference(onnx: Path, image) -> list:
    """The unquantised model on the same frame, as the control."""
    import numpy as np
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    batch = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    return decode(session.run(None, {session.get_inputs()[0].name: batch})[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantise a trained detector for the robot's NPU.")
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: alongside the onnx")
    parser.add_argument("--raw", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--calibration", type=int, default=120, help="frames to quantise against")
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="a frame with a duck in it; by default one the reviewed labels say has one",
    )
    parser.add_argument("--reviewed", type=Path, default=Path("datasets/reviewed"))
    args = parser.parse_args()

    if not args.onnx.exists():
        raise SystemExit(f"no {args.onnx} — `uv run train --export` writes it")
    out = args.out or args.onnx.with_suffix(".rknn")

    frames = sorted(args.raw.glob("*/frame_*.jpg"))
    if not frames:
        raise SystemExit(f"no frames under {args.raw} to calibrate against")

    from rknn.api import RKNN

    with tempfile.TemporaryDirectory() as work:
        listing = calibration_set(frames, Path(work), args.calibration)

        rknn = RKNN(verbose=False)
        rknn.config(mean_values=MEAN, std_values=STD, target_platform=TARGET)
        if rknn.load_onnx(model=str(args.onnx)) != 0:
            raise SystemExit("could not load the onnx")
        print(f"building int8 for {TARGET} …")
        if rknn.build(do_quantization=True, dataset=str(listing)) != 0:
            raise SystemExit("the quantised build failed")
        if rknn.export_rknn(str(out)) != 0:
            raise SystemExit("could not write the rknn")
        print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")

        # The simulator, so that "it converted" and "it still works" are two separate answers.
        if rknn.init_runtime() != 0:
            print("no simulator runtime; skipping the check", file=sys.stderr)
            return
        check = args.check or with_a_duck(args.reviewed, args.raw)
        if check is None:
            print("no reviewed frame with a box to check against; skipping", file=sys.stderr)
            rknn.release()
            return

        import numpy as np

        # An array, not a PIL image: the simulator refuses anything else, and says so in a line
        # that does not mention what it wanted.
        image = np.asarray(letterbox(Image.open(check).convert("RGB")), dtype=np.uint8)
        quantised = decode(rknn.inference(inputs=[image], data_format="nhwc")[0])
        rknn.release()

        # **Both models, one frame.** On its own, "the quantised model found nothing" could equally
        # mean the frame has no duck in it — and the first frame this picked by file size did not.
        # The float model is the control.
        float_boxes = float_inference(args.onnx, image)

        print(f"\ncheck on {check.parent.name}/{check.name}, a frame a person boxed:")
        print(f"  float onnx: {len(float_boxes)} box(es) after nms")
        print(f"  int8 rknn : {len(quantised)} box(es) after nms")

        # **Matched as sets, not best against best.** Two ducks in a frame come back in a different
        # order from the two models, because int8 scores land on their own scale — comparing the
        # top-scoring box of each said "8% overlap" about two boxes that were each correct, on
        # different ducks. What matters is whether every duck the float model finds is still found.
        matched, ious = 0, []
        remaining = list(quantised)
        for _, box in float_boxes:
            best, overlap = None, 0.0
            for index, (_, other) in enumerate(remaining):
                score = iou(box, other)
                if score > overlap:
                    best, overlap = index, score
            if best is not None and overlap >= 0.5:
                matched += 1
                ious.append(overlap)
                remaining.pop(best)
        if float_boxes:
            mean_iou = sum(ious) / len(ious) if ious else 0.0
            print(f"  {matched} of {len(float_boxes)} kept, mean overlap {mean_iou:.0%}")
        if float_boxes and matched < len(float_boxes):
            print(
                "\nQuantisation lost a detection. More calibration frames, or a hybrid build that\n"
                "leaves the detection head in float, are the two things to try."
            )
        float_scores = [f"{score:.2f}" for score, _ in float_boxes[:3]]
        int8_scores = [f"{score:.2f}" for score, _ in quantised[:3]]
        print(f"  scores: float {float_scores}  int8 {int8_scores}")
        print(
            "  the int8 output tensor carries its own scale, so a confidence threshold has to be\n"
            "  set against this model rather than inherited from the float one."
        )


if __name__ == "__main__":
    main()
