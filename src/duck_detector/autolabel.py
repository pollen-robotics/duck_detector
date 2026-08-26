"""Pre-label a session with an open-vocabulary detector, for a person to correct.

Drawing the first thousand boxes by hand is the part of this that nobody does. An
open-vocabulary detector will not be right — it has never seen a Microduck — but it is right often
enough that correcting its boxes is a different job from drawing them, and it is the difference
between a dataset existing and not.

    uv sync --extra label
    uv run autolabel datasets/raw/<session> --sheet

Output is YOLO format, one `.txt` per frame beside a copy of nothing — the images stay where they
are and `images.txt` says which frame each label belongs to. Every box carries the score that
produced it in `labels.json`, because the correction pass wants to start with the confident ones
and the threshold is the first thing to argue with.

**`--sheet` renders the boxes onto a contact sheet.** Look at it before correcting anything: it is
how you find out that the prompt is describing the sofa.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Small, fast, and good at natural phrases. The alternative measured here was OWLv2, which wants
# noun phrases rather than descriptions and did worse on a robot it has never seen.
MODEL = "IDEA-Research/grounding-dino-tiny"

# **The prompt is the whole experiment, and every noun in it is a query.** Grounding DINO grounds
# each noun phrase separately, so "a small robot standing on the floor" asked it to find the floor
# as well — and it did, in a third of the frames, with a box the size of the frame. Noun phrases
# only, and nothing in them that is also furniture.
#
# Colour is deliberately absent: these robots come in blue, white and grey shells, so "yellow duck"
# finds the cushion instead.
PROMPT = "a small robot. a toy robot."

# Two phrases mean the same duck can come back twice, once per phrase. Class-agnostic, because
# there is one class here and the phrase that found it does not matter.
NMS_IOU = 0.5

# A box bigger than this fraction of the frame is the room, not a robot. It happens when a phrase
# grounds to the whole scene, and one such box in a training set teaches the model that everything
# is a duck.
MAX_AREA = 0.45


def pick_device() -> str:
    """CUDA if it can actually run a convolution, and say so when it cannot.

    On this machine the CUDA wheel ships cuDNN sublibraries that disagree with each other
    (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` out of the first conv, torch 2.13+cu130 against
    cuDNN 9.20/9.22), and it is not the loader path — a scrubbed `LD_LIBRARY_PATH` fails the same
    way. Convolutions run fine on CUDA *without* cuDNN, a little slower, so that is the fallback
    rather than dropping to the CPU: probe once, and take the fast path where the wheel is healthy.
    """
    if not torch.cuda.is_available():
        return "cpu"
    probe = lambda: torch.nn.functional.conv2d(  # noqa: E731
        torch.randn(1, 3, 32, 32, device="cuda"), torch.randn(4, 3, 3, 3, device="cuda")
    )
    try:
        probe()
        return "cuda"
    except RuntimeError as e:
        if "CUDNN" not in str(e).upper():
            raise
        torch.backends.cudnn.enabled = False
        try:
            probe()
        except RuntimeError:
            print("cuda cannot convolve at all; falling back to the cpu")
            return "cpu"
        print("cuda with cudnn disabled (the wheel's cudnn disagrees with itself)")
        return "cuda"


def load(device: str):
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to(device)
    model.eval()
    return processor, model


def selected_frames(session: Path, use_all: bool) -> list[Path]:
    """The frames triage picked, or everything if asked."""
    triage = session / "triage.json"
    if use_all or not triage.exists():
        return sorted(session.glob("frame_*.jpg"))
    picked = {f["frame"] for f in json.loads(triage.read_text())["frames"] if f["select"]}
    return [f for f in sorted(session.glob("frame_*.jpg")) if f.name in picked]


def suppress(found: list[tuple[list[float], float, str]]) -> list[tuple[list[float], float, str]]:
    """Keep the best box of each cluster, and drop the ones that are really the room.

    Written out rather than pulled from torchvision: it is fifteen lines, and this package has no
    other reason to depend on it.
    """

    def area(box: list[float]) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    def iou(a: list[float], b: list[float]) -> float:
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = area(a) + area(b) - overlap
        return overlap / union if union > 0 else 0.0

    kept: list[tuple[list[float], float, str]] = []
    for box, score, label in sorted(found, key=lambda f: -f[1]):
        if all(iou(box, other) < NMS_IOU for other, _, _ in kept):
            kept.append((box, score, label))
    return kept


@torch.no_grad()
def detect(processor, model, image: Image.Image, device: str, box_threshold: float):
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
    outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]],
    )[0]
    frame_area = image.width * image.height
    found = [
        (box.tolist(), float(score), label)
        for box, score, label in zip(
            result["boxes"], result["scores"], result["text_labels"], strict=False
        )
    ]
    found = [
        (box, score, label)
        for box, score, label in found
        if (box[2] - box[0]) * (box[3] - box[1]) / frame_area <= MAX_AREA
    ]
    return suppress(found)


def to_yolo(box: list[float], width: int, height: int) -> str:
    """`class cx cy w h`, normalised — one class, so always 0."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
    w, h = (x1 - x0) / width, (y1 - y0) / height
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def sheet(rows: list[tuple[Path, list]], out: Path, columns: int = 6, thumb: int = 220) -> None:
    """A contact sheet with the boxes drawn on, for looking at rather than for training."""
    cells = [(f, boxes) for f, boxes in rows]
    height = thumb * 16 // 9
    grid_rows = (len(cells) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb, grid_rows * height), "black")
    for index, (path, boxes) in enumerate(cells):
        image = Image.open(path)
        scale_x, scale_y = thumb / image.width, height / image.height
        small = image.resize((thumb, height))
        draw = ImageDraw.Draw(small)
        for box, score, label in boxes:
            x0, y0, x1, y1 = box
            draw.rectangle(
                [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y],
                outline="lime",
                width=2,
            )
            draw.text((x0 * scale_x + 2, y0 * scale_y + 2), f"{score:.2f} {label}", fill="lime")
        draw.text((4, 4), path.name.replace("frame_", "").replace(".jpg", ""), fill="yellow")
        canvas.paste(small, ((index % columns) * thumb, (index // columns) * height))
    canvas.save(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-label a session with an open-vocabulary detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("session", type=Path)
    parser.add_argument(
        "--out", type=Path, default=None, help="default datasets/labelled/<session>"
    )
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--all", action="store_true", help="ignore triage.json and do every frame")
    parser.add_argument("--sheet", action="store_true", help="render a contact sheet to look at")
    parser.add_argument("--limit", type=int, default=0, help="stop after N frames (a quick look)")
    args = parser.parse_args()

    frames = selected_frames(args.session, args.all)
    if args.limit:
        frames = frames[: args.limit]
    out = args.out or Path("datasets/labelled") / args.session.name
    out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"{MODEL} on {device}: {len(frames)} frames, threshold {args.threshold}")
    processor, model = load(device)

    records, rows, boxes_total = [], [], 0
    for path in frames:
        image = Image.open(path).convert("RGB")
        found = detect(processor, model, image, device, args.threshold)
        boxes_total += len(found)
        (out / f"{path.stem}.txt").write_text(
            "".join(to_yolo(box, image.width, image.height) + "\n" for box, _, _ in found)
        )
        records.append(
            {
                "frame": path.name,
                "boxes": [
                    {"box": box, "score": round(score, 3), "label": label}
                    for box, score, label in found
                ],
            }
        )
        rows.append((path, found))
        print(f"  {path.name}: {len(found)}")

    (out / "labels.json").write_text(
        json.dumps(
            {"model": MODEL, "prompt": PROMPT, "threshold": args.threshold, "frames": records},
            indent=2,
        )
        + "\n"
    )
    (out / "images.txt").write_text("".join(f"{p}\n" for p in frames))
    print(f"{boxes_total} boxes over {len(frames)} frames → {out}")

    if args.sheet:
        path = out / "sheet.png"
        sheet(rows, path)
        print(f"contact sheet → {path}")


if __name__ == "__main__":
    main()
