"""How much of the pre-labeller's work a person had to undo.

    uv run agreement datasets/raw/<session>

This is the number that decides whether the pre-labeller is worth running at all. If most frames
come out of the correction pass untouched, it can be pointed at hundreds of frames and the human
job stays "glance and press Ctrl+Enter". If a third of the boxes get moved or deleted, the prompt or
the threshold wants work before anybody spends an afternoon in an editor.

Measured on the first session: 88% of frames untouched, 89% precision, 98% recall. Which is why the
threshold stays low — the one box it missed cost more attention than the six it invented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = Path("datasets")

# A box this close to the pre-label was accepted rather than redrawn. Deliberately strict: the
# question is what a person had to *touch*, and a nudge is a touch.
SAME = 0.9

# Below this, a correction is a different box rather than a moved one — the pre-labeller found
# something else, and the person drew the duck.
RELATED = 0.4


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    union = area(a) + area(b) - overlap
    return overlap / union if union > 0 else 0.0


def yolo_to_xyxy(line: str) -> tuple[float, float, float, float]:
    _, cx, cy, w, h = (float(v) for v in line.split())
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def compare(session: Path) -> dict:
    from PIL import Image

    pre_path = DATASETS / "labelled" / session.name / "labels.json"
    reviewed = DATASETS / "reviewed" / session.name
    if not pre_path.exists():
        raise SystemExit(f"no pre-labels at {pre_path}")
    if not reviewed.is_dir():
        raise SystemExit(f"nothing reviewed yet at {reviewed}")

    pre = json.loads(pre_path.read_text())
    guesses_by_frame = {f["frame"]: f["boxes"] for f in pre["frames"]}

    counts = {"kept": 0, "nudged": 0, "added": 0, "deleted": 0}
    frames = untouched = empty = 0

    for label_file in sorted(reviewed.glob("frame_*.txt")):
        frames += 1
        image = session / f"{label_file.stem}.jpg"
        width, height = Image.open(image).size
        truth = [yolo_to_xyxy(line) for line in label_file.read_text().splitlines() if line.strip()]
        guesses = [
            (b["box"][0] / width, b["box"][1] / height, b["box"][2] / width, b["box"][3] / height)
            for b in guesses_by_frame.get(image.name, [])
        ]
        if not truth:
            empty += 1

        matched: set[int] = set()
        changed = False
        for box in truth:
            best, score = None, 0.0
            for index, guess in enumerate(guesses):
                if index in matched:
                    continue
                overlap = iou(box, guess)
                if overlap > score:
                    best, score = index, overlap
            if best is not None and score >= SAME:
                counts["kept"] += 1
                matched.add(best)
            elif best is not None and score >= RELATED:
                counts["nudged"] += 1
                matched.add(best)
                changed = True
            else:
                counts["added"] += 1
                changed = True

        left_over = len(guesses) - len(matched)
        counts["deleted"] += left_over
        changed = changed or bool(left_over)
        untouched += 0 if changed else 1

    drawn = sum(len(v) for v in guesses_by_frame.values())
    real = counts["kept"] + counts["nudged"] + counts["added"]
    right = counts["kept"] + counts["nudged"]
    return {
        "session": session.name,
        "frames": frames,
        "empty_frames": empty,
        "frames_untouched": untouched,
        "boxes": counts,
        "prelabelled": drawn,
        "real": real,
        "precision": round(right / drawn, 3) if drawn else 0.0,
        "recall": round(right / real, 3) if real else 0.0,
        "prompt": pre.get("prompt"),
        "threshold": pre.get("threshold"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the pre-labels with the corrections.")
    parser.add_argument("session", type=Path, help="a datasets/raw/<session> directory")
    parser.add_argument("--json", action="store_true", help="print the numbers and nothing else")
    args = parser.parse_args()

    result = compare(args.session)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    frames, untouched = result["frames"], result["frames_untouched"]
    counts = result["boxes"]
    print(f"{result['session']}")
    print(f"  frames reviewed      {frames}  ({result['empty_frames']} deliberately empty)")
    print(f"  frames untouched     {untouched}  ({100 * untouched / max(frames, 1):.0f}%)")
    print(f"  boxes accepted       {counts['kept']}")
    print(f"  boxes nudged         {counts['nudged']}")
    print(f"  boxes drawn anew     {counts['added']}   ← what the pre-labeller missed")
    print(f"  boxes deleted        {counts['deleted']}   ← what it invented")
    print(
        f"  precision {result['precision']:.0%}  recall {result['recall']:.0%}"
        f"   ({result['prelabelled']} drawn, {result['real']} real)"
    )
    print(f"  threshold {result['threshold']}  prompt {result['prompt']!r}")


if __name__ == "__main__":
    main()
