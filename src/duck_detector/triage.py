"""Decide which frames of a session are worth labelling.

A duck walking around films the floor. Of the first session's 99 frames, most are a soft grey
blur — the camera 20 cm up, pointed slightly down, at 2 Hz while the robot walks — and a handful
have a room and another duck in them. Labelling all of it wastes the only expensive thing in this
pipeline, which is a person looking at pictures.

Two numbers per frame, both cheap and both from Pillow alone:

* **sharpness** — variance of a Laplacian. Motion blur is the dominant defect here and it is what
  makes a box wrong rather than merely loose.
* **content** — mean edge energy. A floor-only frame has almost none; a room, a sofa, a duck have
  plenty.

Neither is a detector. They are a *sort order*: label the frames most likely to contain something,
keep a deliberate sample of the empty ones as negatives, and record the numbers so a later pass can
argue with these thresholds instead of guessing.

    uv run triage datasets/raw/<session>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat

# Frames are scored at this size, not full resolution: the numbers are relative to each other, and
# a 720x1280 JPEG decode per frame is the slow part of an otherwise instant pass.
SCORE_SIZE = (180, 320)

# A 3x3 Laplacian. Variance of the response is the standard cheap sharpness proxy.
LAPLACIAN = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)


@dataclass
class Scored:
    frame: str
    sharpness: float
    content: float
    select: bool
    reason: str


def score(path: Path) -> tuple[float, float]:
    """Sharpness and content for one frame."""
    grey = Image.open(path).convert("L").resize(SCORE_SIZE)
    sharpness = ImageStat.Stat(grey.filter(LAPLACIAN)).var[0]
    content = ImageStat.Stat(grey.filter(ImageFilter.FIND_EDGES)).mean[0]
    return sharpness, content


def triage(session: Path, *, want: int, negatives: int) -> list[Scored]:
    frames = sorted(session.glob("frame_*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {session}")

    scored = [(f, *score(f)) for f in frames]

    # Content decides what is worth a person's attention; sharpness breaks ties, because between
    # two frames of the same scene the sharp one is the one a box can be drawn on.
    ranked = sorted(scored, key=lambda s: (s[2], s[1]), reverse=True)
    chosen = {f for f, _, _ in ranked[:want]}

    # Negatives on purpose, spread across the session rather than taken from one stretch of
    # staring at the same floorboard: a detector that has never seen an empty room finds ducks in
    # curtains, and these frames cost nothing to label because there is nothing in them.
    empties = [f for f, _, _ in ranked[want:]]
    step = max(1, len(empties) // max(1, negatives))
    chosen |= set(empties[::step][:negatives])

    out = []
    for frame, sharpness, content in scored:
        picked = frame in chosen
        if picked:
            reason = "content" if content >= ranked[min(want, len(ranked)) - 1][2] else "negative"
        else:
            reason = "empty"
        out.append(
            Scored(
                frame=frame.name,
                sharpness=round(sharpness, 2),
                content=round(content, 2),
                select=picked,
                reason=reason,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank a session's frames and pick the ones worth labelling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("session", type=Path, help="a datasets/raw/<session> directory")
    parser.add_argument("--want", type=int, default=40, help="frames to put in front of a person")
    parser.add_argument("--negatives", type=int, default=10, help="empty frames to keep as well")
    args = parser.parse_args()

    scored = triage(args.session, want=args.want, negatives=args.negatives)
    path = args.session / "triage.json"
    path.write_text(
        json.dumps(
            {
                "want": args.want,
                "negatives": args.negatives,
                "frames": [asdict(s) for s in scored],
            },
            indent=2,
        )
        + "\n"
    )
    picked = [s for s in scored if s.select]
    print(f"{len(picked)} of {len(scored)} frames selected → {path}")
    for s in sorted(picked, key=lambda s: -s.content)[:5]:
        print(f"  {s.frame}  content {s.content:6.2f}  sharpness {s.sharpness:8.1f}  {s.reason}")


if __name__ == "__main__":
    main()
