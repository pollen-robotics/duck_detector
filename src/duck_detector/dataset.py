"""Assemble the reviewed sessions into a YOLO dataset, split by session.

**The split is the whole reason this file exists.** Frames arrive at 2 Hz, so consecutive ones are
the same picture with the duck a centimetre further on. Split those at random and near-copies land
on both sides: the val score then measures memorisation, comes out high, and the model is bad on
the robot for reasons the numbers never showed. Whole sessions go to one side or the other.

    uv run dataset build                          # every reviewed session, newest held out
    uv run dataset build --val <session> ...      # or name them
    uv run dataset build --smoke                  # one session, split by frame, for plumbing only

Images are symlinked rather than copied: the frames stay the one copy in `datasets/raw/`, and a
rebuild costs nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

DATASETS = Path("datasets")
OUT = DATASETS / "yolo"
CLASS = "duck"


@dataclass
class Labelled:
    """One session's frames that have labels, and where they came from."""

    session: str
    frames: list[tuple[Path, Path]]  # (image, label)
    reviewed: bool

    @property
    def boxes(self) -> int:
        return sum(
            len([line for line in label.read_text().splitlines() if line.strip()])
            for _, label in self.frames
        )


def find(source: str) -> list[Labelled]:
    """Sessions with labels, preferring corrected ones over the pre-labeller's."""
    found = []
    for kind in ("reviewed", "labelled") if source == "auto" else (source,):
        for directory in sorted((DATASETS / kind).glob("*")):
            if not directory.is_dir() or any(f.session == directory.name for f in found):
                continue
            raw = DATASETS / "raw" / directory.name
            pairs = []
            for label in sorted(directory.glob("frame_*.txt")):
                image = raw / f"{label.stem}.jpg"
                if image.exists():
                    pairs.append((image, label))
            if pairs:
                found.append(Labelled(directory.name, pairs, reviewed=kind == "reviewed"))
    return found


def build(sessions: list[Labelled], val: set[str], smoke: bool) -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)

    counts = {"train": 0, "val": 0}
    for session in sessions:
        for index, (image, label) in enumerate(session.frames):
            # A smoke build has one session, so it splits by frame — every fifth to val. Wrong for
            # a number anybody quotes, which is why `data.yaml` says so in a comment.
            if smoke:
                split = "val" if index % 5 == 0 else "train"
            else:
                split = "val" if session.session in val else "train"
            for kind, source in (("images", image), ("labels", label)):
                target = OUT / split / kind / f"{session.session}__{source.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve())
            counts[split] += 1

    note = (
        "# SMOKE BUILD: one session split by frame. Consecutive frames are near-copies, so the\n"
        "# val score here measures memorisation. For plumbing only.\n"
        if smoke
        else "# Split by session: whole sessions are held out, never frames.\n"
    )
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "data.yaml").write_text(
        note + f"path: {OUT.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        f"  0: {CLASS}\n"
    )
    summary = {
        "train_frames": counts["train"],
        "val_frames": counts["val"],
        "smoke": smoke,
        "sessions": [
            {
                "session": s.session,
                "frames": len(s.frames),
                "boxes": s.boxes,
                "reviewed": s.reviewed,
                "split": "by-frame" if smoke else ("val" if s.session in val else "train"),
            }
            for s in sessions
        ],
    }
    (OUT / "build.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a YOLO dataset from reviewed sessions.")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", default="auto", choices=["auto", "reviewed", "labelled"])
    b.add_argument("--val", nargs="*", default=[], help="sessions to hold out; default the newest")
    b.add_argument(
        "--smoke",
        action="store_true",
        help="allow a single session, split by frame — plumbing only, never a number to quote",
    )
    args = parser.parse_args()

    sessions = find(args.source)
    if not sessions:
        raise SystemExit(
            "no labelled sessions. `uv run autolabel datasets/raw/<session>` writes the "
            "pre-labels, `uv run review` corrects them."
        )
    unreviewed = [s.session for s in sessions if not s.reviewed]
    if unreviewed:
        print(
            f"warning: using the pre-labeller's boxes for {', '.join(unreviewed)} — not corrected"
        )

    val = set(args.val)
    if not args.smoke:
        if len(sessions) < 2:
            raise SystemExit(
                "one session cannot be split by session, and splitting it by frame reports a score "
                "the model has not earned. Capture another session, or pass --smoke to build one "
                "for plumbing."
            )
        if not val:
            # Newest by name, which is a timestamp.
            val = {sorted(s.session for s in sessions)[-1]}
        missing = val - {s.session for s in sessions}
        if missing:
            raise SystemExit(f"no such session: {', '.join(sorted(missing))}")

    summary = build(sessions, val, args.smoke)
    print(json.dumps(summary, indent=2))
    print(f"\n{OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
