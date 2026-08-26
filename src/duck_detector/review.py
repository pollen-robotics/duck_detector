"""Correct the pre-labeller's boxes in Label Studio, and get them back as YOLO.

The pre-labeller is right most of the time, which makes this the only stage that costs a person —
so the job is to make it *only* correction: boxes already drawn, one key to accept, the wrong ones
deleted rather than everything drawn from nothing.

    uv run review prepare datasets/raw/<session>     # tasks + config + the command to run
    uv run review serve                              # start Label Studio on this repo's datasets
    uv run review import <export.json>               # corrections back, as YOLO labels

**The trick that makes local files work.** Label Studio will not read a path from disk unless it is
serving one: `LOCAL_FILES_SERVING_ENABLED=true` plus a document root, and tasks that point at
`/data/local-files/?d=<relative path>`. Both are set by `serve`, and `prepare` writes the tasks in
that shape — so importing is drag-and-drop and nothing has to be configured in the UI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Where everything lives, relative to the repo root. Label Studio's document root is the parent of
# both, so one setting covers raw frames and anything derived.
DATASETS = Path("datasets")
REVIEW = DATASETS / "review"
REVIEWED = DATASETS / "reviewed"

# One class. `duck` rather than `robot`, because the thing being found is a Microduck and a Reachy
# Mini in the background is a negative, not a smaller duck.
CLASS = "duck"

LABEL_CONFIG = f"""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image">
    <Label value="{CLASS}" background="#00c853"/>
  </RectangleLabels>
</View>
"""


def local_files_url(path: Path) -> str:
    """The URL Label Studio serves a local file at, relative to the document root.

    Resolved on both sides, because the path may arrive either way: `datasets/raw/x` from a shell
    and `/home/…/datasets/raw/x` from a tab-completion, and only one of those is relative to
    `DATASETS` as written.
    """
    root = (Path.cwd() / DATASETS).resolve()
    return f"/data/local-files/?d={path.resolve().relative_to(root)}"


def prepare(session: Path, labels: Path | None) -> Path:
    """Tasks for one session, with the pre-labeller's boxes as predictions to accept or fix."""
    labels = labels or DATASETS / "labelled" / session.name
    predictions = {}
    labels_json = labels / "labels.json"
    if labels_json.exists():
        loaded = json.loads(labels_json.read_text())
        predictions = {f["frame"]: f["boxes"] for f in loaded["frames"]}
        model = loaded.get("model", "unknown")
    else:
        model = "none"
        print(f"no {labels_json}; tasks will have nothing pre-drawn", file=sys.stderr)

    # Only the frames triage chose, when it has run: the rest are floor, and a person should not be
    # asked to confirm that fifty times.
    triage = session / "triage.json"
    if triage.exists():
        wanted = {f["frame"] for f in json.loads(triage.read_text())["frames"] if f["select"]}
    else:
        wanted = {p.name for p in session.glob("frame_*.jpg")}

    from PIL import Image

    tasks = []
    for frame in sorted(session.glob("frame_*.jpg")):
        if frame.name not in wanted:
            continue
        width, height = Image.open(frame).size
        results = []
        for index, box in enumerate(predictions.get(frame.name, [])):
            x0, y0, x1, y1 = box["box"]
            results.append(
                {
                    "id": f"pre{index}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    # Label Studio speaks percentages of the image, not pixels.
                    "value": {
                        "x": 100 * x0 / width,
                        "y": 100 * y0 / height,
                        "width": 100 * (x1 - x0) / width,
                        "height": 100 * (y1 - y0) / height,
                        "rotation": 0,
                        "rectanglelabels": [CLASS],
                    },
                    "score": box["score"],
                }
            )
        tasks.append(
            {
                "data": {"image": local_files_url(frame)},
                "meta": {"session": session.name, "frame": frame.name},
                "predictions": [{"model_version": model, "result": results}] if results else [],
            }
        )

    out = REVIEW / session.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    (out / "label_config.xml").write_text(LABEL_CONFIG)
    drawn = sum(len(t["predictions"][0]["result"]) for t in tasks if t["predictions"])
    print(f"{len(tasks)} tasks, {drawn} boxes pre-drawn → {out}/tasks.json")
    print(
        "\nNext:\n"
        "  uv run review serve                      # in another terminal, leave it running\n"
        "  then in the browser: new project → Labeling setup → paste "
        f"{out}/label_config.xml\n"
        f"  → Import → {out}/tasks.json\n"
        "  correct, then Export as JSON and:\n"
        "  uv run review import <the export>.json"
    )
    return out


def serve(port: int) -> None:
    """Label Studio, pointed at this repo's datasets so tasks can reference frames on disk."""
    root = (Path.cwd() / DATASETS).resolve()
    env = os.environ | {
        "LOCAL_FILES_SERVING_ENABLED": "true",
        "LOCAL_FILES_DOCUMENT_ROOT": str(root),
    }
    print(f"serving {root} on http://localhost:{port}", file=sys.stderr)
    raise SystemExit(
        subprocess.run(
            ["label-studio", "start", "--port", str(port), "--no-browser"], env=env, check=False
        ).returncode
    )


def import_export(export: Path) -> None:
    """Corrections back out of Label Studio, as YOLO labels beside the frames they belong to.

    Reads the full JSON export (the one with `annotations`), because that is the only shape that
    carries both the boxes and which frame they came from. A task a person never opened has no
    annotation and is skipped rather than written as an empty label — "nobody looked at this" and
    "there is nothing here" are different, and only one of them is training data.
    """
    tasks = json.loads(export.read_text())
    written = kept = empty = 0
    per_session: dict[str, int] = {}

    for task in tasks:
        meta = task.get("meta") or {}
        session, frame = meta.get("session"), meta.get("frame")
        if not session or not frame:
            # Fall back to the image path, for a task list that lost its meta on the way through.
            image = task.get("data", {}).get("image", "")
            parts = Path(image.split("?d=")[-1]).parts
            if len(parts) < 2:
                continue
            session, frame = parts[-2], parts[-1]

        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue
        results = annotations[-1].get("result", [])

        lines = []
        for result in results:
            if result.get("type") != "rectanglelabels":
                continue
            value = result["value"]
            # Percentages back to normalised centre-form, which is what YOLO reads.
            cx = (value["x"] + value["width"] / 2) / 100
            cy = (value["y"] + value["height"] / 2) / 100
            lines.append(
                f"0 {cx:.6f} {cy:.6f} {value['width'] / 100:.6f} {value['height'] / 100:.6f}"
            )

        out = REVIEWED / session
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{Path(frame).stem}.txt").write_text("".join(line + "\n" for line in lines))
        written += 1
        kept += len(lines)
        empty += 1 if not lines else 0
        per_session[session] = per_session.get(session, 0) + 1

    print(f"{written} frames reviewed, {kept} boxes, {empty} of them deliberately empty")
    for session, count in sorted(per_session.items()):
        print(f"  {session}: {count} → {REVIEWED / session}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct pre-labels in Label Studio.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="tasks + labelling config for one session")
    prep.add_argument("session", type=Path)
    prep.add_argument(
        "--labels", type=Path, default=None, help="default datasets/labelled/<session>"
    )

    up = sub.add_parser("serve", help="run Label Studio against this repo's datasets")
    up.add_argument("--port", type=int, default=8080)

    imp = sub.add_parser("import", help="read a Label Studio JSON export back as YOLO labels")
    imp.add_argument("export", type=Path)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.session, args.labels)
    elif args.command == "serve":
        serve(args.port)
    else:
        import_export(args.export)


if __name__ == "__main__":
    main()
