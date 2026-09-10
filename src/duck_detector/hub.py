"""The Hugging Face Hub as the place the data and the models live.

The repository carries the recipe; the Hub carries the rest, in two repos:

* **the dataset** `pollen-robotics/microduck-duck-detector-dataset` — `raw/<session>/` frames and
  `session.json`, `labelled/<session>/` the pre-labeller's boxes, `reviewed/<session>/` what a
  person corrected. The tree is `datasets/` here, minus the two directories that are scratch
  (`review/`, Label Studio's tasks, and `yolo/`, what `dataset build` assembles from the rest).
* **the model** `pollen-robotics/microduck-duck-detector` — `duck_detect.pt`, `.onnx` and `.rknn`
  at the root under fixed names, with one git tag per training run (`duck-v1`, …). A robot that
  wants the current detector downloads `duck_detect.rknn` from `main`; one that wants a known one
  names the tag. The run's `summary.json`, `results.csv` and `args.yaml` ride along so a number in
  the model card can be traced to the run that produced it.

Sessions are pushed whole and never rewritten: a frame that reached the Hub is a frame somebody
may have labelled, and labels name frames by file. Corrections are the one thing that changes, and
they change by session, so `reviewed/<session>/` is replaced as a unit.

`DUCK_DATASET_REPO` and `DUCK_MODEL_REPO` override the repo names, for a fork or a scratch account.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

DATASET_REPO = os.environ.get(
    "DUCK_DATASET_REPO", "pollen-robotics/microduck-duck-detector-dataset"
)
MODEL_REPO = os.environ.get("DUCK_MODEL_REPO", "pollen-robotics/microduck-duck-detector")

DATASETS = Path("datasets")
# What of `datasets/` is data rather than scratch, in the order it is produced.
KINDS = ("raw", "labelled", "reviewed")

WEIGHTS = Path("weights")
# The names on the Hub, and what `deploy/robotd.toml` on the robot calls them.
MODEL_STEM = "duck_detect"
MODEL_FILES = ("best.pt", "best.onnx", "best.rknn")
RUN_FILES = ("summary.json", "build.json", "results.csv", "args.yaml")


def api():
    from huggingface_hub import HfApi

    return HfApi()


def ensure_repo(repo: str, repo_type: str, *, private: bool) -> None:
    """Create the repo if it is not there. Private unless asked: frames of somebody's room."""
    api().create_repo(repo, repo_type=repo_type, private=private, exist_ok=True)


# ── sessions ─────────────────────────────────────────────────────────────────


def local_sessions(root: Path = DATASETS) -> list[str]:
    return sorted(p.name for p in (root / "raw").glob("*") if (p / "session.json").exists())


def hub_files(repo: str = DATASET_REPO) -> set[str]:
    """Every path the dataset repo holds — so a push sends what is missing and nothing twice."""
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        return set(api().list_repo_files(repo, repo_type="dataset"))
    except RepositoryNotFoundError:
        return set()


def sessions_of(files: set[str], kind: str) -> set[str]:
    return {
        path.split("/")[1] for path in files if path.startswith(f"{kind}/") and path.count("/") >= 2
    }


def session_record(session: str, root: Path = DATASETS) -> dict[str, Any]:
    path = root / "raw" / session / "session.json"
    return json.loads(path.read_text()) if path.exists() else {"session": session}


def push_sessions(
    sessions: list[str] | None = None,
    *,
    repo: str = DATASET_REPO,
    root: Path = DATASETS,
    private: bool = True,
    force: bool = False,
) -> list[str]:
    """Upload sessions — every kind that exists locally — in one commit per session.

    Frames already on the Hub are not sent again (`force` sends them anyway); corrections are, so
    re-running after a review updates the labels and nothing else.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    wanted = sessions or local_sessions(root)
    if not wanted:
        raise SystemExit(f"no sessions under {root / 'raw'}")
    ensure_repo(repo, "dataset", private=private)
    on_hub = hub_files(repo)
    hf = api()

    pushed = []
    for session in wanted:
        operations: list[Any] = []
        parts = []
        for kind in KINDS:
            directory = root / kind / session
            if not directory.is_dir():
                continue
            # Frames are immutable once pushed; labels are replaced whole, which is an upload of
            # what is local plus a delete of whatever the Hub has that is not local any more.
            if kind == "raw" and session in sessions_of(on_hub, "raw") and not force:
                continue
            prefix = f"{kind}/{session}/"
            local = {
                prefix + path.relative_to(directory).as_posix(): path
                for path in directory.rglob("*")
                if path.is_file()
            }
            for stale in sorted(p for p in on_hub if p.startswith(prefix) and p not in local):
                operations.append(CommitOperationDelete(path_in_repo=stale))
            for in_repo, path in sorted(local.items()):
                operations.append(
                    CommitOperationAdd(path_in_repo=in_repo, path_or_fileobj=str(path))
                )
            parts.append(f"{kind} ({len(local)} files)")
        if not operations:
            print(f"   {session}: already on the Hub", file=sys.stderr)
            continue
        message = f"{session}: {', '.join(parts)}"
        print(f"== pushing {message}", file=sys.stderr)
        hf.create_commit(repo, repo_type="dataset", operations=operations, commit_message=message)
        pushed.append(session)

    # The card last, from everything now local — after a pull, that is everything there is.
    hf.upload_file(
        path_or_fileobj=dataset_card(root, repo).encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message="dataset card",
    )
    print(f"== https://huggingface.co/datasets/{repo}", file=sys.stderr)
    return pushed


def pull_sessions(
    sessions: list[str] | None = None,
    *,
    repo: str = DATASET_REPO,
    root: Path = DATASETS,
) -> Path:
    """Bring sessions down into `datasets/`, in the same layout, only what is missing or changed."""
    from huggingface_hub import snapshot_download

    patterns = (
        [f"{kind}/{session}/*" for session in sessions for kind in KINDS]
        if sessions
        else [f"{kind}/*" for kind in KINDS]
    )
    path = snapshot_download(
        repo, repo_type="dataset", local_dir=str(root), allow_patterns=patterns
    )
    print(f"== {len(local_sessions(root))} sessions in {root}", file=sys.stderr)
    return Path(path)


def dataset_card(root: Path, repo: str) -> str:
    """A README for the dataset repo, with the sessions table regenerated from `session.json`."""
    rows = []
    for session in local_sessions(root):
        record = session_record(session, root)
        labelled = (root / "reviewed" / session).is_dir()
        rows.append(
            f"| `{session}` | {record.get('tag', '')} | {record.get('robot', '')} | "
            f"{record.get('frames', '')} | {record.get('hz', '')} | "
            f"{'yes' if labelled else ''} | {record.get('note', '')} |"
        )
    table = "\n".join(rows) if rows else "| — | | | | | | |"
    return f"""---
license: apache-2.0
task_categories:
  - object-detection
tags:
  - robotics
  - microduck
pretty_name: Microduck duck detector
---

# Microducks, seen by a Microduck

Frames from a Microduck's own camera — a wide lens 20 cm off the floor, turned upright by the
robot before anything sees it — with bounding boxes on the other Microducks in view. One class,
`duck`. Captured, labelled and trained with
[pollen-robotics/duck_detector](https://github.com/pollen-robotics/duck_detector); the model that
comes out of it is [`{MODEL_REPO}`](https://huggingface.co/{MODEL_REPO}).

## Layout

```
raw/<session>/frame_00000.jpg …   session.json    what the camera saw, and the provenance
labelled/<session>/frame_*.txt    labels.json     the pre-labeller's boxes (Grounding DINO)
reviewed/<session>/frame_*.txt                    what a person corrected — the training labels
```

Labels are YOLO format: `class cx cy w h`, normalised. An empty file is a frame with no duck in
it, kept on purpose — a detector that has never seen an empty room finds ducks in curtains.

**Sessions are the unit.** Frames arrive at 2 Hz, so consecutive frames are near-copies; anything
that splits this data splits it by session, never by frame.

## Sessions

| session | tag | robot | frames | Hz | reviewed | note |
|---|---|---|---|---|---|---|
{table}

```bash
uv run dataset pull          # everything, into datasets/
uv run dataset push          # every local session the Hub does not have yet
```
"""


# ── models ───────────────────────────────────────────────────────────────────


def push_model(
    run: Path,
    *,
    tag: str | None = None,
    repo: str = MODEL_REPO,
    private: bool = True,
) -> str:
    """Upload a training run's weights under the fixed names, and tag the commit after the run.

    Whatever of `.pt`, `.onnx` and `.rknn` exists is sent — the RKNN arrives later than the other
    two, from `scripts/to_rknn.py`, and pushing the same run again adds it under the same tag by
    moving the tag. A robot reads `duck_detect.rknn` off `main`, so the last push is the current
    detector; the tags are how to get back to the previous one.
    """
    from huggingface_hub import CommitOperationAdd

    weights = run / "weights"
    found = [weights / name for name in MODEL_FILES if (weights / name).exists()]
    if not found:
        raise SystemExit(f"no weights in {weights} — `uv run train --export` writes them")
    tag = tag or run.name
    ensure_repo(repo, "model", private=private)

    operations = [
        CommitOperationAdd(path_in_repo=f"{MODEL_STEM}{path.suffix}", path_or_fileobj=str(path))
        for path in found
    ]
    for name in RUN_FILES:
        if (run / name).exists():
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"runs/{tag}/{name}", path_or_fileobj=str(run / name)
                )
            )
    summary = (
        json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    )
    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md", path_or_fileobj=model_card(tag, summary, found).encode()
        )
    )
    message = f"{tag}: {', '.join(p.name for p in found)}"
    print(f"== pushing {message}", file=sys.stderr)
    hf = api()
    commit = hf.create_commit(
        repo, repo_type="model", operations=operations, commit_message=message
    )
    # Moved rather than refused when it exists: the tag names the run, and the run grows an .rknn.
    hf.create_tag(repo, tag=tag, repo_type="model", revision=commit.oid, exist_ok=True)
    print(f"== https://huggingface.co/{repo}/tree/{tag}", file=sys.stderr)
    return commit.oid


def pull_model(
    *,
    revision: str | None = None,
    repo: str = MODEL_REPO,
    into: Path = WEIGHTS,
) -> list[Path]:
    """The current detector (or a tagged one) into `weights/`, every format the Hub has."""
    from huggingface_hub import hf_hub_download, list_repo_files

    into.mkdir(parents=True, exist_ok=True)
    names = [f for f in list_repo_files(repo, revision=revision) if f.startswith(f"{MODEL_STEM}.")]
    if not names:
        raise SystemExit(f"nothing called {MODEL_STEM}.* in {repo}@{revision or 'main'}")
    paths = [
        Path(hf_hub_download(repo, name, revision=revision, local_dir=str(into))) for name in names
    ]
    for path in paths:
        print(f"   {path}  ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return paths


def model_card(tag: str, summary: dict[str, Any], files: list[Path]) -> str:
    sessions = summary.get("sessions") or []
    val = set(summary.get("val_sessions") or [])
    trained_on = (
        "\n".join(f"- `{s}`{' (val)' if s in val else ''}" for s in sessions)
        if sessions
        else "- not recorded for this run"
    )
    metrics = "\n".join(
        f"| {k} | {v} |" for k, v in summary.items() if k not in ("run", "sessions", "val_sessions")
    )
    formats = ", ".join(f"`{MODEL_STEM}{p.suffix}`" for p in files)
    return f"""---
license: apache-2.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - robotics
  - microduck
  - yolo
  - rknn
datasets:
  - {DATASET_REPO}
---

# Microduck duck detector

Finds other Microducks in a Microduck's camera. One class, `duck`, 320×320 letterboxed input,
2100 candidate boxes out; trained with
[pollen-robotics/duck_detector](https://github.com/pollen-robotics/duck_detector) on
[`{DATASET_REPO}`](https://huggingface.co/datasets/{DATASET_REPO}) and run on the robot by
`duck-detect` in [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck).

Current run: **`{tag}`** — {formats}. Every run is a git tag on this repo; `main` is the latest.

| metric | value |
|---|---|
{metrics}

Trained on these sessions of the dataset, whole sessions held out for validation:

{trained_on}

`duck_detect.rknn` is INT8 for the RK3566's NPU, quantised against real frames; `.onnx` is the
float model it came from (static shapes, opset 12), and `.pt` is the ultralytics checkpoint.
**The int8 output carries its own scale**, so a confidence threshold has to be set against the
RKNN, not inherited from the float model.

```bash
uv run model pull                 # weights/duck_detect.* from main
uv run model pull --revision {tag}
```
"""
