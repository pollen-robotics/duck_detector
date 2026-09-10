"""What goes to the Hub, decided without a network."""

from __future__ import annotations

import json
from pathlib import Path

from duck_detector import hub


def make(root: Path, session: str, *, reviewed: bool) -> None:
    raw = root / "raw" / session
    raw.mkdir(parents=True)
    (raw / "frame_00000.jpg").write_bytes(b"jpeg")
    (raw / "session.json").write_text(
        json.dumps(
            {"session": session, "tag": "kitchen", "robot": "graphite", "frames": 1, "hz": 2.0}
        )
    )
    if reviewed:
        out = root / "reviewed" / session
        out.mkdir(parents=True)
        (out / "frame_00000.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    labelled = root / "labelled" / session
    labelled.mkdir(parents=True)
    (labelled / "frame_00000.txt").write_text("0 0.5 0.5 0.2 0.3\n")
    (labelled / "sheet.png").write_bytes(b"png")


def test_sessions_are_read_off_the_hub_listing():
    files = {
        "README.md",
        "raw/20260101T000000Z_a_graphite/frame_00000.jpg",
        "raw/20260101T000000Z_a_graphite/session.json",
        "reviewed/20260101T000000Z_a_graphite/frame_00000.txt",
        "raw/README.md",
    }
    assert hub.sessions_of(files, "raw") == {"20260101T000000Z_a_graphite"}
    assert hub.sessions_of(files, "labelled") == set()


def test_a_push_plans_new_frames_and_replaces_labels(tmp_path, monkeypatch):
    """Frames already on the Hub are not sent twice; a stale label is deleted, a new one added."""
    root = tmp_path / "datasets"
    make(root, "20260101T000000Z_a_graphite", reviewed=True)
    make(root, "20260102T000000Z_b_graphite", reviewed=False)

    commits = []

    class Api:
        def create_repo(self, *a, **k):
            pass

        def create_commit(self, repo, *, repo_type, operations, commit_message):
            commits.append((commit_message, operations))

        def upload_file(self, **k):
            commits.append(("card", k["path_or_fileobj"].decode()))

    monkeypatch.setattr(hub, "api", lambda: Api())
    monkeypatch.setattr(
        hub,
        "hub_files",
        lambda repo: {
            "raw/20260101T000000Z_a_graphite/frame_00000.jpg",
            "raw/20260101T000000Z_a_graphite/session.json",
            "reviewed/20260101T000000Z_a_graphite/frame_00000.txt",
            "reviewed/20260101T000000Z_a_graphite/frame_00007.txt",
        },
    )

    pushed = hub.push_sessions(root=root, repo="x/y")
    assert pushed == ["20260101T000000Z_a_graphite", "20260102T000000Z_b_graphite"]

    first_message, first_ops = commits[0]
    paths = [(type(op).__name__, op.path_in_repo) for op in first_ops]
    # The frames stay; the corrections are what change.
    assert not any(p.startswith("raw/") for _, p in paths)
    assert (
        "CommitOperationDelete",
        "reviewed/20260101T000000Z_a_graphite/frame_00007.txt",
    ) in paths
    assert ("CommitOperationAdd", "reviewed/20260101T000000Z_a_graphite/frame_00000.txt") in paths
    assert "reviewed (1 files)" in first_message
    # The contact sheet is for looking at, and the Hub's viewer mistakes it for the dataset.
    assert not any(p.endswith("sheet.png") for _, p in paths)
    assert ("CommitOperationAdd", "labelled/20260101T000000Z_a_graphite/frame_00000.txt") in paths

    second_message, second_ops = commits[1]
    assert sorted(op.path_in_repo for op in second_ops) == [
        "labelled/20260102T000000Z_b_graphite/frame_00000.txt",
        "raw/20260102T000000Z_b_graphite/frame_00000.jpg",
        "raw/20260102T000000Z_b_graphite/session.json",
    ]

    kind, card = commits[2]
    assert kind == "card"
    assert "| `20260101T000000Z_a_graphite` | kitchen | graphite | 1 | 2.0 | yes |" in card
    assert "| `20260102T000000Z_b_graphite` | kitchen | graphite | 1 | 2.0 |  |" in card


def test_the_model_card_names_the_run_and_its_numbers(tmp_path):
    card = hub.model_card(
        "duck-v1",
        {"map50": 0.9764, "recall": 0.9355},
        [tmp_path / "best.onnx", tmp_path / "best.rknn"],
    )
    assert "**`duck-v1`**" in card
    assert "`duck_detect.onnx`, `duck_detect.rknn`" in card
    assert "| map50 | 0.9764 |" in card
    assert card.startswith("---\n"), "front matter, or the Hub renders a page with no metadata"
