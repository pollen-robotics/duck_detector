"""Scoring a model's boxes against the corrections, without a model."""

import json

from PIL import Image

from duck_detector import watch


def test_the_score_matches_boxes_at_iou_half(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = tmp_path / "datasets/raw/20260101T000000Z_a_duck"
    reviewed = tmp_path / "datasets/reviewed/20260101T000000Z_a_duck"
    session.mkdir(parents=True)
    reviewed.mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (100, 200), "grey").save(session / f"frame_{i:05d}.jpg")
    # One duck at the centre; an empty frame; a duck the model will miss.
    (reviewed / "frame_00000.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (reviewed / "frame_00001.txt").write_text("")
    (reviewed / "frame_00002.txt").write_text("0 0.2 0.2 0.1 0.1\n")

    predictions = {
        # 40..60 × 80..120 in pixels is exactly the truth box.
        "frame_00000.jpg": [((40, 80, 60, 120), 0.9)],
        # A false box on the empty frame.
        "frame_00001.jpg": [((0, 0, 20, 20), 0.6)],
        "frame_00002.jpg": [],
    }
    numbers = watch.score(session, predictions)
    assert numbers == {
        "frames": 3,
        "real": 2,
        "predicted": 2,
        "hit": 1,
        "missed": 1,
        "false": 1,
        "precision": 0.5,
        "recall": 0.5,
    }
    assert watch.score(tmp_path / "datasets/raw/nothing", {}) is None, "no corrections, no score"


def test_the_held_out_session_comes_from_the_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "build.json").write_text(
        json.dumps(
            {"sessions": [{"session": "a", "split": "train"}, {"session": "b", "split": "val"}]}
        )
    )
    assert watch.held_out(run).name == "b"
    assert watch.held_out(tmp_path / "none") is None
