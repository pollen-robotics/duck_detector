"""The rules that must not quietly stop holding."""

import json
import sys

import pytest
from PIL import Image

from duck_detector import dataset, review


def make_session(root, name, frames=4, labelled=True):
    """A session on disk: frames, and the pre-labeller's boxes for them."""
    raw = root / "datasets" / "raw" / name
    raw.mkdir(parents=True)
    for i in range(frames):
        Image.new("RGB", (720, 1280), "grey").save(raw / f"frame_{i:05d}.jpg")
    if labelled:
        out = root / "datasets" / "labelled" / name
        out.mkdir(parents=True)
        records = []
        for i in range(frames):
            (out / f"frame_{i:05d}.txt").write_text("0 0.5 0.5 0.2 0.3\n")
            records.append(
                {
                    "frame": f"frame_{i:05d}.jpg",
                    "boxes": [{"box": [216, 448, 360, 832], "score": 0.5}],
                }
            )
        (out / "labels.json").write_text(
            json.dumps({"model": "test", "prompt": "p", "threshold": 0.3, "frames": records})
        )
    return raw


def test_one_session_cannot_be_split_by_session(tmp_path, monkeypatch):
    """The discipline this whole file exists for.

    Frames arrive at 2 Hz, so consecutive ones are near-copies. Splitting a single session by frame
    puts copies on both sides and the val score becomes memorisation — high, and wrong. It has to
    be refused rather than warned about, because a warning in a log is not what stops somebody
    quoting the number.
    """
    monkeypatch.chdir(tmp_path)
    make_session(tmp_path, "20260101T000000Z_a_duck")
    monkeypatch.setattr(sys, "argv", ["dataset", "build"])
    with pytest.raises(SystemExit) as raised:
        dataset.main()
    assert "one session" in str(raised.value)

    # And `--smoke` allows it, but stamps the dataset so the numbers cannot be quoted innocently.
    monkeypatch.setattr(sys, "argv", ["dataset", "build", "--smoke"])
    dataset.main()
    assert "SMOKE" in (tmp_path / "datasets/yolo/data.yaml").read_text()


def test_a_split_holds_whole_sessions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_session(tmp_path, "20260101T000000Z_a_duck")
    make_session(tmp_path, "20260102T000000Z_b_duck")
    monkeypatch.setattr(sys, "argv", ["dataset", "build"])
    dataset.main()

    build = json.loads((tmp_path / "datasets/yolo/build.json").read_text())
    val = {s["session"] for s in build["sessions"] if s["split"] == "val"}
    train = {s["session"] for s in build["sessions"] if s["split"] == "train"}
    assert val and train and not (val & train), build
    # The newest is held out when nobody names one, so a rebuild is deterministic.
    assert val == {"20260102T000000Z_b_duck"}
    # Images are symlinked: the frames stay the one copy in raw/.
    images = list((tmp_path / "datasets/yolo/train/images").iterdir())
    assert images and all(p.is_symlink() for p in images)


def test_the_review_round_trip_keeps_the_boxes(tmp_path, monkeypatch):
    """Out to Label Studio as percentages, back as normalised centre-form.

    Two conversions with two conventions, and nothing downstream would notice a mistake in either:
    a box off by a factor is still a box, and the model would just be worse.
    """
    monkeypatch.chdir(tmp_path)
    name = "20260101T000000Z_a_duck"
    make_session(tmp_path, name)
    review.prepare(tmp_path / "datasets/raw" / name, tmp_path / "datasets/labelled" / name)

    tasks = json.loads((tmp_path / "datasets/review" / name / "tasks.json").read_text())
    assert len(tasks) == 4
    box = tasks[0]["predictions"][0]["result"][0]["value"]
    # 216/720 and 448/1280 as percentages.
    assert box["x"] == pytest.approx(30.0)
    assert box["y"] == pytest.approx(35.0)
    assert box["width"] == pytest.approx(20.0)
    assert box["height"] == pytest.approx(30.0)
    assert box["rectanglelabels"] == ["duck"]

    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            [
                {
                    "meta": {"session": name, "frame": "frame_00000.jpg"},
                    "annotations": [
                        {
                            "result": [
                                {
                                    "type": "rectanglelabels",
                                    "value": {
                                        "x": 30.0,
                                        "y": 35.0,
                                        "width": 20.0,
                                        "height": 30.0,
                                        "rectanglelabels": ["duck"],
                                    },
                                }
                            ]
                        }
                    ],
                },
                # A frame somebody opened and left empty: that is data.
                {
                    "meta": {"session": name, "frame": "frame_00001.jpg"},
                    "annotations": [{"result": []}],
                },
                # A frame nobody opened: that is not.
                {"meta": {"session": name, "frame": "frame_00002.jpg"}, "annotations": []},
            ]
        )
    )
    review.import_export(export)

    reviewed = tmp_path / "datasets/reviewed" / name
    written = sorted(p.name for p in reviewed.glob("*.txt"))
    assert written == ["frame_00000.txt", "frame_00001.txt"], "an unopened task is not a label"
    cx, cy, w, h = (float(v) for v in reviewed.joinpath("frame_00000.txt").read_text().split()[1:])
    assert (cx, cy, w, h) == pytest.approx((0.40, 0.50, 0.20, 0.30))
    assert reviewed.joinpath("frame_00001.txt").read_text() == "", "an empty frame is a negative"
