"""The receiving end of `media.stream`, exercised by a fake duck."""

from __future__ import annotations

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from duck_detector import capture, robot


def jpeg(width: int = 72, height: int = 128) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "grey").save(buffer, format="JPEG")
    return buffer.getvalue()


HELLO = {
    "type": "hello",
    "robot": {"name": "graphite", "serial": "cec2b3808a7238ff", "release": "0.9.4"},
    "frames": {"encoding": "jpeg", "longest": 1280, "fps": 2.0, "rotate": 0, "mount_rotate": 90},
}


class FakeLane:
    """Answers `media.stream` the way `mediad` does, and remembers what it was asked."""

    def __init__(self, dial):
        self.calls = []
        self.dial = dial
        self.loop = None

    def call(self, method, params=None):
        # Called off the event loop's thread (`asyncio.to_thread`), as the real lane is.
        self.calls.append((method, params))
        if params and params.get("url"):
            # The robot dials the url it was given, from another task.
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.dial(params["url"])))
            return {"streaming": True, "url": params["url"]}
        if params is not None and params.get("url", "x") is None:
            return {"streaming": False}
        return {"streaming": True, "connected": False, "reconnects": 3}


async def fake_duck(url: str, frames: int, hello: dict = HELLO):
    from websockets.asyncio.client import connect

    async with connect(url) as socket:
        await socket.send(json.dumps(hello))
        for _ in range(frames):
            await socket.send(jpeg())
            await asyncio.sleep(0.01)
        # Stays open until the receiver is done, like the real one: it never hangs up first.
        await asyncio.sleep(1)


def args(tmp_path, **over):
    base = dict(
        tag="kitchen",
        seconds=2,
        hz=2.0,
        longest=1280,
        quality=90,
        port=0,
        advertise="127.0.0.1",
        root=str(tmp_path / "raw"),
        note="two ducks",
    )
    base.update(over)
    return SimpleNamespace(**base)


DUCK = robot.Duck(
    peer_id="p1",
    name="graphite",
    kind="microduck",
    release="0.9.4",
    busy=False,
    active_app=None,
    age=1.0,
)


def go(a, lane, duck, out):
    async def main():
        lane.loop = asyncio.get_running_loop()
        await capture.run(a, lane, duck, out)

    asyncio.run(main())


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_session_is_captured_off_the_stream_and_the_stream_is_stopped(tmp_path):
    """The whole point: frames in, `session.json` beside them, and the robot told to stop.

    `--seconds 2` at 2 Hz is four frames; the fake sends more, and the receiver stops at four
    rather than at whatever the robot felt like sending.
    """
    lane = FakeLane(lambda url: fake_duck(url, frames=10))
    a = args(tmp_path, port=free_port())
    out = {}
    go(a, lane, DUCK, out)

    directory = out["dir"]
    assert directory.name.endswith("_kitchen_graphite")
    frames = sorted(p.name for p in directory.glob("frame_*.jpg"))
    assert frames == ["frame_00000.jpg", "frame_00001.jpg", "frame_00002.jpg", "frame_00003.jpg"]

    record = json.loads((directory / "session.json").read_text())
    assert record["robot"] == "graphite"
    assert record["serial"] == "cec2b3808a7238ff"
    assert record["frames"] == 4
    assert (record["width"], record["height"]) == (72, 128)
    # Upright already, and the record says so — the one field that would cost an afternoon.
    assert record["rotate"] == 0 and record["mount_rotate"] == 90
    assert record["hello"]["frames"]["encoding"] == "jpeg"

    methods = [(m, (p or {}).get("url", "absent")) for m, p in lane.calls]
    assert methods[0][0] == "media.stream" and methods[0][1].startswith("ws://127.0.0.1:")
    assert methods[0][1].endswith("/frames")
    assert lane.calls[0][1]["encoding"] == "jpeg" and lane.calls[0][1]["fps"] == 2.0
    # Stopped: `url: null` is the stop, and it is the last thing said.
    assert methods[-1] == ("media.stream", None)


def test_a_robot_that_never_dials_is_reported_with_its_own_view(tmp_path, monkeypatch):
    """Accepted and silent is the failure this transport has: the robot cannot reach the laptop."""
    monkeypatch.setattr(capture, "HELLO_TIMEOUT", 0.2)

    async def nobody(url):
        pass

    lane = FakeLane(nobody)
    with pytest.raises(robot.RobotError) as raised:
        go(args(tmp_path, port=free_port()), lane, DUCK, {})
    message = str(raised.value)
    assert "never dialled ws://127.0.0.1:" in message
    assert '"reconnects": 3' in message, "the robot's own counters are the diagnosis"
    # And it still asked the robot to stop.
    assert lane.calls[-1] == ("media.stream", {"url": None})


def test_the_wrong_encoding_is_refused_rather_than_written(tmp_path):
    """An H.264 unit saved as `.jpg` is a dataset of files nothing can open."""
    hello = {**HELLO, "frames": {**HELLO["frames"], "encoding": "h264"}}
    lane = FakeLane(lambda url: fake_duck(url, frames=3, hello=hello))
    with pytest.raises(robot.RobotError, match="h264"):
        go(args(tmp_path, port=free_port()), lane, DUCK, {})


def test_the_session_record_round_trips(tmp_path):
    """The session record is what makes a train/val split by session possible later."""
    session = capture.Session(
        session="20260826T120000Z_kitchen_graphite",
        tag="kitchen",
        robot="graphite",
        serial="cec2b3808a7238ff",
        frames=240,
        hz=2.0,
        seconds=120,
        width=720,
        height=1280,
        rotate=0,
        mount_rotate=90,
        started_utc="20260826T120000Z",
        release="0.9.4",
        note="two ducks, one walking",
    )
    path = tmp_path / "session.json"
    session.write(path)
    back = json.loads(path.read_text())
    assert back["rotate"] == 0, "the orientation must be recorded, or the dataset is ambiguous"
    assert back["frames"] == 240
    assert back["transport"] == "media.stream"


def test_the_lan_address_is_an_address():
    import ipaddress

    ipaddress.ip_address(capture.lan_address())
