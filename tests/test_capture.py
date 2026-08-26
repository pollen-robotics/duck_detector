"""The parts of capture that can be wrong without anybody noticing."""

from fractions import Fraction

from duck_detector import capture


def test_the_rate_crosses_as_a_fraction():
    """`framerate=2.0/1` is a negotiation failure, not a rounding.

    GStreamer caps hold fractions, and a float formatted into one does not fail loudly — the
    pipeline refuses to start and says so in a line nobody reads. This is also what buys `--hz
    0.5`: one frame every two seconds, for walking a duck slowly round a room.
    """
    for hz, expected in [(2.0, (2, 1)), (0.5, (1, 2)), (1.0, (1, 1)), (0.2, (1, 5))]:
        rate = Fraction(hz).limit_denominator(100)
        assert (rate.numerator, rate.denominator) == expected, hz


def test_the_capture_script_is_packaged_and_parses():
    """It ships inside the package because it is pushed over ssh, not installed on the robot."""
    from importlib import resources

    script = resources.files("duck_detector").joinpath("robot_capture.sh").read_text()
    assert script.startswith("#!/bin/sh")
    # The two things a silent mistake here would cost: a dataset captured sideways, and a session
    # that leaves the robot without its camera daemon.
    assert "videoflip video-direction=" in script
    assert "trap restore EXIT" in script
    # And the rate must reach the caps as a fraction.
    assert 'framerate="$HZ_NUM"/"$HZ_DEN"' in script


def test_the_session_record_round_trips(tmp_path):
    """The session record is what makes a train/val split by session possible later."""
    session = capture.Session(
        session="20260826T120000Z_kitchen_duck-c51b",
        tag="kitchen",
        host="microduck@192.168.10.124",
        robot="duck-c51b",
        serial="bb7b734a7717ac41",
        frames=240,
        hz=2.0,
        seconds=120,
        width=1280,
        height=720,
        flip="90r",
        started_utc="20260826T120000Z",
        release="0.9.3",
        note="two ducks, one walking",
    )
    path = tmp_path / "session.json"
    session.write(path)

    import json

    back = json.loads(path.read_text())
    assert back["flip"] == "90r", "the rotation must be recorded, or the dataset is ambiguous"
    assert back["frames"] == 240
    assert back["robot"] == "duck-c51b"
