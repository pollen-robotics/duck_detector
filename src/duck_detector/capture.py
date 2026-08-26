"""Pull a session of stills off a Microduck's own camera.

The dataset has to come from *this* camera: a wide lens 20 cm off the floor, an ISP with its own
colour, and a picture the daemon turns a quarter turn before anything sees it. Frames scraped from
anywhere else train a detector for a camera nobody has.

    uv run capture --host microduck@192.168.10.124 --seconds 120 --tag kitchen-afternoon

What it does, in order: check the board can capture, stop `mediad` (V4L2 is exclusive and it holds
the camera), capture, pull the frames back, start `mediad` again, and write a `session.json` beside
them. Nothing is installed on the robot — the capture script goes over ssh on stdin.

**Sessions are the unit, not frames.** Two frames half a second apart are the same picture for
training purposes, so a split that mixes them across train and val reports a score the model has
not earned. `session.json` is what makes splitting by session possible later, and the tag is what
makes a session findable when the model turns out to be bad at one kind of room.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from importlib import resources
from pathlib import Path

# Where a pulled session lands. Not in the repo — see .gitignore.
DEFAULT_ROOT = Path("datasets/raw")

# The quarter turn `mediad` applies before its tee, so what we capture matches what a model will
# be handed at inference. Keep in step with `mediad --rotate` (default 90).
DEFAULT_FLIP = "90r"


@dataclass
class Session:
    """What a session is, beyond its frames — everything needed to split, filter or redo it."""

    session: str
    tag: str
    host: str
    robot: str | None
    serial: str | None
    frames: int
    hz: float
    seconds: int
    width: int
    height: int
    flip: str
    started_utc: str
    release: str | None
    note: str

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def ssh(host: str, command: str, *, check: bool = True, quiet: bool = False) -> str:
    """Run one command on the robot and return its stdout."""
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and check:
        raise SystemExit(
            f"ssh {host}: `{command}` failed ({result.returncode})\n{result.stderr.strip()}"
        )
    if result.stderr.strip() and not quiet:
        print(result.stderr.strip(), file=sys.stderr)
    return result.stdout.strip()


def robot_identity(host: str) -> tuple[str | None, str | None, str | None]:
    """The robot's name, serial and release, for the session record.

    All three are `None` on a board where `robotctl` cannot answer — a capture from a robot whose
    daemons are down is still a good capture, and refusing it over a missing label would be silly.
    """
    raw = ssh(host, "robotctl system info --json 2>/dev/null || true", check=False, quiet=True)
    name = serial = None
    if raw:
        try:
            info = json.loads(raw)
            name, serial = info.get("name"), info.get("serial")
        except json.JSONDecodeError:
            pass
    release = ssh(
        host,
        "readlink /opt/robot/daemon/current 2>/dev/null | sed 's|releases/||' || true",
        check=False,
        quiet=True,
    )
    return name, serial, release or None


def capture(args: argparse.Namespace) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # Asked of the robot, so a session record says which duck was looking — skipped on a dry run,
    # which must reach nothing.
    name, serial, release = (None, None, None) if args.dry_run else robot_identity(args.host)
    session = f"{stamp}_{args.tag}_{name or 'unknown'}"
    remote_dir = f"/var/tmp/duck-capture/{session}"
    local_dir = Path(args.root) / session

    script = resources.files("duck_detector").joinpath("robot_capture.sh").read_text()
    # `framerate=2.0/1` is not a caps value GStreamer will negotiate, so the rate travels as a
    # fraction — which also buys `--hz 0.5`, one frame every two seconds, for a long slow walk.
    rate = Fraction(args.hz).limit_denominator(100)
    env = {
        "DIR": remote_dir,
        "SECONDS_": str(args.seconds),
        "HZ_NUM": str(rate.numerator),
        "HZ_DEN": str(rate.denominator),
        "WIDTH": str(args.width),
        "HEIGHT": str(args.height),
        "FLIP": args.flip,
        "DEVICE": args.device,
        "QUALITY": str(args.quality),
    }
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())

    print(f"== {session}: {args.seconds}s at {args.hz} Hz from {args.host}", file=sys.stderr)
    if args.dry_run:
        print(f"would run on {args.host}:\n  env {exports} sh -s", file=sys.stderr)
        return local_dir

    # No `-tt`: a tty would echo this script back into stdout and CRLF every line. An interrupt
    # still reaches the robot — ssh dying hangs up the remote shell, and its EXIT trap is what puts
    # mediad back.
    result = subprocess.run(
        ["ssh", args.host, f"env {exports} sh -s"],
        input=script,
        capture_output=True,
        text=True,
        check=False,
    )
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise SystemExit(f"capture failed on {args.host} ({result.returncode})")
    # The count is the last thing the script prints, but stdout is not ours alone — read the last
    # line that is a number rather than trusting the last line.
    counts = [line.strip() for line in result.stdout.splitlines() if line.strip().isdigit()]
    if not counts:
        raise SystemExit(f"the robot reported no frame count:\n{result.stdout.strip()}")
    frames = int(counts[-1])

    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", "--remove-source-files", f"{args.host}:{remote_dir}/", f"{local_dir}/"],
        check=True,
    )
    ssh(args.host, f"rmdir {shlex.quote(remote_dir)} 2>/dev/null || true", check=False, quiet=True)

    Session(
        session=session,
        tag=args.tag,
        host=args.host,
        robot=name,
        serial=serial,
        frames=frames,
        hz=args.hz,
        seconds=args.seconds,
        width=args.width,
        height=args.height,
        flip=args.flip,
        started_utc=stamp,
        release=release,
        note=args.note,
    ).write(local_dir / "session.json")

    print(f"== {frames} frames in {local_dir}", file=sys.stderr)
    return local_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a labelling session from a Microduck's camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", required=True, help="user@address of the robot doing the looking")
    parser.add_argument(
        "--tag",
        required=True,
        help="what makes this session different: the room, the light, what is in front of it",
    )
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--hz", type=float, default=2.0, help="frames kept per second")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--flip",
        default=DEFAULT_FLIP,
        choices=["90r", "180", "90l", "identity"],
        help="must match `mediad --rotate` or the dataset is sideways",
    )
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--note", default="", help="anything worth remembering about this session")
    parser.add_argument("--dry-run", action="store_true", help="print what would run, do nothing")
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
