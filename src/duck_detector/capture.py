"""Pull a session of stills off a Microduck's own camera, without touching the robot.

The dataset has to come from *this* camera: a wide lens 20 cm off the floor, an ISP with its own
colour, and a picture the daemon turns a quarter turn before anything sees it. Frames scraped from
anywhere else train a detector for a camera nobody has.

    uv run capture --tag kitchen-afternoon --seconds 120

**Nothing is stopped and nothing is installed.** `mediad` holds the camera and keeps holding it:
this opens the same session `duckctl open`'s page does — WebRTC on the LAN, a `control`
datachannel — and asks for `media.stream`. The robot then dials a WebSocket this tool opens on the
laptop and pushes JPEG frames down it, upright, at the rate asked for, straight off the tee that
already feeds the console's video and the robot's own duck detector. So the dataset is captured
through exactly what a model will be handed at inference, by construction.

    laptop ──ws://robot:8443, datachannel: media.stream {url: "ws://laptop:8765/frames"}──► robot
    robot  ═══════════════ JPEG frames, LAN, direct ═══════════════════════════════════► laptop

Everything stays on the LAN (`robot.py`). `--host` is the robot's address, or `duckctl ip` finds it
over Bluetooth when it is left out; `--advertise` is the laptop's address as the robot should dial
it, when the guess is wrong.

**Sessions are the unit, not frames.** Two frames half a second apart are the same picture for
training purposes, so a split that mixes them across train and val reports a score the model has
not earned. `session.json` is what makes splitting by session possible later, and the tag is what
makes a session findable when the model turns out to be bad at one kind of room.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from duck_detector import robot

# Where a pulled session lands. Not in the repo — see .gitignore — but on the Hub, see `hub.py`.
DEFAULT_ROOT = Path("datasets/raw")

# The upright frame is 720 wide and 1280 tall; `longest` is a downscale-only cap, so this asks for
# every pixel the camera has. The console streams at 640 because a browser does not want more.
DEFAULT_LONGEST = 1280
DEFAULT_PORT = 8765

# How long the robot gets to dial us after it accepted `media.stream`. It redials on a backoff
# starting at 2 s, so this is several attempts — and a robot that has not arrived by then is on a
# network that cannot reach this laptop, which is what the message says.
HELLO_TIMEOUT = 25.0


@dataclass
class Session:
    """What a session is, beyond its frames — everything needed to split, filter or redo it."""

    session: str
    tag: str
    robot: str | None
    serial: str | None
    frames: int
    hz: float
    seconds: int
    width: int
    height: int
    # Frames arrive upright: `mediad` turns them by the mount angle before its tee, and the hello
    # says `rotate: 0` about what it sends. Recorded so a session from a robot mounted otherwise is
    # a different domain that can be told apart, rather than a mystery.
    rotate: int
    mount_rotate: int | None
    started_utc: str
    release: str | None
    note: str
    transport: str = "media.stream"
    longest: int = DEFAULT_LONGEST
    quality: int = 90
    hello: dict[str, Any] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def lan_address(probe: str = "1.1.1.1") -> str:
    """The address this machine is reachable at from the network it talks to the world on.

    No packet is sent: connecting a UDP socket only picks the interface the kernel would route
    through, and reading its name back is the one portable way to ask "which of my addresses is
    the LAN one". Wrong on a machine with two LANs, which is what `--advertise` is for.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe_socket:
        probe_socket.connect((probe, 80))
        return probe_socket.getsockname()[0]


class Receiver:
    """The far end of `media.stream`: one hello, then a JPEG per message, written as they land."""

    def __init__(self, directory: Path, *, limit: int | None = None):
        self.directory = directory
        self.limit = limit
        self.hello: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.done = asyncio.Event()
        self.frames = 0
        self.bytes = 0
        self.connections = 0
        self.first_at: float | None = None
        self.last_at: float | None = None

    async def handle(self, connection) -> None:
        """One robot's connection. A redial after a hiccup is a second; the numbering goes on."""
        self.connections += 1
        async for message in connection:
            if isinstance(message, str):
                # **What is coming, before any of it arrives**: the robot's name, the encoding, the
                # size. Kept whole in `session.json` — it is the provenance of every frame after it.
                try:
                    hello = json.loads(message)
                except ValueError:
                    continue
                encoding = (hello.get("frames") or {}).get("encoding")
                if encoding not in (None, "jpeg"):
                    if not self.hello.done():
                        self.hello.set_exception(
                            robot.RobotError(
                                f"the robot is sending {encoding}, and this tool asked for jpeg"
                            )
                        )
                    return
                if not self.hello.done():
                    self.hello.set_result(hello)
                continue
            now = time.monotonic()
            self.first_at = self.first_at or now
            self.last_at = now
            (self.directory / f"frame_{self.frames:05d}.jpg").write_bytes(message)
            self.frames += 1
            self.bytes += len(message)
            if self.frames % 10 == 0:
                rate = (self.frames - 1) / max(now - self.first_at, 1e-6)
                print(f"   {self.frames} frames, {rate:.1f}/s", file=sys.stderr, end="\r")
            if self.limit is not None and self.frames >= self.limit:
                self.done.set()
                return


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.width, image.height


async def run(args: argparse.Namespace, lane: robot.Lane, out: dict) -> Path:
    from websockets.asyncio.server import serve

    duck = lane.producer
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session = f"{stamp}_{args.tag}_{duck.name}"
    local_dir = Path(args.root) / session
    local_dir.mkdir(parents=True, exist_ok=True)
    # Handed back through `out` rather than returned: a Ctrl-C cancels this task, and `asyncio.run`
    # then re-raises `KeyboardInterrupt` after it has finished — the return value never arrives.
    out["dir"] = local_dir

    limit = round(args.hz * args.seconds) if args.seconds else None
    receiver = Receiver(local_dir, limit=limit)
    advertise = args.advertise or lan_address()
    request = {
        "url": f"ws://{advertise}:{args.port}/frames",
        "encoding": "jpeg",
        "fps": args.hz,
        "longest": args.longest,
        "quality": args.quality,
    }
    print(f"== {session}: {args.seconds or '∞'}s at {args.hz} Hz from {duck.name}", file=sys.stderr)

    async with serve(receiver.handle, "0.0.0.0", args.port):
        answer = await lane.call("media.stream", request)
        print(f"== media.stream accepted: {json.dumps(answer)}", file=sys.stderr)
        try:
            try:
                hello = await asyncio.wait_for(asyncio.shield(receiver.hello), HELLO_TIMEOUT)
            except TimeoutError:
                status = await lane.call("media.stream")
                raise robot.RobotError(
                    f"the robot accepted the stream and never dialled {request['url']} "
                    f"(its view: {json.dumps(status)}). It has to reach this laptop: same "
                    "network, no firewall on the port, and --advertise <ip> if the address "
                    "above is not the one it should use."
                ) from None
            frames = hello.get("frames") or {}
            print(
                f"== hello from {(hello.get('robot') or {}).get('name')}: "
                f"{frames.get('encoding')}, "
                f"longest {frames.get('longest')}, {frames.get('fps')} fps",
                file=sys.stderr,
            )
            # Stopped by the frame count when there is one, else by Ctrl-C. `wait_for` on the
            # event rather than a sleep, so a slow robot still delivers `--seconds` worth.
            if limit is None:
                await receiver.done.wait()
            else:
                await asyncio.wait_for(receiver.done.wait(), timeout=args.seconds * 2 + 30)
        except (asyncio.CancelledError, KeyboardInterrupt, TimeoutError):
            print("\n== stopping", file=sys.stderr)
        finally:
            # **The stream is stopped whatever happened above**, including Ctrl-C: a robot left
            # streaming JPEGs at a laptop that has gone is a robot spending a core on nobody, and
            # redialling every 30 s until somebody notices.
            try:
                await lane.call("media.stream", {"url": None})
            except (robot.RobotError, robot.RpcError) as e:
                print(f"== could not stop the stream: {e}", file=sys.stderr)

    print(file=sys.stderr)
    if receiver.frames == 0:
        raise robot.RobotError(f"no frames arrived in {local_dir}")
    width, height = image_size(local_dir / "frame_00000.jpg")
    info = receiver.hello.result().get("robot") or {}
    frames_info = receiver.hello.result().get("frames") or {}
    Session(
        session=session,
        tag=args.tag,
        robot=info.get("name") or duck.name,
        serial=info.get("serial") or duck.serial,
        frames=receiver.frames,
        hz=args.hz,
        seconds=args.seconds,
        width=width,
        height=height,
        rotate=int(frames_info.get("rotate") or 0),
        mount_rotate=frames_info.get("mount_rotate"),
        started_utc=stamp,
        release=info.get("release") or duck.release,
        note=args.note,
        longest=args.longest,
        quality=args.quality,
        hello=receiver.hello.result(),
    ).write(local_dir / "session.json")

    span = (receiver.last_at or 0) - (receiver.first_at or 0)
    rate = (receiver.frames - 1) / span if receiver.frames > 1 and span > 0 else 0.0
    print(
        f"== {receiver.frames} frames ({receiver.bytes / 1e6:.1f} MB, {rate:.1f}/s, "
        f"{width}x{height}) in {local_dir}",
        file=sys.stderr,
    )
    return local_dir


async def session(args: argparse.Namespace, out: dict) -> None:
    host = robot.find_host(args.host)
    async with robot.Lane(host, port=args.signalling_port) as lane:
        duck = lane.producer
        print(f"== {duck.name} ({duck.release or 'release unknown'}) at {host}", file=sys.stderr)
        if args.dry_run:
            print(
                f"would ask {duck.name} to stream jpeg at {args.hz} fps, longest {args.longest}, "
                f"to ws://{args.advertise or lan_address()}:{args.port}/frames",
                file=sys.stderr,
            )
            return
        await run(args, lane, out)


def capture(args: argparse.Namespace) -> Path | None:
    out: dict[str, Path] = {}
    try:
        asyncio.run(session(args, out))
    except KeyboardInterrupt:
        pass
    return out.get("dir")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture a labelling session from a Microduck's camera, over its own stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="what makes this session different: the room, the light, what is in front of it",
    )
    parser.add_argument("--host", help="the robot's address; default: what `duckctl ip` finds")
    parser.add_argument(
        "--signalling-port", type=int, default=robot.SIGNALLING_PORT, help="mediad's `--port`"
    )
    parser.add_argument("--seconds", type=int, default=60, help="0 to run until Ctrl-C")
    parser.add_argument("--hz", type=float, default=2.0, help="frames per second (0.2 to 15)")
    parser.add_argument("--longest", type=int, default=DEFAULT_LONGEST, help="longest side, px")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality on the robot")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="where the robot dials in")
    parser.add_argument("--advertise", help="this laptop's address as the robot sees it")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--note", default="", help="anything worth remembering about this session")
    parser.add_argument("--push", action="store_true", help="upload the session to the Hub after")
    parser.add_argument("--dry-run", action="store_true", help="open the session, stream nothing")
    parser.add_argument("-v", "--verbose", action="store_true", help="log the signalling and calls")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    try:
        local_dir = capture(args)
    except (robot.RobotError, robot.RpcError) as e:
        raise SystemExit(f"capture: {e}") from None
    if args.push and local_dir is not None:
        from duck_detector import hub

        hub.push_sessions([local_dir.name])


if __name__ == "__main__":
    main()
