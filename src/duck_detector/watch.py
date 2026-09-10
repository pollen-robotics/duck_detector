"""Look at a trained detector through a duck's eyes before shipping it.

    uv run watch runs/detect/duck --host 192.168.10.139     # live: boxes on the robot's stream
    uv run watch runs/detect/duck --session datasets/raw/<session>   # offline: a sheet, and P/R

**Live is the review that matters.** The val numbers come from frames the model was not trained on,
but they are frames from the same rooms, the same day, the same two ducks — and a detector that
scores well there can still lose a duck against a window or find one in a chair leg the moment the
camera looks somewhere new. Pointing the robot at things and watching the boxes is how that shows
up before the model is quantised and pushed. The stream is the one `capture` uses (`media.stream`,
JPEG, LAN), so the pixels are the pixels the deployed detector will get.

The boxes are drawn on each frame and served as MJPEG on a local port; a browser opens on it. No
GUI toolkit: this repo already opens a browser for the review step, and a page needs nothing
installed.

**Offline is for a number.** `--session` runs the model over a captured session — the held-out one
by default, read from the run's `build.json` — writes a contact sheet under `datasets/predicted/`
and, when the session has corrections, prints precision and recall against them at IoU 0.5. That is
the honest score: whole sessions the model never saw, at the confidence threshold you would deploy.

The weights are the run's `best.onnx` when it exists, else `best.pt`: the ONNX is what gets
quantised, so what is reviewed is one step closer to what ships. `--weights` overrides.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

from duck_detector import robot
from duck_detector.capture import Receiver, streaming

PREDICTED = Path("datasets/predicted")
REVIEWED = Path("datasets/reviewed")
VIEW_PORT = 8090
IMGSZ = 320
CONF = 0.25
IOU_MATCH = 0.5

Box = tuple[float, float, float, float]


# ── the model ────────────────────────────────────────────────────────────────


def weights_of(run: Path, override: Path | None) -> Path:
    if override:
        return override
    if run.suffix in (".pt", ".onnx"):
        return run
    for name in ("best.onnx", "best.pt"):
        if (run / "weights" / name).exists():
            return run / "weights" / name
    raise SystemExit(f"no weights under {run / 'weights'} — `uv run train --export` writes them")


class Detector:
    """One loaded model; `boxes` gives (x0, y0, x1, y1, score) in the frame's own pixels."""

    def __init__(self, weights: Path, *, conf: float = CONF, imgsz: int = IMGSZ):
        from duck_detector.torchsetup import missing, prepare_cuda

        try:
            from ultralytics import YOLO
        except ImportError as error:  # pragma: no cover - depends on how the venv was synced
            raise SystemExit(missing("ultralytics")) from error
        self.model = YOLO(str(weights), task="detect")
        self.conf = conf
        self.imgsz = imgsz
        # ONNX on the CPU: onnxruntime's CUDA provider hits the same broken cuDNN the README
        # describes, and a nano model at 320 px runs in a few milliseconds without it. A `.pt`
        # goes through the probe that decides whether this machine's convolutions work at all.
        self.device = "cpu" if weights.suffix == ".onnx" else prepare_cuda()

    def boxes(self, image) -> list[tuple[Box, float]]:
        result = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        return [
            (tuple(float(v) for v in xyxy), float(score))
            for xyxy, score in zip(
                result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), strict=True
            )
        ]


def draw(image, boxes: list[tuple[Box, float]], note: str = ""):
    from PIL import ImageDraw

    canvas = image.copy()
    pen = ImageDraw.Draw(canvas)
    for (x0, y0, x1, y1), score in boxes:
        pen.rectangle([x0, y0, x1, y1], outline="lime", width=3)
        pen.text((x0 + 3, y0 + 3), f"{score:.2f}", fill="lime")
    if note:
        pen.text((6, 6), note, fill="yellow")
    return canvas


# ── live ─────────────────────────────────────────────────────────────────────


class Viewer:
    """The newest annotated frame, served as MJPEG to whoever is looking."""

    PAGE = (
        b"<!doctype html><title>duck watch</title>"
        b"<body style='margin:0;background:#111;display:grid;place-items:center;height:100vh'>"
        b"<img src='/stream' style='max-height:100vh;max-width:100vw'></body>"
    )

    def __init__(self) -> None:
        self.latest: bytes | None = None
        self.changed = asyncio.Event()
        self.viewers = 0

    def publish(self, jpeg: bytes) -> None:
        self.latest = jpeg
        self.changed.set()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = await reader.readline()
        while (await reader.readline()).strip():
            pass  # the headers; nothing in them matters here
        path = request.split()[1] if len(request.split()) > 1 else b"/"
        if path != b"/stream":
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: "
                + str(len(self.PAGE)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + self.PAGE
            )
            await writer.drain()
            writer.close()
            return
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
        )
        self.viewers += 1
        try:
            while True:
                await self.changed.wait()
                self.changed.clear()
                if self.latest is None:
                    continue
                writer.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(self.latest)).encode()
                    + b"\r\n\r\n"
                    + self.latest
                    + b"\r\n"
                )
                await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            self.viewers -= 1
            writer.close()


class Live:
    """Frames in from the robot, boxes on, frames out to the viewer; the rest dropped."""

    def __init__(self, detector: Detector, viewer: Viewer):
        self.detector = detector
        self.viewer = viewer
        self.loop = asyncio.get_running_loop()
        self.pending: bytes | None = None
        self.busy = False
        self.frames = self.seen = self.detections = 0
        self.started = time.monotonic()

    def on_frame(self, jpeg: bytes) -> None:
        # Only the newest frame is worth inferring on; a queue would show the past.
        self.frames += 1
        self.pending = jpeg
        if not self.busy:
            self.busy = True
            asyncio.ensure_future(self.infer())

    async def infer(self) -> None:
        from PIL import Image

        try:
            while self.pending is not None:
                jpeg, self.pending = self.pending, None
                image = Image.open(io.BytesIO(jpeg)).convert("RGB")
                boxes = await asyncio.to_thread(self.detector.boxes, image)
                self.seen += 1
                self.detections += len(boxes)
                note = f"{len(boxes)} duck(s)   {self.seen}/{self.frames} frames"
                out = io.BytesIO()
                draw(image, boxes, note).save(out, format="JPEG", quality=80)
                self.viewer.publish(out.getvalue())
                if self.seen % 10 == 0:
                    rate = self.seen / max(time.monotonic() - self.started, 1e-6)
                    print(
                        f"   {self.seen} inferred of {self.frames} received, {rate:.1f}/s, "
                        f"{self.detections} boxes so far, {self.viewer.viewers} viewer(s)",
                        file=sys.stderr,
                        end="\r",
                    )
        finally:
            self.busy = False


async def live(args: argparse.Namespace, detector: Detector) -> None:
    host = robot.find_host(args.host)
    viewer = Viewer()
    server = await asyncio.start_server(viewer.handle, "127.0.0.1", args.view_port)
    url = f"http://127.0.0.1:{args.view_port}/"
    print(f"== boxes at {url}", file=sys.stderr)
    if not args.no_open:
        webbrowser.open(url)
    try:
        async with robot.Lane(host, port=args.signalling_port) as lane:
            duck = lane.producer
            print(
                f"== {duck.name} ({duck.release or 'release unknown'}) at {host}", file=sys.stderr
            )
            pipeline = Live(detector, viewer)
            receiver = Receiver(None, on_frame=pipeline.on_frame)
            async with streaming(
                lane,
                receiver,
                port=args.port,
                advertise=args.advertise,
                hz=args.hz,
                longest=args.longest,
                quality=args.quality,
            ):
                print("== watching; Ctrl-C to stop", file=sys.stderr)
                await asyncio.Event().wait()
    finally:
        server.close()


# ── offline ──────────────────────────────────────────────────────────────────


def held_out(run: Path) -> Path | None:
    """The session(s) the run's build held out for validation, if the run recorded them."""
    build = run / "build.json"
    if run.is_dir() and build.exists():
        val = [
            s["session"] for s in json.loads(build.read_text())["sessions"] if s["split"] == "val"
        ]
        if val:
            return Path("datasets/raw") / val[-1]
    return None


def score(session: Path, predictions: dict[str, list[tuple[Box, float]]]) -> dict[str, Any] | None:
    """Precision and recall against the corrections at IoU 0.5, or `None` without corrections."""
    from PIL import Image

    from duck_detector.agreement import iou, yolo_to_xyxy

    reviewed = REVIEWED / session.name
    if not reviewed.is_dir():
        return None
    hit = predicted = real = frames = 0
    for label in sorted(reviewed.glob("frame_*.txt")):
        frame = session / f"{label.stem}.jpg"
        if frame.name not in predictions:
            continue
        frames += 1
        width, height = Image.open(frame).size
        truth = [yolo_to_xyxy(line) for line in label.read_text().splitlines() if line.strip()]
        guesses = [
            (x0 / width, y0 / height, x1 / width, y1 / height)
            for (x0, y0, x1, y1), _ in predictions[frame.name]
        ]
        real += len(truth)
        predicted += len(guesses)
        taken: set[int] = set()
        for box in truth:
            best = max(
                ((iou(box, g), i) for i, g in enumerate(guesses) if i not in taken),
                default=(0.0, None),
            )
            if best[1] is not None and best[0] >= IOU_MATCH:
                hit += 1
                taken.add(best[1])
    return {
        "frames": frames,
        "real": real,
        "predicted": predicted,
        "hit": hit,
        "missed": real - hit,
        "false": predicted - hit,
        "precision": round(hit / predicted, 3) if predicted else 0.0,
        "recall": round(hit / real, 3) if real else 0.0,
    }


def offline(session: Path, detector: Detector, *, columns: int = 6) -> Path:
    from PIL import Image

    from duck_detector.autolabel import sheet

    frames = sorted(session.glob("frame_*.jpg"))
    if not frames:
        raise SystemExit(f"no frames in {session}")
    # Only the frames somebody looked at, when there are corrections: the sheet then lines up with
    # the numbers, and a 120-frame session is a sheet nobody scrolls.
    reviewed = REVIEWED / session.name
    if reviewed.is_dir():
        looked_at = {p.stem for p in reviewed.glob("frame_*.txt")}
        frames = [f for f in frames if f.stem in looked_at] or frames

    predictions: dict[str, list[tuple[Box, float]]] = {}
    for index, frame in enumerate(frames):
        predictions[frame.name] = detector.boxes(Image.open(frame).convert("RGB"))
        print(f"   {index + 1}/{len(frames)}", file=sys.stderr, end="\r")
    print(file=sys.stderr)

    out = PREDICTED / session.name
    out.mkdir(parents=True, exist_ok=True)
    rows = [(f, [(box, s, "duck") for box, s in predictions[f.name]]) for f in frames]
    sheet(rows, out / "sheet.png", columns=columns)
    (out / "predictions.json").write_text(
        json.dumps(
            {
                "conf": detector.conf,
                "imgsz": detector.imgsz,
                "frames": [
                    {
                        "frame": f.name,
                        "boxes": [{"box": b, "score": s} for b, s in predictions[f.name]],
                    }
                    for f in frames
                ],
            },
            indent=2,
        )
        + "\n"
    )
    total = sum(map(len, predictions.values()))
    print(f"== {len(frames)} frames, {total} boxes → {out / 'sheet.png'}")
    numbers = score(session, predictions)
    if numbers is None:
        print("   no corrections for this session, so no score — the sheet is the review")
    else:
        print(
            f"   against the corrections: {numbers['hit']} of {numbers['real']} ducks found, "
            f"{numbers['false']} false box(es)  →  precision {numbers['precision']:.0%}, "
            f"recall {numbers['recall']:.0%}  (IoU ≥ {IOU_MATCH}, conf ≥ {detector.conf})"
        )
        (out / "score.json").write_text(json.dumps(numbers, indent=2) + "\n")
    return out


# ── cli ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained detector on a duck's live stream, or on a captured session.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run", type=Path, help="a runs/detect/<name> directory, or a .pt/.onnx")
    parser.add_argument("--weights", type=Path, help="override which file under the run to load")
    parser.add_argument("--conf", type=float, default=CONF, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument(
        "--session",
        type=Path,
        nargs="?",
        const=Path("-"),
        help="offline, on this datasets/raw/<session>; alone, the run's held-out session",
    )
    parser.add_argument("--host", help="the robot's address; default: what `duckctl ip` finds")
    parser.add_argument("--signalling-port", type=int, default=robot.SIGNALLING_PORT)
    parser.add_argument("--hz", type=float, default=5.0, help="frames per second from the robot")
    parser.add_argument("--longest", type=int, default=640, help="longest side asked of the robot")
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--port", type=int, default=8765, help="where the robot dials in")
    parser.add_argument("--advertise", help="this laptop's address as the robot sees it")
    parser.add_argument("--view-port", type=int, default=VIEW_PORT, help="the page with the boxes")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    weights = weights_of(args.run, args.weights)
    print(f"== {weights}", file=sys.stderr)
    detector = Detector(weights, conf=args.conf, imgsz=args.imgsz)

    if args.session is not None:
        session = held_out(args.run) if str(args.session) == "-" else args.session
        if session is None:
            raise SystemExit("this run recorded no held-out session; name one with --session <dir>")
        if not session.is_dir():
            raise SystemExit(f"no such session: {session}")
        offline(session, detector)
        return
    try:
        asyncio.run(live(args, detector))
    except KeyboardInterrupt:
        pass
    except (robot.RobotError, robot.RpcError) as e:
        raise SystemExit(f"watch: {e}") from None


if __name__ == "__main__":
    main()
