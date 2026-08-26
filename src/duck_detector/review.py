"""One command: raw session in, corrected labels out.

    uv run review datasets/raw/<session>

Everything between those two things is machinery, so it is done for you: the frames are triaged,
pre-labelled if they have not been, turned into Label Studio tasks with the boxes already drawn,
and Label Studio is started, logged into, given a project and the tasks, and opened in a browser.
Correct them, press Ctrl-C in the terminal, and the corrections come back out through the API as
YOLO labels in `datasets/reviewed/<session>/`.

Nothing is exported by hand and no token is copied out of a settings page. Two things make that
possible:

* **Label Studio runs out of `.label-studio/` in the repo**, with a user this tool creates and a
  password it writes down there. A self-contained instance cannot collide with whatever else is
  installed, and `rm -rf .label-studio` is a clean slate.
* **The API is reached with a session cookie**, because Label Studio 1.23 disabled legacy tokens
  for an organisation ("legacy token authentication has been disabled") and its `/api/token/`
  endpoint wants an authenticated session already. So: log in at the form the browser uses, keep
  the cookie, and echo the CSRF token on every write.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import secrets
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

DATASETS = Path("datasets")
REVIEW = DATASETS / "review"
REVIEWED = DATASETS / "reviewed"
INSTANCE = Path(".label-studio")

# One class. `duck` rather than `robot`, because a Reachy Mini in the background is a negative
# rather than a smaller duck.
CLASS = "duck"

# `hotkey="1"` because with one class there is no reason to reach for the mouse to say which label
# you mean, and the shortcut only exists if the config asks for it.
LABEL_CONFIG = f"""<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <RectangleLabels name="label" toName="image">
    <Label value="{CLASS}" background="#00c853" hotkey="1"/>
  </RectangleLabels>
</View>
"""


# ── the frames ────────────────────────────────────────────────────────────────


def local_files_url(path: Path) -> str:
    """The URL Label Studio serves a local file at, relative to the document root.

    Resolved on both sides, because the path arrives either way: `datasets/raw/x` from a shell and
    `/home/…/datasets/raw/x` from a tab-completion.
    """
    root = (Path.cwd() / DATASETS).resolve()
    return f"/data/local-files/?d={path.resolve().relative_to(root)}"


def ensure_prepared(session: Path, relabel: bool) -> Path:
    """Triage and pre-label the session if that has not happened, then write the tasks.

    Skipped where the outputs already exist, so re-running this on a session you are halfway
    through costs a second rather than a GPU pass.
    """
    from duck_detector import autolabel, triage

    triage_json = session / "triage.json"
    if not triage_json.exists():
        print("== triage: ranking the frames (most of a session is blurred floor)")
        scored = triage.triage(session, want=40, negatives=10)
        triage_json.write_text(
            json.dumps(
                {"want": 40, "negatives": 10, "frames": [s.__dict__ for s in scored]},
                indent=2,
            )
            + "\n"
        )

    labels = DATASETS / "labelled" / session.name
    if relabel or not (labels / "labels.json").exists():
        print("== pre-label: drawing the first boxes (the slow step)")
        autolabel.run(session, labels, threshold=0.30, use_all=False, limit=0, make_sheet=True)

    return prepare(session, labels)


def prepare(session: Path, labels: Path) -> Path:
    """Label Studio tasks for one session, with the pre-labeller's boxes as predictions."""
    from PIL import Image

    predictions: dict[str, list] = {}
    model = "none"
    labels_json = labels / "labels.json"
    if labels_json.exists():
        loaded = json.loads(labels_json.read_text())
        predictions = {f["frame"]: f["boxes"] for f in loaded["frames"]}
        model = loaded.get("model", "unknown")

    triage_json = session / "triage.json"
    if triage_json.exists():
        wanted = {f["frame"] for f in json.loads(triage_json.read_text())["frames"] if f["select"]}
    else:
        wanted = {p.name for p in session.glob("frame_*.jpg")}

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
    path = out / "tasks.json"
    path.write_text(json.dumps(tasks, indent=2) + "\n")
    drawn = sum(len(t["predictions"][0]["result"]) for t in tasks if t["predictions"])
    print(f"== {len(tasks)} frames to review, {drawn} boxes already drawn")
    return path


# ── the instance ──────────────────────────────────────────────────────────────


def credentials() -> dict:
    """This repo's Label Studio account: invented once, written down, gitignored."""
    path = INSTANCE / "credentials.json"
    if path.exists():
        return json.loads(path.read_text())
    INSTANCE.mkdir(parents=True, exist_ok=True)
    creds = {"email": "duck@localhost", "password": secrets.token_urlsafe(12)}
    path.write_text(json.dumps(creds, indent=2) + "\n")
    return creds


def free_port(start: int) -> int:
    """`start`, or the next port nothing is listening on.

    A Label Studio somebody already has running should not turn this into a confusing failure.
    """
    for port in range(start, start + 20):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit(f"no free port between {start} and {start + 20}")


def start_label_studio(port: int, creds: dict) -> subprocess.Popen:
    root = (Path.cwd() / DATASETS).resolve()
    env = os.environ | {
        # Without these two, Label Studio refuses to read a frame off the disk and every task shows
        # an empty image area.
        "LOCAL_FILES_SERVING_ENABLED": "true",
        "LOCAL_FILES_DOCUMENT_ROOT": str(root),
        # Its own database, inside the repo: this cannot disturb another Label Studio, and deleting
        # the directory is how you start over.
        "LABEL_STUDIO_BASE_DATA_DIR": str((Path.cwd() / INSTANCE).resolve()),
    }
    print(f"== label studio on :{port} (the first run migrates a database, ~30 s)")
    return subprocess.Popen(
        [
            "label-studio",
            "start",
            "--port",
            str(port),
            "--no-browser",
            "--username",
            creds["email"],
            "--password",
            creds["password"],
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # **Its own process group.** Ctrl-C in a terminal goes to every process in the foreground
        # group, so without this Label Studio starts shutting down at the same moment we try to
        # export the corrections out of it — which is what "Connection reset by peer" on the way
        # out was. Now the interrupt reaches this process alone, and the child is stopped
        # deliberately, after the export.
        start_new_session=True,
    )


class Client:
    """The little of Label Studio's API this needs, over a logged-in session."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))

    def _cookie(self, name: str) -> str:
        return next((c.value or "" for c in self.jar if c.name == name), "")

    def _call(self, method: str, path: str, body: bytes | None = None, form: bool = False):
        request = urllib.request.Request(f"{self.base}{path}", data=body, method=method)
        # Django's CSRF check wants both of these on a write, and the browser sends them.
        request.add_header("Referer", f"{self.base}/")
        if body is not None:
            request.add_header(
                "Content-Type",
                "application/x-www-form-urlencoded" if form else "application/json",
            )
            request.add_header("X-CSRFToken", self._cookie("csrftoken"))
        with self.opener.open(request, timeout=300) as answer:
            raw = answer.read()
        return json.loads(raw) if raw[:1] in (b"{", b"[") else raw

    def wait(self, timeout: float = 300) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self.opener.open(f"{self.base}/user/login/", timeout=5)
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(1.0)
        raise SystemExit("label studio did not come up")

    def login(self, creds: dict) -> None:
        # Fetching the form first is what sets the CSRF cookie the POST has to echo.
        self.opener.open(f"{self.base}/user/login/", timeout=30)
        body = urllib.parse.urlencode(
            {
                "email": creds["email"],
                "password": creds["password"],
                "csrfmiddlewaretoken": self._cookie("csrftoken"),
            }
        ).encode()
        self._call("POST", "/user/login/", body, form=True)
        # Proved rather than assumed: a failed login also returns a page, and the next call would
        # fail somewhere less obvious.
        projects = self._call("GET", "/api/projects")
        if not isinstance(projects, dict) or "results" not in projects:
            raise SystemExit(f"logged in, but the api answered: {str(projects)[:200]}")

    def project(self, title: str) -> tuple[int, int]:
        """The project for this session, created if absent. Returns (id, tasks it already has)."""
        listing = self._call("GET", f"/api/projects?title={urllib.parse.quote(title)}")
        for found in listing.get("results", []):
            if found["title"] == title:
                return found["id"], found.get("task_number") or 0
        created = self._call(
            "POST",
            "/api/projects",
            json.dumps({"title": title, "label_config": LABEL_CONFIG}).encode(),
        )
        if "id" not in created:
            raise SystemExit(f"could not create the project: {str(created)[:300]}")
        return created["id"], 0

    def ensure_local_storage(self, project: int, path: Path) -> None:
        """Register the frames directory as a local storage, or the images do not load.

        `LOCAL_FILES_SERVING_ENABLED` and a document root are necessary and **not sufficient**: in
        1.23 `/data/local-files/` answers 404 until the file also belongs to a storage attached to
        the project. The path has to be a *subdirectory* of the document root — pointing it at the
        root itself is refused, in the one error message that explains any of this.

        Registered, never synced: a sync would invent one task per file and lose the boxes that
        were imported with ours.
        """
        listing = self._call("GET", f"/api/storages/localfiles?project={project}")
        storages = listing if isinstance(listing, list) else listing.get("results", [])
        wanted = str(path.resolve())
        if any(storage.get("path") == wanted for storage in storages):
            return
        self._call(
            "POST",
            "/api/storages/localfiles",
            json.dumps(
                {"project": project, "path": wanted, "title": "frames", "use_blob_urls": False}
            ).encode(),
        )

    def import_tasks(self, project: int, tasks: Path) -> dict:
        return self._call("POST", f"/api/projects/{project}/import", tasks.read_bytes())

    def export(self, project: int) -> list:
        return self._call("GET", f"/api/projects/{project}/export?exportType=JSON")


# ── the corrections, coming back ──────────────────────────────────────────────


def write_labels(tasks: list, session_name: str | None = None) -> tuple[int, int]:
    """Label Studio annotations to YOLO labels. Returns (frames written, boxes).

    A task nobody opened has no annotation and is skipped rather than written as an empty label:
    "there is nothing here" and "nobody looked" are different, and only the first is training data.
    """
    written = boxes = 0
    for task in tasks:
        meta = task.get("meta") or {}
        session = meta.get("session") or session_name
        frame = meta.get("frame")
        if not frame:
            # Fall back to the image path, for a task list that lost its meta on the way through.
            image = task.get("data", {}).get("image", "")
            parts = Path(urllib.parse.unquote(image.split("?d=")[-1])).parts
            if len(parts) < 2:
                continue
            session, frame = session or parts[-2], parts[-1]
        if not session:
            continue

        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue

        lines = []
        for result in annotations[-1].get("result", []):
            if result.get("type") != "rectanglelabels":
                continue
            value = result["value"]
            # Percentages back to normalised centre-form, which is what YOLO reads.
            cx = (value["x"] + value["width"] / 2) / 100
            cy = (value["y"] + value["height"] / 2) / 100
            width, height = value["width"] / 100, value["height"] / 100
            lines.append(f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")

        out = REVIEWED / session
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{Path(frame).stem}.txt").write_text("".join(line + "\n" for line in lines))
        written += 1
        boxes += len(lines)
    return written, boxes


def import_export(export: Path) -> None:
    """A Label Studio export read off disk — the escape hatch for when the API round trip did not
    happen (a crash, or a project somebody set up by hand)."""
    written, boxes = write_labels(json.loads(export.read_text()))
    print(f"{written} frames, {boxes} boxes → {REVIEWED}")


# ── the one command ───────────────────────────────────────────────────────────


def _interrupt(*_: object) -> None:
    """Turn a `SIGTERM` into the same exception Ctrl-C raises.

    Python's default `SIGTERM` handling exits without unwinding, so the `finally` below would not
    run: the corrections would stay in the project and Label Studio would be left running with
    nobody's hand on it. `SIGINT` already raises this; this makes the two behave alike.
    """
    raise KeyboardInterrupt


def run(session: Path, port: int, relabel: bool, open_browser: bool) -> None:
    signal.signal(signal.SIGTERM, _interrupt)
    tasks = ensure_prepared(session, relabel)
    creds = credentials()
    port = free_port(port)
    server = start_label_studio(port, creds)
    client = Client(f"http://localhost:{port}")
    project = None

    try:
        client.wait()
        client.login(creds)
        project, existing = client.project(session.name)
        client.ensure_local_storage(project, DATASETS / "raw")
        if existing:
            print(f"== project {project} already holds {existing} tasks; not importing again")
        else:
            result = client.import_tasks(project, tasks)
            print(
                f"== imported {result.get('task_count')} frames, "
                f"{result.get('prediction_count')} with a box already drawn"
            )

        # **The labelling stream, not the table.** In the data manager a task opens in a modal
        # where submitting does not move on and half the shortcuts are unbound — which is
        # infuriating for exactly the fifty frames this is for. `labeling=1` asks for the stream;
        # if a version ever ignores it you land on the table, and "Label All Tasks" is the button
        # that gets there.
        url = f"http://localhost:{port}/projects/{project}/data?labeling=1"
        print(
            f"\n  {url}\n"
            f"  sign in once as {creds['email']} / {creds['password']}\n\n"
            "  the keys that matter:\n"
            "    1              select the duck label (then drag a box)\n"
            "    Ctrl+Enter     submit and go to the next frame\n"
            "    Delete         remove the selected box\n"
            "    Ctrl+Z         undo\n"
            "  a frame with nothing in it: submit it empty — that is a negative, and useful.\n"
            "  if you land on a table of tasks instead, click 'Label All Tasks' once.\n\n"
            "  then Ctrl-C here, and the corrections come back as labels.\n"
        )
        if open_browser:
            webbrowser.open(url)

        while server.poll() is None:
            time.sleep(1.0)
        print("label studio exited on its own")
    except KeyboardInterrupt:
        print("\n== pulling the corrections back")
    finally:
        pulled = None
        if project is not None:
            try:
                pulled = client.export(project)
            except Exception as error:  # pragma: no cover - depends how the server went away
                print(
                    f"could not export through the api ({error}). The annotations are still in the "
                    "project; `uv run review --import <export>.json` reads a manual export."
                )
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        if pulled is not None:
            written, boxes = write_labels(pulled, session.name)
            print(f"== {written} frames reviewed, {boxes} boxes → {REVIEWED / session.name}")
            if written:
                print("   next: uv run dataset build   (with two sessions or more)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review a session's pre-labels in Label Studio; get YOLO labels back.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("session", type=Path, nargs="?", help="a datasets/raw/<session> directory")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--relabel", action="store_true", help="run the pre-labeller again")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    parser.add_argument(
        "--import",
        dest="import_file",
        type=Path,
        default=None,
        help="read a Label Studio JSON export from disk instead (an escape hatch)",
    )
    args = parser.parse_args()

    if args.import_file:
        import_export(args.import_file)
        return
    if not args.session:
        parser.error("give a session directory, or --import <export>.json")
    if not args.session.is_dir():
        raise SystemExit(f"no such session: {args.session}")
    run(args.session, args.port, args.relabel, not args.no_open)


if __name__ == "__main__":
    main()
