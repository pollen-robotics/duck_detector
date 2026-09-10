"""Talking to a duck the way the vision-demo Space does: through the Hugging Face rendezvous.

Every duck that is online holds an outbound stream to `pollen-robotics/reachy-mini-central`, the
Space the Reachy Mini fleet registers with (`mediad::relay`). That is what `duckctl open` and the
console page lean on from off the LAN, and it is what this module leans on for everything: which
ducks this account has, and a control lane to one of them.

**JSON-RPC over the rendezvous, with no WebRTC in the path.** The service forwards every key of a
`peer` envelope except `type` and `sessionId` to the session partner without reading it, so an
envelope carrying `rpc` is a control call and `mediad`'s control lane answers it out of the same
routing table the console's datachannel uses:

    POST /send  {"type": "peer", "sessionId": S, "rpc": {"jsonrpc": "2.0", "id": 1, …}}
    SSE         {"type": "peer", "sessionId": S, "rpc": {"jsonrpc": "2.0", "id": 1, "result": …}}

This is a port of `spaces/vision-demo/{rendezvous,wire,control}.py` in the robot repository, cut
down to what a command-line tool needs and moved onto `httpx`, which `huggingface_hub` already
brings in. The protocol notes that matter are kept as comments where they bite.

What it costs, said once: **one consumer at a time.** The rendezvous's rule, and the robot's own
console counts — a duck somebody is watching in a browser refuses a second session. And it is not
a lane for pixels: the frames go point to point (`capture.py`), and only the instruction to send
them crosses here.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import queue
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# `mediad::relay::DEFAULT_RENDEZVOUS`. The same override the mini's own clients read.
DEFAULT_RENDEZVOUS = os.environ.get(
    "REACHY_CENTRAL_URL", "https://pollen-robotics-reachy-mini-central.hf.space"
).rstrip("/")

# `meta.kind` on the wire: an account with a duck and a mini on it lists both.
DUCK = "microduck"

CONNECT_TIMEOUT = 20.0
# The SSE stream sends a comment ping every 30 s; appreciably longer than that is a dead stream.
READ_TIMEOUT = 90.0


class RobotError(Exception):
    """Something the person at the keyboard can act on."""


def token() -> str:
    """The Hugging Face token: `HF_TOKEN`, else what `hf auth login` stored."""
    from huggingface_hub import get_token

    found = (os.environ.get("HF_TOKEN") or get_token() or "").strip()
    if not found:
        raise RobotError("no Hugging Face token: run `hf auth login`, or set HF_TOKEN")
    return found


@dataclass
class Duck:
    """One robot as the rendezvous lists it."""

    peer_id: str
    name: str
    kind: str | None
    release: str
    busy: bool
    active_app: str | None
    age: float | None

    @classmethod
    def parse(cls, entry: dict[str, Any]) -> Duck:
        meta = entry.get("meta") or {}
        return cls(
            peer_id=entry.get("peerId") or entry.get("id") or "",
            name=meta.get("name") or entry.get("robotName") or "a duck with no name",
            kind=meta.get("kind"),
            release=meta.get("release") or "release unknown",
            busy=bool(entry.get("busy")),
            # The rendezvous reports the *consumer's* label here — `duckctl open`'s page, or
            # another capture.
            active_app=entry.get("activeApp"),
            age=entry.get("last_seen_age_seconds"),
        )

    def label(self) -> str:
        bits = [self.name, self.release]
        if self.busy:
            bits.append(f"busy with {self.active_app or 'something'}")
        if self.age is not None and self.age > 60:
            bits.append(f"last heard from {self.age / 60:.0f} min ago")
        return " — ".join(bits)


def ducks(hf_token: str, base: str = DEFAULT_RENDEZVOUS) -> list[Duck]:
    """This account's ducks, online or recently so.

    `GET /api/robot-status` is one `whoami-v2` call and no session: a listing that opened `/events`
    would supersede a session the same token holds (§3.7 of the design), which is exactly what a
    tool that is about to open one must not do.
    """
    try:
        answer = httpx.get(
            f"{base}/api/robot-status",
            headers={"Authorization": f"Bearer {hf_token}"},
            timeout=CONNECT_TIMEOUT,
        )
    except httpx.HTTPError as e:
        raise RobotError(f"the rendezvous could not be reached: {e}") from None
    if answer.status_code == 401:
        raise RobotError("the rendezvous refused this token — `hf auth login` again")
    if answer.status_code == 429:
        raise RobotError("the rendezvous is rate-limiting this token; wait a minute")
    if answer.status_code != 200:
        raise RobotError(f"the rendezvous answered HTTP {answer.status_code}: {answer.text[:200]}")
    listed = answer.json().get("robots") or []
    return [duck for duck in map(Duck.parse, listed) if duck.peer_id and duck.kind == DUCK]


def choose(found: list[Duck], wanted: str | None) -> Duck:
    """The duck named, or the only one there is."""
    if wanted:
        for duck in found:
            if wanted in (duck.name, duck.peer_id):
                return duck
        names = ", ".join(d.name for d in found) or "none online"
        raise RobotError(f"no duck called {wanted!r} — listed: {names}")
    if not found:
        raise RobotError(
            "no duck is online for this account. A duck registers with the rendezvous as soon as "
            "it has a network; `duckctl ip` says whether it has one."
        )
    if len(found) > 1:
        names = ", ".join(d.name for d in found)
        raise RobotError(f"{len(found)} ducks online ({names}); say which with --robot")
    return found[0]


class RpcError(Exception):
    """A refusal from the robot, distinct from a transport failure on purpose."""

    def __init__(self, method: str, error: dict[str, Any]):
        self.method = method
        self.code = error.get("code")
        self.message = error.get("message") or "refused, with nothing said about why"
        super().__init__(f"{method}: {self.message}")


class Lane:
    """One control-only session with one duck: requests out, answers back, over the rendezvous.

    Blocking, because a capture has nothing else to do while it waits for an answer. A thread
    reads the event stream and settles the futures the calls are waiting on.
    """

    def __init__(
        self,
        hf_token: str,
        peer_id: str,
        *,
        label: str = "duck-detector/capture",
        base: str = DEFAULT_RENDEZVOUS,
        timeout: float = 30.0,
    ):
        self._token = hf_token
        self._peer_id = peer_id
        self._label = label
        self._base = base.rstrip("/")
        self.timeout = timeout

        # **Two clients, because the stream holds its connection for the life of the session**
        # while every call posts from another thread. One shared pool across those two is a stream
        # that loses messages.
        self._streaming = httpx.Client(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT))
        self._posting = httpx.Client(timeout=CONNECT_TIMEOUT)
        self._response: httpx.Response | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[str, Future]] = {}
        self._lock = threading.Lock()
        self._welcome: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self.session_id: str | None = None
        self.error: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def open(self) -> Lane:
        """Open the stream, register, and ask for a session. Raises `RobotError` when refused."""
        headers = {"authorization": f"Bearer {self._token}", "accept": "text/event-stream"}
        try:
            request = self._streaming.build_request("GET", f"{self._base}/events", headers=headers)
            response = self._streaming.send(request, stream=True)
        except httpx.HTTPError as e:
            raise RobotError(f"the rendezvous could not be reached: {e}") from None
        if response.status_code == 401:
            raise RobotError("the rendezvous refused this token — `hf auth login` again")
        if response.status_code != 200:
            raise RobotError(f"the event stream answered HTTP {response.status_code}")
        self._response = response
        self._thread = threading.Thread(target=self._pump, name="duck-lane", daemon=True)
        self._thread.start()

        try:
            welcome = self._welcome.get(timeout=CONNECT_TIMEOUT)
        except queue.Empty:
            self.close()
            raise RobotError("the rendezvous accepted the stream and never said hello") from None
        logger.info("welcome: account %s", welcome.get("username") or "unknown")

        # A name in the listing, so whoever else looks sees what holds the robot.
        self._post({"type": "setPeerStatus", "roles": ["listener"], "meta": {"name": self._label}})

        # **`startSession`'s answer is in the POST body**, not on the stream — the one shape a
        # reader of this protocol gets wrong once. Nothing about the session commits either end to
        # WebRTC: no offer is sent, and the control lane needs none.
        answer = self._post({"type": "startSession", "peerId": self._peer_id}) or {}
        kind = answer.get("type")
        if kind == "sessionRejected":
            self.close()
            raise RobotError(
                f"this duck is busy with {answer.get('activeApp') or 'something else'} — one "
                "consumer at a time, and a console open on it (`duckctl open`) counts"
            )
        if kind != "sessionStarted":
            self.close()
            raise RobotError(f"`startSession` answered {answer!r} rather than a session")
        self.session_id = answer.get("sessionId")
        logger.info("session %s open, control only", (self.session_id or "?")[:8])
        return self

    def close(self) -> None:
        self._stop.set()
        if self.session_id:
            with contextlib.suppress(RobotError):
                self._post({"type": "endSession", "sessionId": self.session_id})
        self.session_id = None
        if self._response is not None:
            self._response.close()
            self._response = None
        self._streaming.close()
        self._posting.close()
        self._abandon("the lane was closed")

    def __enter__(self) -> Lane:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── calls ────────────────────────────────────────────────────────────────

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None):
        """One request, one answer. `RpcError` is the robot saying no; `RobotError` is no robot."""
        if not self.session_id:
            raise RobotError("no session — the lane is not open")
        call_id = next(self._ids)
        future: Future = Future()
        with self._lock:
            self._pending[call_id] = (method, future)
        envelope = {"jsonrpc": "2.0", "id": call_id, "method": method, "params": params or {}}
        logger.info("→ %s %s", method, json.dumps(params or {}))
        try:
            self._post({"type": "peer", "sessionId": self.session_id, "rpc": envelope})
        except RobotError:
            with self._lock:
                self._pending.pop(call_id, None)
            raise
        try:
            return future.result(timeout=self.timeout if timeout is None else timeout)
        except FutureTimeout:
            with self._lock:
                self._pending.pop(call_id, None)
            raise RobotError(
                f"{method}: no answer from the robot. The session opened, so it is listening; "
                "a `mediad` from before the control lane drops these envelopes silently — update "
                "the robot."
            ) from None

    def _post(self, message: dict[str, Any]) -> dict[str, Any] | None:
        try:
            answer = self._posting.post(
                f"{self._base}/send",
                headers={"authorization": f"Bearer {self._token}"},
                json=message,
            )
        except httpx.HTTPError as e:
            raise RobotError(f"POST /send: {e}") from None
        if answer.status_code == 429:
            raise RobotError("the rendezvous is rate-limiting this token (1200 requests a minute)")
        if answer.status_code == 400:
            raise RobotError("the rendezvous says this peer does not exist — its stream is gone")
        if answer.status_code != 200:
            raise RobotError(f"POST /send answered HTTP {answer.status_code}: {answer.text[:200]}")
        try:
            body = answer.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) and body.get("type") else None

    # ── the stream ───────────────────────────────────────────────────────────

    def _pump(self) -> None:
        """Read SSE frames and dispatch them. CRLF normalised: the framing is a proxy's."""
        assert self._response is not None
        buffer = ""
        try:
            for chunk in self._response.iter_text():
                if self._stop.is_set():
                    return
                buffer += chunk.replace("\r\n", "\n")
                while "\n\n" in buffer:
                    frame, _, buffer = buffer.partition("\n\n")
                    data = sse_data(frame)
                    if data:
                        self._handle(data)
        except Exception as e:  # noqa: BLE001 - closing the response mid-read raises anything
            if not self._stop.is_set():
                self.error = f"the event stream failed: {type(e).__name__}: {e}"
                logger.warning("%s", self.error)
        finally:
            if not self._stop.is_set():
                self._abandon(self.error or "the rendezvous closed the event stream")

    def _handle(self, data: str) -> None:
        try:
            message = json.loads(data)
        except ValueError:
            return
        kind = message.get("type")
        if kind == "welcome":
            with contextlib.suppress(queue.Full):
                self._welcome.put_nowait(message)
        elif kind == "peer":
            payload = message.get("rpc")
            # An `sdp` or `ice` is the robot's media path answering a negotiation this lane never
            # started; not an error, and nothing here can use it.
            if isinstance(payload, dict):
                self._settle(payload)
        elif kind in ("endSession", "sessionRejected"):
            self.session_id = None
            self._abandon(f"the session ended: {message.get('reason') or 'no reason given'}")

    def _settle(self, payload: dict[str, Any]) -> None:
        call_id = payload.get("id")
        if call_id is None:
            # A notification — `robot.state`, `media.detections`. Nobody here asked for one.
            return
        with self._lock:
            waiting = self._pending.pop(call_id, None)
        if waiting is None:
            return
        method, future = waiting
        if "error" in payload:
            future.set_exception(RpcError(method, payload.get("error") or {}))
        else:
            future.set_result(payload.get("result"))

    def _abandon(self, why: str) -> None:
        with self._lock:
            pending, self._pending = self._pending, {}
        for method, future in pending.values():
            if not future.done():
                future.set_exception(RobotError(f"{method}: {why}"))


def sse_data(frame: str) -> str:
    """The `data:` payload of one server-sent event, joined across lines."""
    return "".join(
        line[len("data:") :].strip() for line in frame.split("\n") if line.startswith("data:")
    )
