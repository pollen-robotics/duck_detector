"""The control lane, against a rendezvous small enough to run in a test.

The real one is `pollen-robotics/reachy-mini-central`; what matters here is the protocol's shape,
which is the part that was got wrong once already: `startSession` answers in the POST body,
everything else on the stream, and a `peer` envelope's `rpc` is the call.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from duck_detector import robot


class FakeRendezvous:
    """`GET /events` streams; `POST /send` answers `startSession` in the body and relays `rpc`.

    The "robot" on the far end answers every `rpc` with a canned result for its method, or refuses
    it — and `media.stream` is answered the way `mediad` does, so the shape of the whole exchange
    is the one a capture makes.
    """

    def __init__(self):
        self.events: queue.Queue[dict] = queue.Queue()
        self.posted: list[dict] = []
        self.refuse: dict[str, dict] = {}
        self.busy = False
        rendezvous = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.headers.get("authorization") != "Bearer good":
                    self.send_response(401)
                    self.end_headers()
                    return
                if self.path == "/api/robot-status":
                    body = json.dumps(
                        {
                            "robots": [
                                {
                                    "peerId": "duck-peer",
                                    "meta": {
                                        "name": "graphite",
                                        "kind": "microduck",
                                        "release": "0.9.4",
                                    },
                                    "busy": rendezvous.busy,
                                    "activeApp": "console" if rendezvous.busy else None,
                                },
                                {
                                    "peerId": "mini-peer",
                                    "meta": {"name": "mini", "kind": "reachy_mini"},
                                },
                                {"meta": {"name": "no peer id", "kind": "microduck"}},
                            ]
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                # CRLF on purpose: a proxy may rewrite the framing, and the reader must not care.
                self.wfile.write(
                    b'data: {"type": "welcome", "peerId": "me", "username": "u"}\r\n\r\n'
                )
                self.wfile.flush()
                try:
                    while True:
                        event = rendezvous.events.get()
                        if event is None:
                            return
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                message = json.loads(self.rfile.read(length) or b"{}")
                rendezvous.posted.append(message)
                answer: dict = {"status": "ok"}
                if message.get("type") == "startSession":
                    answer = (
                        {"type": "sessionRejected", "activeApp": "console"}
                        if rendezvous.busy
                        else {"type": "sessionStarted", "sessionId": "S1"}
                    )
                elif message.get("type") == "peer":
                    rpc = message["rpc"]
                    method = rpc["method"]
                    if method in rendezvous.refuse:
                        reply = {
                            "jsonrpc": "2.0",
                            "id": rpc["id"],
                            "error": rendezvous.refuse[method],
                        }
                    elif method == "media.stream":
                        params = rpc.get("params") or {}
                        reply = {
                            "jsonrpc": "2.0",
                            "id": rpc["id"],
                            "result": {"streaming": params.get("url") is not None, **params},
                        }
                    elif method == "silent":
                        reply = None
                    else:
                        reply = {"jsonrpc": "2.0", "id": rpc["id"], "result": {"method": method}}
                    if reply is not None:
                        # A notification first, which nobody asked for and nothing must break on.
                        rendezvous.events.put(
                            {
                                "type": "peer",
                                "sessionId": "S1",
                                "rpc": {"jsonrpc": "2.0", "method": "robot.state", "params": {}},
                            }
                        )
                        rendezvous.events.put({"type": "peer", "sessionId": "S1", "rpc": reply})
                body = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.events.put(None)
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def rendezvous():
    fake = FakeRendezvous()
    yield fake
    fake.close()


def test_the_listing_keeps_ducks_with_a_peer_id(rendezvous):
    found = robot.ducks("good", base=rendezvous.base)
    assert [d.name for d in found] == ["graphite"], (
        "a mini is not a duck, and no peer id is no robot"
    )
    assert robot.choose(found, None).peer_id == "duck-peer"
    assert robot.choose(found, "graphite").peer_id == "duck-peer"
    with pytest.raises(robot.RobotError, match="no duck called"):
        robot.choose(found, "onyx")
    with pytest.raises(robot.RobotError, match="no duck is online"):
        robot.choose([], None)


def test_a_bad_token_is_said_plainly(rendezvous):
    with pytest.raises(robot.RobotError, match="refused this token"):
        robot.ducks("bad", base=rendezvous.base)


def test_a_call_crosses_the_lane_and_a_refusal_is_a_refusal(rendezvous):
    rendezvous.refuse["media.video"] = {"code": -32601, "message": "not over this lane"}
    with robot.Lane("good", "duck-peer", base=rendezvous.base, timeout=5) as lane:
        assert lane.session_id == "S1"
        answer = lane.call("media.stream", {"url": "ws://10.0.0.2:8765/frames", "encoding": "jpeg"})
        assert answer["streaming"] is True and answer["url"] == "ws://10.0.0.2:8765/frames"
        assert lane.call("media.stream", {"url": None}) == {"streaming": False, "url": None}
        with pytest.raises(robot.RpcError, match="not over this lane"):
            lane.call("media.video")

    kinds = [m["type"] for m in rendezvous.posted]
    assert kinds[:2] == ["setPeerStatus", "startSession"], (
        "the stream binds the peer; then a session"
    )
    assert kinds[-1] == "endSession", "closing the lane ends the session for the next consumer"
    envelope = next(m for m in rendezvous.posted if m["type"] == "peer")
    assert envelope["sessionId"] == "S1" and envelope["rpc"]["jsonrpc"] == "2.0"


def test_a_busy_duck_is_refused_before_anything_is_asked(rendezvous):
    rendezvous.busy = True
    with pytest.raises(robot.RobotError, match="busy with console"):
        robot.Lane("good", "duck-peer", base=rendezvous.base).open()


def test_silence_from_the_robot_times_out_with_a_diagnosis(rendezvous):
    with robot.Lane("good", "duck-peer", base=rendezvous.base, timeout=0.3) as lane:
        with pytest.raises(robot.RobotError, match="no answer from the robot"):
            lane.call("silent")


def test_sse_data_joins_lines_and_ignores_comments():
    assert robot.sse_data(': ping\ndata: {"a":\ndata: 1}') == '{"a":1}'
    assert robot.sse_data(": ping") == ""
