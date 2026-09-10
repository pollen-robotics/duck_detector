"""The control lane, against a signalling server and a WebRTC peer standing in for `mediad`.

Real WebRTC over loopback: the fake robot offers, opens a `control` datachannel and answers
JSON-RPC on it, so the whole handshake the console page does — welcome, list, startSession, offer
in, answer out, channel up — is exercised rather than mocked.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiortc import RTCPeerConnection, RTCSessionDescription

from duck_detector import robot


class FakeMediad:
    """gst-plugins-rs's signalling protocol, one producer, and the robot's end of the channel."""

    def __init__(self, *, producers=True, refuse_session=False):
        self.producers = producers
        self.refuse_session = refuse_session
        self.received: list[dict] = []
        self.pc: RTCPeerConnection | None = None
        self.server = None

    async def __aenter__(self):
        from websockets.asyncio.server import serve

        self.server = await serve(self.handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_):
        if self.pc is not None:
            await self.pc.close()
        self.server.close()
        await self.server.wait_closed()

    async def handle(self, ws):
        await ws.send(json.dumps({"type": "welcome", "peerId": "consumer-1"}))
        async for raw in ws:
            message = json.loads(raw)
            self.received.append(message)
            kind = message.get("type")
            if kind == "list":
                producers = (
                    [
                        {
                            "id": "prod-1",
                            "meta": {"name": "graphite", "serial": "cec2", "release": "0.9.4"},
                        }
                    ]
                    if self.producers
                    else []
                )
                await ws.send(json.dumps({"type": "list", "producers": producers}))
            elif kind == "startSession":
                if self.refuse_session:
                    await ws.send(json.dumps({"type": "error", "details": "session already open"}))
                    continue
                await ws.send(json.dumps({"type": "sessionStarted", "sessionId": "S1"}))
                await self.offer(ws)
            elif kind == "peer" and message.get("sdp"):
                sdp = message["sdp"]
                await self.pc.setRemoteDescription(RTCSessionDescription(sdp["sdp"], sdp["type"]))
            elif kind == "endSession":
                await ws.send(json.dumps({"type": "endSession", "sessionId": "S1"}))

    async def offer(self, ws):
        self.pc = RTCPeerConnection()
        channel = self.pc.createDataChannel("control")

        @channel.on("message")
        def on_message(raw):
            request = json.loads(raw)
            method = request["method"]
            if method == "media.video":
                reply = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "error": {"code": -32601, "message": "not here"},
                }
            elif method == "silent":
                return
            else:
                reply = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"echo": request.get("params")},
                }
            # A notification first, which nobody asked for and nothing must break on.
            channel.send(json.dumps({"jsonrpc": "2.0", "method": "robot.state", "params": {}}))
            channel.send(json.dumps(reply))

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        local = self.pc.localDescription
        await ws.send(
            json.dumps(
                {"type": "peer", "sessionId": "S1", "sdp": {"type": local.type, "sdp": local.sdp}}
            )
        )


def run(coroutine):
    return asyncio.run(asyncio.wait_for(coroutine, 30))


def test_a_call_crosses_the_channel_and_a_refusal_is_a_refusal():
    async def go():
        async with FakeMediad() as fake:
            async with robot.Lane("127.0.0.1", port=fake.port, timeout=5) as lane:
                assert lane.session_id == "S1"
                assert lane.producer.name == "graphite" and lane.producer.serial == "cec2"
                answer = await lane.call("media.stream", {"url": "ws://10.0.0.2:8765/frames"})
                assert answer == {"echo": {"url": "ws://10.0.0.2:8765/frames"}}
                with pytest.raises(robot.RpcError, match="not here"):
                    await lane.call("media.video")
            kinds = [m["type"] for m in fake.received]
            assert kinds[:2] == ["list", "startSession"]
            assert "endSession" in kinds, "closing the lane frees the robot for the next consumer"
            assert any(m.get("sdp", {}).get("type") == "answer" for m in fake.received)

    run(go())


def test_a_robot_with_no_producer_says_so():
    async def go():
        async with FakeMediad(producers=False) as fake:
            with pytest.raises(robot.RobotError, match="lists no producer"):
                await robot.Lane("127.0.0.1", port=fake.port).open()

    run(go())


def test_a_refused_session_is_said_plainly():
    async def go():
        async with FakeMediad(refuse_session=True) as fake:
            with pytest.raises(robot.RobotError, match="refused the session"):
                await robot.Lane("127.0.0.1", port=fake.port).open()

    run(go())


def test_nothing_listening_is_the_first_diagnosis():
    async def go():
        with pytest.raises(robot.RobotError, match="nothing answers at ws://127.0.0.1:1"):
            await robot.Lane("127.0.0.1", port=1).open()

    run(go())


def test_silence_from_the_robot_times_out():
    async def go():
        async with FakeMediad() as fake:
            async with robot.Lane("127.0.0.1", port=fake.port, timeout=0.3) as lane:
                with pytest.raises(robot.RobotError, match="no answer from the robot"):
                    await lane.call("silent")

    run(go())


def test_find_host_takes_what_it_is_given():
    assert robot.find_host("192.168.10.124") == "192.168.10.124"
