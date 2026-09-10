"""A control channel to a duck on the LAN, the way `duckctl open`'s page gets one.

`mediad` runs a WebRTC signalling server on the robot (port 8443, gst-plugins-rs's protocol) and
opens a `control` datachannel to every peer that negotiates a session. The console page does this
in a browser; this does it in Python with `aiortc`, and speaks JSON-RPC 2.0 down the channel —
one object per message, `duck-ipc-proto`'s wire.

    ws://robot:8443   →  welcome  →  list  →  startSession  →  sessionStarted
                      ←  peer {sdp: offer}         →  peer {sdp: answer}
                      ⇄  peer {ice}
    datachannel "control"  ⇄  {"jsonrpc": "2.0", "id": 1, "method": "media.stream", …}

Nothing here leaves the LAN and no account is involved: whoever can reach the signalling port can
drive the robot, which is `mediad`'s documented stance and why `--host` is an address rather than a
login. The robot also streams its video track to any peer; it is drained and discarded here, since
the frames a capture wants come the other way (`capture.py`).

The rendezvous path — the same handshake relayed through a Hugging Face Space — exists in the robot
repo and worked for the control lane, but media across the internet does not yet, so this tool is
LAN only on purpose.

**One consumer at a time.** `mediad` refuses a second session, so a console open on the robot
(`duckctl open`) makes it busy, and vice versa.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

SIGNALLING_PORT = 8443
CONNECT_TIMEOUT = 20.0
CONTROL_LABEL = "control"


def patch_dtls_ciphers() -> None:
    """Add a DTLS cipher GStreamer's `webrtcsink` accepts to aiortc's list.

    **Without this, ICE completes and the channel never opens.** aiortc's default cipher list
    shares no cipher with the robot's `webrtcsink`, so the DTLS handshake fails silently after a
    successful ICE — a failure that looks exactly like a firewall and is not one. The same shim
    `reachy_mini.media.central_consumer` applies, written here so this repo does not depend on
    theirs for five lines. Upstream: aiortc PR #1392; drop this once a release negotiates a common
    cipher by default. Idempotent.
    """
    from aiortc.rtcdtlstransport import RTCCertificate

    if getattr(RTCCertificate, "_duck_cipher_patched", False):
        return
    original = RTCCertificate._create_ssl_context
    ciphers = (
        b"ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-ECDSA-CHACHA20-POLY1305:"
        b"ECDHE-ECDSA-AES128-SHA:ECDHE-ECDSA-AES256-SHA:"
        b"ECDHE-RSA-AES128-GCM-SHA256"
    )

    def patched(self, *args, **kwargs):
        context = original(self, *args, **kwargs)
        try:
            context.set_cipher_list(ciphers)
        except Exception as e:  # noqa: BLE001 - a context that will not take it still works
            logger.warning("could not extend the DTLS cipher list: %r", e)
        return context

    RTCCertificate._create_ssl_context = patched
    RTCCertificate._duck_cipher_patched = True


class RobotError(Exception):
    """Something the person at the keyboard can act on."""


class RpcError(Exception):
    """A refusal from the robot, distinct from a transport failure on purpose."""

    def __init__(self, method: str, error: dict[str, Any]):
        self.method = method
        self.code = error.get("code")
        self.message = error.get("message") or "refused, with nothing said about why"
        super().__init__(f"{method}: {self.message}")


@dataclass
class Producer:
    """The robot as its signalling server lists it, before a session exists.

    `meta` is what `mediad/src/producer.rs` registers: `name`, `serial`, `release`, `api_version`.
    """

    peer_id: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.peer_id[:8]

    @property
    def serial(self) -> str | None:
        return self.meta.get("serial")

    @property
    def release(self) -> str | None:
        return self.meta.get("release")


def find_host(host: str | None) -> str:
    """The address given, else the one `duckctl ip` finds over Bluetooth."""
    if host:
        return host
    if not shutil.which("duckctl"):
        raise RobotError("no --host, and no `duckctl` on this machine to find one with")
    logger.info("no --host; asking duckctl over Bluetooth")
    try:
        found = subprocess.run(
            ["duckctl", "ip"], capture_output=True, text=True, timeout=60, check=False
        )
    except subprocess.TimeoutExpired:
        raise RobotError("`duckctl ip` did not answer in a minute; pass --host") from None
    address = found.stdout.strip().splitlines()[-1] if found.stdout.strip() else ""
    if found.returncode != 0 or not address:
        raise RobotError(
            f"`duckctl ip` found no robot:\n{found.stderr.strip()[-600:]}\nPass --host <address>."
        )
    return address


class Lane:
    """One session with one duck: JSON-RPC over its `control` datachannel.

    `async`, because both halves — the signalling socket and the peer connection — are, and so is
    the capture that uses this. `open` returns once the channel is open, which is the point at
    which the robot will answer a call.
    """

    def __init__(self, host: str, *, port: int = SIGNALLING_PORT, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.producer: Producer | None = None
        self.session_id: str | None = None
        self._ws = None
        self._pc = None
        self._channel = None
        self._channel_open: asyncio.Future | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple[str, asyncio.Future]] = {}
        self._pump: asyncio.Task | None = None
        # Candidates that arrive before the offer — ordinary trickle, held until there is a
        # remote description to add them to.
        self._early_ice: list[dict[str, Any]] = []
        self._remote_set = False

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def open(self) -> Lane:
        from aiortc import RTCPeerConnection
        from websockets.asyncio.client import connect

        patch_dtls_ciphers()
        loop = asyncio.get_running_loop()
        self._channel_open = loop.create_future()
        try:
            self._ws = await asyncio.wait_for(connect(self.url), CONNECT_TIMEOUT)
        except (OSError, TimeoutError) as e:
            raise RobotError(
                f"nothing answers at {self.url}: {e}. That is mediad's signalling port — is the "
                "robot on, and on this network? `duckctl ip` says where it is."
            ) from None

        try:
            welcome = await self._expect("welcome")
            logger.info("welcome: peer %s", (welcome.get("peerId") or "?")[:8])
            await self._send({"type": "list"})
            listing = await self._expect("list")
            producers = [
                Producer(p["id"], p.get("meta") or {}) for p in listing.get("producers") or []
            ]
            if not producers:
                raise RobotError(
                    f"{self.url} lists no producer. mediad registers one when its pipeline is "
                    "playing — `journalctl -u mediad -b` on the robot says why it is not."
                )
            self.producer = producers[0]
            logger.info("producer: %s %s", self.producer.name, json.dumps(self.producer.meta))

            # No offer from us: the producer offers, because it knows what it is sending.
            self._pc = RTCPeerConnection()
            self._pc.on("datachannel", self._on_datachannel)
            self._pc.on("track", self._on_track)
            for state in ("connectionstatechange", "iceconnectionstatechange"):
                self._pc.on(state, self._log_state)
            await self._send({"type": "startSession", "peerId": self.producer.peer_id})
            started = await self._expect("sessionStarted", also=("sessionRejected", "error"))
            if started.get("type") != "sessionStarted":
                raise RobotError(
                    f"{self.producer.name} refused the session: "
                    f"{started.get('details') or started.get('reason') or started}. One consumer "
                    "at a time — a console open on it (`duckctl open`) counts."
                )
            self.session_id = started.get("sessionId")

            self._pump = asyncio.ensure_future(self._read_signalling())
            try:
                await asyncio.wait_for(asyncio.shield(self._channel_open), CONNECT_TIMEOUT)
            except TimeoutError:
                raise RobotError(
                    "the session opened and the control channel never did "
                    f"(ice {self._pc.iceConnectionState}, connection {self._pc.connectionState}). "
                    "ICE failed is a firewall or two networks that do not route UDP; ICE completed "
                    "and connection failed is DTLS — `journalctl -u mediad -b` on the robot."
                ) from None
            return self
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        if self._pump is not None:
            self._pump.cancel()
            self._pump = None
        if self._ws is not None and self.session_id:
            try:
                await self._send({"type": "endSession", "sessionId": self.session_id})
            except Exception:  # noqa: BLE001 - a teardown that cannot be delivered still ends
                pass
        self.session_id = None
        if self._pc is not None:
            await self._pc.close()
            self._pc = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._abandon("the lane was closed")

    async def __aenter__(self) -> Lane:
        return await self.open()

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── calls ────────────────────────────────────────────────────────────────

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """One request, one answer. `RpcError` is the robot saying no; `RobotError` is no robot."""
        if self._channel is None or self._channel.readyState != "open":
            raise RobotError(f"{method}: no control channel")
        call_id = next(self._ids)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[call_id] = (method, future)
        logger.info("→ %s %s", method, json.dumps(params or {}))
        self._channel.send(
            json.dumps({"jsonrpc": "2.0", "id": call_id, "method": method, "params": params or {}})
        )
        try:
            return await asyncio.wait_for(future, self.timeout)
        except TimeoutError:
            self._pending.pop(call_id, None)
            raise RobotError(f"{method}: no answer from the robot in {self.timeout:.0f}s") from None

    # ── the wire ─────────────────────────────────────────────────────────────

    async def _send(self, message: dict[str, Any]) -> None:
        logger.debug("→ signalling %s", json.dumps(message)[:200])
        await self._ws.send(json.dumps(message))

    async def _receive(self) -> dict[str, Any]:
        raw = await asyncio.wait_for(self._ws.recv(), CONNECT_TIMEOUT)
        message = json.loads(raw)
        if message.get("type") != "peerStatusChanged":
            logger.debug("← signalling %s", raw[:200])
        return message

    async def _expect(self, kind: str, also: tuple[str, ...] = ()) -> dict[str, Any]:
        """The next message of one of these kinds, skipping the status chatter the server pushes."""
        try:
            while True:
                message = await self._receive()
                if message.get("type") in (kind, *also):
                    return message
        except TimeoutError:
            raise RobotError(f"{self.url} never sent {kind!r}") from None

    async def _read_signalling(self) -> None:
        """After the session: offers and candidates in, answers and candidates out."""
        try:
            while True:
                message = json.loads(await self._ws.recv())
                kind = message.get("type")
                if kind == "peer" and message.get("sdp"):
                    await self._answer(message["sdp"])
                elif kind == "peer" and message.get("ice"):
                    if self._remote_set:
                        await self._add_ice(message["ice"])
                    else:
                        self._early_ice.append(message["ice"])
                elif kind == "endSession":
                    self.session_id = None
                    self._abandon("the robot ended the session")
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - the socket closing mid-read raises anything
            self._abandon(f"the signalling socket failed: {type(e).__name__}: {e}")

    async def _answer(self, offer: dict[str, Any]) -> None:
        from aiortc import RTCSessionDescription

        await self._pc.setRemoteDescription(RTCSessionDescription(offer["sdp"], offer["type"]))
        self._remote_set = True
        early, self._early_ice = self._early_ice, []
        for ice in early:
            await self._add_ice(ice)
        answer = await self._pc.createAnswer()
        # `setLocalDescription` gathers ICE before it returns, so the answer carries every
        # candidate and nothing needs trickling from this side.
        await self._pc.setLocalDescription(answer)
        local = self._pc.localDescription
        await self._send(
            {
                "type": "peer",
                "sessionId": self.session_id,
                "sdp": {"type": local.type, "sdp": local.sdp},
            }
        )

    async def _add_ice(self, ice: dict[str, Any]) -> None:
        from aiortc.sdp import candidate_from_sdp

        text = (ice.get("candidate") or "").strip()
        if not text:
            return  # end-of-candidates; aiortc wants nothing for it
        try:
            candidate = candidate_from_sdp(text.removeprefix("candidate:"))
            candidate.sdpMid = ice.get("sdpMid")
            if ice.get("sdpMLineIndex") is not None:
                candidate.sdpMLineIndex = int(ice["sdpMLineIndex"])
            await self._pc.addIceCandidate(candidate)
        except Exception as e:  # noqa: BLE001 - one bad candidate is not a dead session
            logger.warning("addIceCandidate failed: %r (%r)", e, text[:80])

    def _log_state(self) -> None:
        logger.info(
            "peer connection: ice %s, connection %s",
            self._pc.iceConnectionState,
            self._pc.connectionState,
        )

    def _on_datachannel(self, channel) -> None:
        logger.info("datachannel: %s", channel.label)
        if channel.label != CONTROL_LABEL:
            return
        self._channel = channel
        channel.on("message", self._on_message)
        channel.on("close", lambda: self._abandon("the control channel closed"))
        if channel.readyState == "open":
            self._opened()
        else:
            channel.on("open", self._opened)

    def _opened(self) -> None:
        if self._channel_open is not None and not self._channel_open.done():
            self._channel_open.set_result(True)

    def _on_track(self, track) -> None:
        # The video the console would show. Drained so aiortc's decoder queue does not grow
        # without bound; the frames a capture wants arrive by `media.stream`, not here.
        async def drain() -> None:
            try:
                while True:
                    await track.recv()
            except Exception:  # noqa: BLE001 - the track ending is the only way out
                pass

        asyncio.ensure_future(drain())

    def _on_message(self, raw: Any) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            message = json.loads(raw)
        except ValueError:
            return
        call_id = message.get("id")
        if call_id is None:
            # `robot.state`, `media.detections`, `media.video`: notifications nobody here asked for.
            return
        waiting = self._pending.pop(call_id, None)
        if waiting is None:
            return
        method, future = waiting
        if future.done():
            return
        if "error" in message:
            future.set_exception(RpcError(method, message.get("error") or {}))
        else:
            future.set_result(message.get("result"))

    def _abandon(self, why: str) -> None:
        pending, self._pending = self._pending, {}
        for method, future in pending.values():
            if not future.done():
                future.set_exception(RobotError(f"{method}: {why}"))
