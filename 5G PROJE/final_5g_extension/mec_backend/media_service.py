"""GPU WebRTC media plane for SİNAPTİC5G.

The BFF is contacted only to exchange complete ICE-gathered SDP. Encoded RTP
arrives here directly from Android and is never relayed through the BFF.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from latest_frame import LatestFrameQueue
from telemetry_hub import TelemetryHub
from vehicle_sentinel import VehicleSentinel
from auth_utils import verify_token

try:
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
except ImportError:  # Health endpoint remains available in a non-media dev env.
    RTCConfiguration = RTCIceServer = RTCPeerConnection = RTCSessionDescription = None


logger = logging.getLogger(__name__)
BFF_BASE_URL = os.getenv("BFF_BASE_URL", "http://live_bff:8000").rstrip("/")
MEDIA_SERVICE_TOKEN = os.getenv("MEDIA_SERVICE_TOKEN", "")
ICE_SERVERS_JSON = os.getenv("WEBRTC_ICE_SERVERS_JSON", "[]")
MODEL_PATH = os.getenv("MEDIA_MODEL_PATH", "/models/yolov8n.onnx")
MANIFEST_PATH = os.getenv(
    "MEDIA_MODEL_MANIFEST_PATH", "/models/android_vehicle_sentinel.manifest.json"
)


class PeerSession:
    def __init__(self, device_id: str, peer: Any, detector: VehicleSentinel, hub: TelemetryHub) -> None:
        self.device_id = device_id
        self.peer = peer
        self.detector = detector
        self.hub = hub
        self.frames: LatestFrameQueue[tuple[Any, int, float]] = LatestFrameQueue()
        self.frame_id = 0
        self.received_rtp_frames = 0
        self.receiver_task: asyncio.Task | None = None
        self.worker_task = asyncio.create_task(self._infer_loop())

    def attach_track(self, track: Any) -> None:
        if track.kind != "video" or self.receiver_task:
            return
        self.receiver_task = asyncio.create_task(self._receive(track))

    async def _receive(self, track: Any) -> None:
        try:
            while True:
                frame = await track.recv()
                self.frame_id += 1
                self.received_rtp_frames += 1
                self.frames.put_latest((frame, self.frame_id, time.time()))
        except Exception as exc:
            logger.info("Video track ended for device %s: %s", self.device_id, type(exc).__name__)

    async def _infer_loop(self) -> None:
        while True:
            frame, frame_id, received_at = await self.frames.get()
            try:
                bgr = frame.to_ndarray(format="bgr24")
                detections, processing_ms = await asyncio.to_thread(self.detector.infer, bgr)
                completed_at = time.time()
                confidence = max((item["confidence"] for item in detections), default=0.0)
                risk_score = round(confidence * 100.0, 2)
                risk_level = "kritik" if risk_score >= 70 else "orta" if risk_score >= 45 else "dusuk"
                await self.hub.publish(self.device_id, {
                    "type": "analysis_result",
                    "event_id": str(uuid.uuid4()),
                    "frame_id": frame_id,
                    "timestamp": completed_at,
                    # RTP PTS is not a wall clock. Until a synchronized capture
                    # extension is negotiated, receipt time is the honest bound.
                    "captured_at": received_at,
                    "media_time_seconds": float(frame.time) if frame.time is not None else None,
                    "received_at": received_at,
                    "inference_completed_at": completed_at,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "detections": detections,
                    "processing_time_ms": round(processing_ms, 2),
                    "dropped_frames": self.frames.dropped_count,
                    "license_plate": "01A0000",
                    "vehicle_type": None,
                    "vehicle_color": None,
                    "behavior_class": None,
                    "speed_kmh": -1.0,
                    "driver_state": None,
                    "network_metrics": None,
                })
            finally:
                self.frames.task_done()

    async def close(self) -> None:
        for task in (self.receiver_task, self.worker_task):
            if task:
                task.cancel()
        await self.peer.close()


class MediaRuntime:
    def __init__(self) -> None:
        self.hub = TelemetryHub()
        self.detector: VehicleSentinel | None = None
        self.sessions: dict[str, PeerSession] = {}
        self.poller: asyncio.Task | None = None
        self.client: httpx.AsyncClient | None = None

    @property
    def ready(self) -> bool:
        return bool(self.detector and RTCPeerConnection and MEDIA_SERVICE_TOKEN)

    async def start(self) -> None:
        self.client = httpx.AsyncClient(timeout=25.0)
        if RTCPeerConnection is None:
            logger.error("aiortc is unavailable; media plane is not ready")
            return
        if not MEDIA_SERVICE_TOKEN:
            logger.error("MEDIA_SERVICE_TOKEN is missing; media plane is not ready")
            return
        try:
            self.detector = VehicleSentinel(MODEL_PATH, MANIFEST_PATH)
        except Exception as exc:
            logger.error("Vehicle sentinel unavailable: %s", exc)
            return
        self.poller = asyncio.create_task(self._poll_offers())

    def _configuration(self) -> Any:
        servers = []
        for entry in json.loads(ICE_SERVERS_JSON):
            urls = entry.get("urls", [])
            servers.append(RTCIceServer(
                urls=urls if isinstance(urls, list) else [urls],
                username=entry.get("username"),
                credential=entry.get("credential"),
            ))
        return RTCConfiguration(iceServers=servers)

    async def _poll_offers(self) -> None:
        assert self.client and self.detector
        headers = {"Authorization": f"Bearer {MEDIA_SERVICE_TOKEN}"}
        while True:
            try:
                response = await self.client.get(
                    f"{BFF_BASE_URL}/webrtc/offer",
                    params={"wait_seconds": 20},
                    headers=headers,
                )
                if response.status_code == 204:
                    continue
                response.raise_for_status()
                asyncio.create_task(self._accept_offer(response.json(), headers))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Signaling poll failed: %s", type(exc).__name__)
                await asyncio.sleep(1)

    async def _accept_offer(self, offer: dict, headers: dict[str, str]) -> None:
        assert self.client and self.detector
        signaling_id = str(offer["signaling_id"])
        device_id = str(offer["device_id"])
        peer = RTCPeerConnection(configuration=self._configuration())
        session = PeerSession(device_id, peer, self.detector, self.hub)
        self.sessions[signaling_id] = session

        @peer.on("track")
        def on_track(track: Any) -> None:
            session.attach_track(track)

        @peer.on("connectionstatechange")
        async def on_connection_state_change() -> None:
            if peer.connectionState in {"failed", "closed", "disconnected"}:
                await session.close()
                self.sessions.pop(signaling_id, None)

        try:
            await peer.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type="offer"))
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
            response = await self.client.post(
                f"{BFF_BASE_URL}/webrtc/answer",
                headers=headers,
                json={
                    "signaling_id": signaling_id,
                    "type": "answer",
                    "sdp": peer.localDescription.sdp,
                },
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Offer could not be answered: %s", type(exc).__name__)
            await session.close()
            self.sessions.pop(signaling_id, None)

    async def stop(self) -> None:
        if self.poller:
            self.poller.cancel()
            with suppress(asyncio.CancelledError):
                await self.poller
        for session in list(self.sessions.values()):
            await session.close()
        if self.client:
            await self.client.aclose()


runtime = MediaRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await runtime.start()
    yield
    await runtime.stop()


app = FastAPI(title="SİNAPTİC5G GPU Media Service", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ready" if runtime.ready else "degraded",
        "media_plane": "direct_webrtc_rtp",
        "signaling_plane": "bff_sdp_only",
        "active_peers": len(runtime.sessions),
        "telemetry_clients": runtime.hub.client_count,
        "received_rtp_frames": sum(s.received_rtp_frames for s in runtime.sessions.values()),
        "dropped_frames": sum(s.frames.dropped_count for s in runtime.sessions.values()),
    }


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    authorization = websocket.headers.get("authorization", "")
    token = authorization.removeprefix("Bearer ").strip()
    device_id = websocket.headers.get("x-device-id", "")
    device = verify_token(token) if token else None
    if not device or device.get("device_id") != device_id:
        await websocket.close(code=1008, reason="valid device session required")
        return
    await websocket.accept()
    runtime.hub.attach(websocket, device_id)
    snapshot = runtime.hub.snapshot(device_id)
    if snapshot:
        await websocket.send_json({**snapshot, "type": "analysis_snapshot"})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        runtime.hub.detach(websocket)
