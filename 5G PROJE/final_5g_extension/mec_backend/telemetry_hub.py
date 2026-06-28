"""Authenticated, idempotent result telemetry for the GPU media service."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class TelemetryHub:
    def __init__(self, max_devices: int = 1_000) -> None:
        self._clients: dict[Any, str] = {}
        self._snapshots: OrderedDict[str, dict] = OrderedDict()
        self._max_devices = max_devices

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def attach(self, websocket: Any, device_id: str) -> None:
        self._clients[websocket] = device_id

    def detach(self, websocket: Any) -> None:
        self._clients.pop(websocket, None)

    def snapshot(self, device_id: str) -> dict | None:
        value = self._snapshots.get(device_id)
        return dict(value) if value else None

    async def publish(self, device_id: str, envelope: dict) -> None:
        if not envelope.get("event_id"):
            raise ValueError("telemetry requires event_id")
        self._snapshots[device_id] = dict(envelope)
        self._snapshots.move_to_end(device_id)
        while len(self._snapshots) > self._max_devices:
            self._snapshots.popitem(last=False)
        stale = []
        for websocket, owner in list(self._clients.items()):
            if owner != device_id:
                continue
            try:
                await websocket.send_json(envelope)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.detach(websocket)

