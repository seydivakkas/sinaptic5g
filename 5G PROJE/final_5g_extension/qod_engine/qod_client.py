"""Provider-isolated CAMARA/Turkcell Quality on Demand adapter.

Provider paths and field names are intentionally confined to this module.
The configured ``qosProfile`` is treated as an opaque onboarding value; the
application never invents numeric bandwidth or latency guarantees for it.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import asdict, dataclass
from typing import Any, Optional

import httpx

from .oauth_manager import OAuthManager


LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlowDescriptor:
    application_server_ipv4: str
    application_server_port: int
    device_port: int
    protocol: str = "TCP"

    def validate(self) -> None:
        if not self.application_server_ipv4:
            raise ValueError("application server IPv4/CIDR is required")
        if not 0 <= self.device_port <= 65535:
            raise ValueError("device port is outside [0, 65535]")
        if not 1 <= self.application_server_port <= 65535:
            raise ValueError("application server port is outside [1, 65535]")
        if self.protocol.upper() not in {"TCP", "UDP"}:
            raise ValueError("transport protocol must be TCP or UDP")


class CamaraQodAdapter:
    MAX_RETRIES = 3

    def __init__(
        self,
        base_url: str,
        oauth_manager: Optional[OAuthManager] = None,
        timeout_seconds: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.oauth = oauth_manager or OAuthManager()
        self.timeout_seconds = timeout_seconds

    async def create_session(
        self,
        *,
        phone_number: str,
        flow_descriptor: FlowDescriptor,
        qos_profile: str,
        requested_duration: int,
        notification_sink: str | None = None,
    ) -> dict[str, Any]:
        flow_descriptor.validate()
        if not qos_profile:
            raise ValueError("provider-issued qosProfile is required")
        if not 1 <= requested_duration <= 3600:
            raise ValueError("requested duration is outside [1, 3600]")

        payload: dict[str, Any] = {
            "device": {"phoneNumber": phone_number},
            "applicationServer": {"ipv4Address": flow_descriptor.application_server_ipv4},
            "devicePorts": {"ports": [flow_descriptor.device_port]},
            "applicationServerPorts": {"ports": [flow_descriptor.application_server_port]},
            "protocol": flow_descriptor.protocol.upper(),
            "qosProfile": qos_profile,
            "duration": requested_duration,
        }
        if notification_sink:
            payload["sink"] = notification_sink
        return await self._request("POST", "/sessions", scope="qod:sessions:write", json=payload)

    async def get_session(self, session_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/sessions/{session_id}", scope="qod:sessions:read")

    async def retrieve_sessions(self, phone_number: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/retrieve-sessions",
            scope="qod:sessions:read",
            json={"device": {"phoneNumber": phone_number}},
        )

    async def extend_session(self, session_id: str, additional_duration: int) -> dict[str, Any]:
        if not 1 <= additional_duration <= 3600:
            raise ValueError("additional duration is outside [1, 3600]")
        return await self._request(
            "POST",
            f"/sessions/{session_id}/extend",
            scope="qod:sessions:write",
            json={"requestedAdditionalDuration": additional_duration},
        )

    async def delete_session(self, session_id: str) -> bool:
        try:
            await self._request("DELETE", f"/sessions/{session_id}", scope="qod:sessions:write")
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return True
            raise

    async def _request(self, method: str, path: str, *, scope: str, json: dict | None = None) -> dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("QoD provider URL is not configured")
        token = await self.oauth.get_token(scope=scope)
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        delay = 0.5
        last_error: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.request(
                        method,
                        f"{self.base_url}{path}",
                        headers=headers,
                        json=json,
                    )
                if response.status_code == 404 and method == "DELETE":
                    response.raise_for_status()
                if response.status_code < 500 and response.status_code != 429:
                    response.raise_for_status()
                    return response.json() if response.content else {}
                last_error = httpx.HTTPStatusError(
                    f"provider returned {response.status_code}", request=response.request, response=response
                )
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise
                last_error = exc
            if attempt + 1 < self.MAX_RETRIES:
                await asyncio.sleep(delay + random.uniform(0.0, delay * 0.2))
                delay *= 2
        raise RuntimeError(f"QoD provider unavailable after {self.MAX_RETRIES} attempts") from last_error


# Backward-compatible name for imports; semantics are the provider adapter above.
QoDClient = CamaraQodAdapter
