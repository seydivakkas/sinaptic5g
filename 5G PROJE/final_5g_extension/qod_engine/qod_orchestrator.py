"""Benefit-gated QoD state machine.

QoD is an optional network optimization. Inference always continues in Best
Effort when this component is disabled, rejected or times out.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .qod_client import CamaraQodAdapter, FlowDescriptor


class QodState(str, Enum):
    IDLE = "IDLE"
    OBSERVE = "OBSERVE"
    REQUESTING = "REQUESTING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class BenefitSignals:
    target_present: bool
    vehicle_is_approaching: bool
    recognizability_gap: float
    model_uncertainty: float
    media_degradation: float
    network_degradation: float

    def values(self) -> dict[str, float]:
        return {
            "recognizability_gap": self.recognizability_gap,
            "model_uncertainty": self.model_uncertainty,
            "media_degradation": self.media_degradation,
            "network_degradation": self.network_degradation,
        }


@dataclass(frozen=True)
class BenefitDecision:
    allowed: bool
    expected_benefit: float
    reason: str


class CalibratedBenefitModel:
    """Linear gate loaded only from a measured validation artifact."""

    def __init__(self, *, threshold: float, bias: float, weights: dict[str, float], run_id: str):
        self.threshold = float(threshold)
        self.bias = float(bias)
        self.weights = {key: float(value) for key, value in weights.items()}
        self.run_id = run_id

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedBenefitModel":
        with Path(path).open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        if artifact.get("status") != "OLCULDU":
            raise ValueError("QoD benefit artifact must have status=OLCULDU")
        if not artifact.get("run_id") or not artifact.get("dataset_manifest_sha256"):
            raise ValueError("QoD benefit artifact lacks evidence identity")
        return cls(
            threshold=artifact["threshold"],
            bias=artifact.get("bias", 0.0),
            weights=artifact["weights"],
            run_id=artifact["run_id"],
        )

    def evaluate(self, signals: BenefitSignals) -> BenefitDecision:
        if not signals.target_present:
            return BenefitDecision(False, 0.0, "target_absent")
        if not signals.vehicle_is_approaching:
            return BenefitDecision(False, 0.0, "target_not_approaching")
        score = self.bias + sum(
            self.weights.get(name, 0.0) * max(0.0, min(1.0, value))
            for name, value in signals.values().items()
        )
        return BenefitDecision(score >= self.threshold, score, "calibrated_gate")


class QodOrchestrator:
    def __init__(
        self,
        adapter: CamaraQodAdapter,
        model: CalibratedBenefitModel | None,
        *,
        flow_descriptor: FlowDescriptor,
        qos_profile: str,
        duration_seconds: int,
        notification_sink: str | None = None,
        cooldown_seconds: float = 30.0,
    ):
        self.adapter = adapter
        self.model = model
        self.flow_descriptor = flow_descriptor
        self.qos_profile = qos_profile
        self.duration_seconds = duration_seconds
        self.notification_sink = notification_sink
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.states: dict[str, QodState] = {}
        self.sessions: dict[str, str] = {}
        self.cooldown_until: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def decide(self, device_id: str, signals: BenefitSignals) -> BenefitDecision:
        if device_id in self.sessions:
            return BenefitDecision(False, 0.0, "equivalent_session_active")
        state = self.states.get(device_id, QodState.IDLE)
        if state in {QodState.REQUESTING, QodState.ACTIVE}:
            return BenefitDecision(False, 0.0, "state_does_not_allow_request")
        if time.monotonic() < self.cooldown_until.get(device_id, 0.0):
            self.states[device_id] = QodState.COOLDOWN
            return BenefitDecision(False, 0.0, "cooldown_active")
        self.cooldown_until.pop(device_id, None)
        if self.model is None:
            return BenefitDecision(False, 0.0, "benefit_model_not_measured")
        self.states[device_id] = QodState.OBSERVE
        return self.model.evaluate(signals)

    async def request(self, device_id: str, phone_number: str, signals: BenefitSignals) -> dict[str, Any]:
        async with self._locks[device_id]:
            decision = self.decide(device_id, signals)
            if not decision.allowed:
                return {
                    "status": "best_effort",
                    "state": self.states.get(device_id, QodState.IDLE).value,
                    "reason": decision.reason,
                    "expected_benefit": decision.expected_benefit,
                }
            self.states[device_id] = QodState.REQUESTING
            try:
                existing = await self.reconcile(device_id, phone_number)
                if existing:
                    return {
                        "status": "active",
                        "state": QodState.ACTIVE.value,
                        "session_id": existing,
                        "qos_profile": self.qos_profile,
                        "expected_benefit": decision.expected_benefit,
                        "benefit_run_id": self.model.run_id,
                    }
                provider = await self.adapter.create_session(
                    phone_number=phone_number,
                    flow_descriptor=self.flow_descriptor,
                    qos_profile=self.qos_profile,
                    requested_duration=self.duration_seconds,
                    notification_sink=self.notification_sink,
                )
                session_id = str(provider["sessionId"])
                # The provider response, not Redis, is the source of truth.
                provider_state = str(provider.get("status", provider.get("state", ""))).upper()
                if provider_state and provider_state not in {"AVAILABLE", "ACTIVE", "REQUESTED"}:
                    raise RuntimeError(f"provider session is not available: {provider_state}")
                self.sessions[device_id] = session_id
                self.states[device_id] = QodState.ACTIVE
                return {
                    "status": "active",
                    "state": QodState.ACTIVE.value,
                    "session_id": session_id,
                    "qos_profile": self.qos_profile,
                    "expected_benefit": decision.expected_benefit,
                    "benefit_run_id": self.model.run_id,
                }
            except Exception:
                self.states[device_id] = QodState.FAILED
                self.cooldown_until[device_id] = time.monotonic() + self.cooldown_seconds
                return {
                    "status": "best_effort",
                    "state": QodState.FAILED.value,
                    "reason": "provider_unavailable",
                    "expected_benefit": decision.expected_benefit,
                    "benefit_run_id": self.model.run_id,
                }

    async def stop(self, device_id: str, session_id: str) -> bool:
        if self.sessions.get(device_id) != session_id:
            return False
        deleted = await self.adapter.delete_session(session_id)
        if self.sessions.get(device_id) == session_id:
            self.sessions.pop(device_id, None)
        self.states[device_id] = QodState.COOLDOWN
        self.cooldown_until[device_id] = time.monotonic() + self.cooldown_seconds
        return deleted

    def owns_session(self, device_id: str, session_id: str) -> bool:
        return self.sessions.get(device_id) == session_id

    async def extend(self, device_id: str, session_id: str, additional_duration: int) -> dict[str, Any]:
        if not self.owns_session(device_id, session_id):
            raise PermissionError("session not owned by device")
        provider = await self.adapter.extend_session(session_id, additional_duration)
        self.states[device_id] = QodState.ACTIVE
        return provider

    async def reconcile(self, device_id: str, phone_number: str) -> str | None:
        provider = await self.adapter.retrieve_sessions(phone_number)
        sessions = provider.get("sessions", provider if isinstance(provider, list) else [])
        if not isinstance(sessions, list):
            return None
        for session in sessions:
            state = str(session.get("status", session.get("state", ""))).upper()
            if state in {"AVAILABLE", "ACTIVE", "REQUESTED"} and session.get("sessionId"):
                session_id = str(session["sessionId"])
                self.sessions[device_id] = session_id
                self.states[device_id] = QodState.ACTIVE
                return session_id
        return None
