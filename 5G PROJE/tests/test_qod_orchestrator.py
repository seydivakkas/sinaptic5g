from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from api.qod_client import FlowDescriptor
from api.qod_orchestrator import BenefitSignals, CalibratedBenefitModel, QodOrchestrator


class FakeAdapter:
    def __init__(self):
        self.create_count = 0
        self.sessions = []

    async def retrieve_sessions(self, phone_number):
        return {"sessions": list(self.sessions)}

    async def create_session(self, **kwargs):
        self.create_count += 1
        session = {"sessionId": f"session-{self.create_count}", "status": "AVAILABLE"}
        self.sessions.append(session)
        return session

    async def delete_session(self, session_id):
        self.sessions = [item for item in self.sessions if item["sessionId"] != session_id]
        return True

    async def extend_session(self, session_id, additional_duration):
        return {"sessionId": session_id, "additionalDuration": additional_duration}


class FailingAdapter(FakeAdapter):
    async def retrieve_sessions(self, phone_number):
        raise TimeoutError("provider timeout")


def signals(**overrides):
    values = dict(
        target_present=True,
        vehicle_is_approaching=True,
        recognizability_gap=1.0,
        model_uncertainty=1.0,
        media_degradation=1.0,
        network_degradation=1.0,
    )
    values.update(overrides)
    return BenefitSignals(**values)


class QodOrchestratorTests(unittest.TestCase):
    def make_orchestrator(self, model):
        adapter = FakeAdapter()
        orchestrator = QodOrchestrator(
            adapter,
            model,
            flow_descriptor=FlowDescriptor("203.0.113.10", 443, 0, "TCP"),
            qos_profile="provider-profile",
            duration_seconds=300,
        )
        return adapter, orchestrator

    def test_qod_disabled_without_measured_model(self):
        adapter, orchestrator = self.make_orchestrator(None)
        result = asyncio.run(orchestrator.request("device", "+905551234567", signals()))
        self.assertEqual(result["status"], "best_effort")
        self.assertEqual(result["reason"], "benefit_model_not_measured")
        self.assertEqual(adapter.create_count, 0)

    def test_hard_gate_blocks_non_approaching_target(self):
        model = CalibratedBenefitModel(
            threshold=0.1, bias=0.0, weights={"recognizability_gap": 1.0}, run_id="run"
        )
        adapter, orchestrator = self.make_orchestrator(model)
        result = asyncio.run(orchestrator.request(
            "device", "+905551234567", signals(vehicle_is_approaching=False)
        ))
        self.assertEqual(result["reason"], "target_not_approaching")
        self.assertEqual(adapter.create_count, 0)

    def test_provider_is_reconciled_before_create(self):
        model = CalibratedBenefitModel(
            threshold=0.1, bias=0.0, weights={"recognizability_gap": 1.0}, run_id="run"
        )
        adapter, orchestrator = self.make_orchestrator(model)
        adapter.sessions.append({"sessionId": "provider-existing", "status": "ACTIVE"})
        result = asyncio.run(orchestrator.request("device", "+905551234567", signals()))
        self.assertEqual(result["session_id"], "provider-existing")
        self.assertEqual(adapter.create_count, 0)

    def test_unmeasured_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps({
                "status": "HEDEF",
                "run_id": "x",
                "dataset_manifest_sha256": "y",
                "threshold": 0.5,
                "weights": {},
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                CalibratedBenefitModel.load(path)

    def test_device_cannot_delete_another_session(self):
        model = CalibratedBenefitModel(
            threshold=0.1, bias=0.0, weights={"recognizability_gap": 1.0}, run_id="run"
        )
        adapter, orchestrator = self.make_orchestrator(model)
        asyncio.run(orchestrator.request("device-a", "+905551234567", signals()))
        deleted = asyncio.run(orchestrator.stop("device-b", "session-1"))
        self.assertFalse(deleted)
        self.assertEqual(len(adapter.sessions), 1)

    def test_device_cannot_extend_another_session(self):
        model = CalibratedBenefitModel(
            threshold=0.1, bias=0.0, weights={"recognizability_gap": 1.0}, run_id="run"
        )
        _, orchestrator = self.make_orchestrator(model)
        asyncio.run(orchestrator.request("device-a", "+905551234567", signals()))
        with self.assertRaises(PermissionError):
            asyncio.run(orchestrator.extend("device-b", "session-1", 60))

    def test_provider_failure_falls_back_without_interrupting_inference(self):
        model = CalibratedBenefitModel(
            threshold=0.1, bias=0.0, weights={"recognizability_gap": 1.0}, run_id="run"
        )
        adapter = FailingAdapter()
        orchestrator = QodOrchestrator(
            adapter,
            model,
            flow_descriptor=FlowDescriptor("203.0.113.10", 443, 0, "TCP"),
            qos_profile="provider-profile",
            duration_seconds=300,
        )
        result = asyncio.run(orchestrator.request("device", "+905551234567", signals()))
        self.assertEqual(result["status"], "best_effort")
        self.assertEqual(result["state"], "FAILED")
        self.assertEqual(result["reason"], "provider_unavailable")

    def test_cooldown_blocks_immediate_recreation(self):
        model = CalibratedBenefitModel(
            threshold=0.1, bias=0.0, weights={"recognizability_gap": 1.0}, run_id="run"
        )
        adapter, orchestrator = self.make_orchestrator(model)
        created = asyncio.run(orchestrator.request("device", "+905551234567", signals()))
        asyncio.run(orchestrator.stop("device", created["session_id"]))
        result = asyncio.run(orchestrator.request("device", "+905551234567", signals()))
        self.assertEqual(result["status"], "best_effort")
        self.assertEqual(result["reason"], "cooldown_active")
        self.assertEqual(adapter.create_count, 1)


if __name__ == "__main__":
    unittest.main()
