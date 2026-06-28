import time
from contextlib import contextmanager

from fastapi.testclient import TestClient

from api.webrtc_signaling import AnswerEnvelope, OfferEnvelope
from auth_utils import get_current_device
from server import app, require_media_service


class StubMailbox:
    def __init__(self):
        self.answer = None

    async def create_offer(self, *, device_id, sdp):
        if len(sdp.encode("utf-8")) > 64 * 1024:
            raise ValueError("SDP must be at most 65536 UTF-8 bytes")
        now = time.time()
        return OfferEnvelope("sid", device_id, sdp, now, now + 60)

    async def consume_answer(self, *, signaling_id, device_id):
        return self.answer


class StubRedis:
    async def ping(self):
        return True


@contextmanager
def configured_app(mailbox):
    old_mailbox = getattr(app.state, "signaling", None)
    app.state.signaling = mailbox
    app.dependency_overrides[get_current_device] = lambda: {"device_id": "device-a"}
    try:
        yield TestClient(app)
    finally:
        app.state.signaling = old_mailbox
        app.dependency_overrides.clear()


def test_android_offer_requires_matching_header_body_and_token_device():
    with configured_app(StubMailbox()) as client:
        response = client.post(
            "/webrtc/offer",
            headers={"X-Device-Id": "device-b"},
            json={"device_id": "device-a", "type": "offer", "sdp": "v=0\nm=video"},
        )
    assert response.status_code == 403


def test_signaling_returns_503_when_redis_mailbox_is_unavailable():
    with configured_app(None) as client:
        response = client.post(
            "/webrtc/offer",
            headers={"X-Device-Id": "device-a"},
            json={"device_id": "device-a", "type": "offer", "sdp": "v=0\nm=video"},
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "signaling_unavailable"


def test_sdp_limit_is_enforced_in_utf8_bytes():
    with configured_app(StubMailbox()) as client:
        response = client.post(
            "/webrtc/offer",
            headers={"X-Device-Id": "device-a"},
            json={"device_id": "device-a", "type": "offer", "sdp": "v=0\nm=video " + "ğ" * 40_000},
        )
    assert response.status_code == 422


def test_answer_pending_then_consumed_with_same_event_owner():
    mailbox = StubMailbox()
    with configured_app(mailbox) as client:
        pending = client.get("/webrtc/answer?signaling_id=sid")
        mailbox.answer = AnswerEnvelope("sid", "device-a", "v=0\nm=video", time.time(), time.time() + 30)
        ready = client.get("/webrtc/answer?signaling_id=sid")
    assert pending.status_code == 202
    assert ready.status_code == 200
    assert ready.json()["type"] == "answer"


def test_gpu_offer_claim_requires_service_token():
    with configured_app(StubMailbox()) as client:
        response = client.get("/webrtc/offer")
    assert response.status_code == 401


def test_health_endpoints_are_scoped_to_live_5g():
    with configured_app(StubMailbox()) as client:
        live = client.get("/health/live")
        health = client.get("/health")

    assert live.status_code == 200
    live_payload = live.json()
    assert live_payload["mode"] == "live_5g"
    assert live_payload["status"] == "ok"
    assert health.status_code == 200
    assert health.json()["mode"] == "live_5g"


def test_ready_health_reflects_live_dependencies_and_cached_ice_config():
    with configured_app(StubMailbox()) as client:
        old_redis = getattr(app.state, "redis", None)
        old_configuration_issues = list(getattr(app.state, "configuration_issues", []))
        old_ice_servers = getattr(app.state, "ice_servers", None)
        old_ice_configuration_issue = getattr(app.state, "ice_configuration_issue", None)
        app.state.redis = StubRedis()
        app.state.configuration_issues = []
        app.state.ice_servers = [{"urls": ["stun:stun.example.org"]}]
        app.state.ice_configuration_issue = None

        try:
            ready = client.get("/health/ready")
            config = client.get("/webrtc/config")
        finally:
            app.state.redis = old_redis
            app.state.configuration_issues = old_configuration_issues
            app.state.ice_servers = old_ice_servers
            app.state.ice_configuration_issue = old_ice_configuration_issue

    assert ready.status_code == 200
    assert ready.json()["readiness"] == "ready"
    assert ready.json()["dependencies"]["webrtc_ice_configuration"] == "ready"
    assert config.status_code == 200
    assert config.json() == {"ice_servers": [{"urls": ["stun:stun.example.org"]}]}
