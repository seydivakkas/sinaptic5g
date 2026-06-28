from fastapi.routing import APIWebSocketRoute

from server import app
from media_service import app as media_app


def test_bff_has_sdp_routes_but_no_media_or_telemetry_route():
    paths = {route.path for route in app.routes}
    assert "/webrtc/offer" in paths
    assert "/webrtc/answer" in paths
    assert "/ws/telemetry" not in paths
    assert not any(isinstance(route, APIWebSocketRoute) for route in app.routes)
    assert paths.isdisjoint({"/frame", "/frames", "/media", "/ws/analyze"})


def test_gpu_service_exposes_telemetry_but_no_frame_ingest_route():
    paths = {route.path for route in media_app.routes}
    assert "/ws/telemetry" in paths
    assert paths.isdisjoint({"/frame", "/frames", "/media", "/upload", "/ws/analyze"})
