import asyncio

from api.telemetry_hub import TelemetryHub


class FakeSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, value):
        self.messages.append(value)


def test_reconnect_snapshot_preserves_event_id_and_device_isolation():
    async def scenario():
        hub = TelemetryHub()
        socket_a = FakeSocket()
        socket_b = FakeSocket()
        hub.attach(socket_a, "device-a")
        hub.attach(socket_b, "device-b")
        event = {"type": "analysis_result", "event_id": "event-1", "frame_id": 7}
        await hub.publish("device-a", event)
        assert socket_a.messages == [event]
        assert socket_b.messages == []
        assert hub.snapshot("device-a")["event_id"] == "event-1"

    asyncio.run(scenario())
