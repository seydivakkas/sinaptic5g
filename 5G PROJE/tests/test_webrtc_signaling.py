from __future__ import annotations

import asyncio
import unittest

from api.webrtc_signaling import (
    RedisSignalingMailbox,
    SignalingConflictError,
    SignalingExpiredError,
    validate_sdp,
)


OFFER = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\n"
ANSWER = "v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=recvonly\r\n"


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    def set(self, *args, **kwargs):
        self.commands.append(("set", args, kwargs))
        return self

    def lpush(self, *args, **kwargs):
        self.commands.append(("lpush", args, kwargs))
        return self

    async def execute(self):
        results = []
        for name, args, kwargs in self.commands:
            results.append(await getattr(self.redis, name)(*args, **kwargs))
        return results


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.queues = {}
        self.ttls = {}

    def pipeline(self, transaction=True):
        return FakePipeline(self)

    async def set(self, key, value, ex=None, nx=False):
        if nx and key in self.values:
            return False
        self.values[key] = value
        self.ttls[key] = ex or -1
        return True

    async def get(self, key):
        return self.values.get(key)

    async def ttl(self, key):
        return self.ttls.get(key, -2)

    async def lpush(self, key, value):
        self.queues.setdefault(key, []).insert(0, value)
        return len(self.queues[key])

    async def rpop(self, key):
        queue = self.queues.get(key, [])
        return queue.pop() if queue else None

    async def brpop(self, key, timeout=0):
        value = await self.rpop(key)
        return (key, value) if value is not None else None

    async def eval(self, script, count, offer_key, answer_key, expected_answer):
        if self.values.get(answer_key) != expected_answer:
            return 0
        self.values.pop(offer_key, None)
        self.values.pop(answer_key, None)
        self.ttls.pop(offer_key, None)
        self.ttls.pop(answer_key, None)
        return 1


class WebRtcSignalingTests(unittest.TestCase):
    def test_offer_answer_is_single_use(self):
        async def scenario():
            mailbox = RedisSignalingMailbox(FakeRedis(), ttl_seconds=60)
            created = await mailbox.create_offer(device_id="device-a", sdp=OFFER)
            claimed = await mailbox.claim_offer()
            self.assertEqual(claimed.signaling_id, created.signaling_id)
            await mailbox.put_answer(signaling_id=created.signaling_id, sdp=ANSWER)
            consumed = await mailbox.consume_answer(
                signaling_id=created.signaling_id, device_id="device-a"
            )
            self.assertEqual(consumed.sdp, ANSWER)
            with self.assertRaises(SignalingExpiredError):
                await mailbox.consume_answer(
                    signaling_id=created.signaling_id, device_id="device-a"
                )

        asyncio.run(scenario())

    def test_pending_answer_returns_none(self):
        async def scenario():
            mailbox = RedisSignalingMailbox(FakeRedis())
            created = await mailbox.create_offer(device_id="device-a", sdp=OFFER)
            pending = await mailbox.consume_answer(
                signaling_id=created.signaling_id, device_id="device-a"
            )
            self.assertIsNone(pending)

        asyncio.run(scenario())

    def test_other_device_cannot_consume_answer(self):
        async def scenario():
            mailbox = RedisSignalingMailbox(FakeRedis())
            created = await mailbox.create_offer(device_id="device-a", sdp=OFFER)
            await mailbox.put_answer(signaling_id=created.signaling_id, sdp=ANSWER)
            with self.assertRaises(PermissionError):
                await mailbox.consume_answer(
                    signaling_id=created.signaling_id, device_id="device-b"
                )

        asyncio.run(scenario())

    def test_duplicate_answer_is_rejected(self):
        async def scenario():
            mailbox = RedisSignalingMailbox(FakeRedis())
            created = await mailbox.create_offer(device_id="device-a", sdp=OFFER)
            await mailbox.put_answer(signaling_id=created.signaling_id, sdp=ANSWER)
            with self.assertRaises(SignalingConflictError):
                await mailbox.put_answer(signaling_id=created.signaling_id, sdp=ANSWER)

        asyncio.run(scenario())

    def test_sdp_validation(self):
        with self.assertRaises(ValueError):
            validate_sdp("not-sdp")
        with self.assertRaises(ValueError):
            validate_sdp("v=0\r\nm=video " + "x" * (64 * 1024))


if __name__ == "__main__":
    unittest.main()
