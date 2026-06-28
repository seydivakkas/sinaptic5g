import asyncio

from api.latest_frame import LatestFrameQueue


def test_slow_consumer_keeps_only_latest_frame():
    async def scenario():
        queue = LatestFrameQueue[int]()
        for frame_id in range(100):
            queue.put_latest(frame_id)
            assert queue.size <= 1
        assert await queue.get() == 99
        assert queue.dropped_count == 99

    asyncio.run(scenario())

