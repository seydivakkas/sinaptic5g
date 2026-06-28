"""A bounded latest-frame handoff for real-time inference."""

from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestFrameQueue(Generic[T]):
    """Capacity-one queue that evicts stale work instead of growing latency."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=1)
        self.dropped_count = 0

    @property
    def size(self) -> int:
        return self._queue.qsize()

    def put_latest(self, item: T) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.dropped_count += 1
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(item)

    async def get(self) -> T:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

