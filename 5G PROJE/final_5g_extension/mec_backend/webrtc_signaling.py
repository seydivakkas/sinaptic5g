"""Short-lived SDP mailbox used by the BFF signaling plane.

Only SDP envelopes pass through this component. RTP/RTCP and decoded frames
never enter the BFF or Redis.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Any


MAX_SDP_BYTES = 64 * 1024
DEFAULT_TTL_SECONDS = 60
QUEUE_KEY = "webrtc:offers"


class SignalingExpiredError(LookupError):
    pass


class SignalingConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class OfferEnvelope:
    signaling_id: str
    device_id: str
    sdp: str
    created_at: float
    expires_at: float


@dataclass(frozen=True)
class AnswerEnvelope:
    signaling_id: str
    device_id: str
    sdp: str
    created_at: float
    expires_at: float


def validate_sdp(sdp: str) -> None:
    if not sdp or len(sdp.encode("utf-8")) > MAX_SDP_BYTES:
        raise ValueError(f"SDP must be between 1 and {MAX_SDP_BYTES} UTF-8 bytes")
    if "v=" not in sdp or "m=" not in sdp:
        raise ValueError("SDP lacks mandatory session/media lines")


def _decode(value: Any) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class RedisSignalingMailbox:
    def __init__(self, redis_client: Any, *, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.redis = redis_client
        self.ttl_seconds = int(ttl_seconds)
        if not 1 <= self.ttl_seconds <= 300:
            raise ValueError("signaling TTL must be in [1, 300] seconds")

    @staticmethod
    def _offer_key(signaling_id: str) -> str:
        return f"webrtc:offer:{signaling_id}"

    @staticmethod
    def _answer_key(signaling_id: str) -> str:
        return f"webrtc:answer:{signaling_id}"

    async def create_offer(self, *, device_id: str, sdp: str) -> OfferEnvelope:
        validate_sdp(sdp)
        now = time.time()
        envelope = OfferEnvelope(
            signaling_id=secrets.token_urlsafe(24),
            device_id=device_id,
            sdp=sdp,
            created_at=now,
            expires_at=now + self.ttl_seconds,
        )
        pipeline = self.redis.pipeline(transaction=True)
        pipeline.set(
            self._offer_key(envelope.signaling_id),
            json.dumps(asdict(envelope), separators=(",", ":")),
            ex=self.ttl_seconds,
            nx=True,
        )
        pipeline.lpush(QUEUE_KEY, envelope.signaling_id)
        results = await pipeline.execute()
        if not results or not results[0]:
            raise SignalingConflictError("could not reserve signaling id")
        return envelope

    async def claim_offer(self, *, wait_seconds: int = 0) -> OfferEnvelope | None:
        wait_seconds = max(0, min(int(wait_seconds), 20))
        deadline = time.monotonic() + wait_seconds
        while True:
            remaining = max(0, int(deadline - time.monotonic()))
            if wait_seconds:
                item = await self.redis.brpop(QUEUE_KEY, timeout=max(1, remaining))
                signaling_id = _decode(item[1]) if item else None
            else:
                signaling_id = _decode(await self.redis.rpop(QUEUE_KEY))
            if signaling_id is None:
                return None
            raw = _decode(await self.redis.get(self._offer_key(signaling_id)))
            if raw:
                return OfferEnvelope(**json.loads(raw))
            if not wait_seconds or time.monotonic() >= deadline:
                return None

    async def put_answer(self, *, signaling_id: str, sdp: str) -> AnswerEnvelope:
        validate_sdp(sdp)
        raw_offer = _decode(await self.redis.get(self._offer_key(signaling_id)))
        if not raw_offer:
            raise SignalingExpiredError("offer expired or was already consumed")
        offer = OfferEnvelope(**json.loads(raw_offer))
        ttl = int(await self.redis.ttl(self._offer_key(signaling_id)))
        if ttl <= 0:
            raise SignalingExpiredError("offer expired")
        now = time.time()
        answer = AnswerEnvelope(
            signaling_id=signaling_id,
            device_id=offer.device_id,
            sdp=sdp,
            created_at=now,
            expires_at=min(offer.expires_at, now + ttl),
        )
        stored = await self.redis.set(
            self._answer_key(signaling_id),
            json.dumps(asdict(answer), separators=(",", ":")),
            ex=ttl,
            nx=True,
        )
        if not stored:
            raise SignalingConflictError("answer already exists")
        return answer

    async def consume_answer(self, *, signaling_id: str, device_id: str) -> AnswerEnvelope | None:
        raw_offer = _decode(await self.redis.get(self._offer_key(signaling_id)))
        if not raw_offer:
            raise SignalingExpiredError("offer expired or was already consumed")
        offer = OfferEnvelope(**json.loads(raw_offer))
        if not secrets.compare_digest(offer.device_id, device_id):
            raise PermissionError("signaling session is owned by another device")
        raw_answer = _decode(await self.redis.get(self._answer_key(signaling_id)))
        if not raw_answer:
            return None
        answer = AnswerEnvelope(**json.loads(raw_answer))
        script = (
            "if redis.call('get', KEYS[2]) == ARGV[1] then "
            "redis.call('del', KEYS[1], KEYS[2]); return 1 else return 0 end"
        )
        consumed = await self.redis.eval(
            script,
            2,
            self._offer_key(signaling_id),
            self._answer_key(signaling_id),
            raw_answer,
        )
        if not consumed:
            raise SignalingConflictError("answer was already consumed")
        return answer
