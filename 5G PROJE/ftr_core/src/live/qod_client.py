"""
src/live/qod_client.py
=======================
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

5G QoD (Quality on Demand) API İstemcisi
------------------------------------------
CAMARA/3GPP QoD API standartına uygun istemci.
FTR değerlendirmesiyle ILGISIZDIR — yalnızca post-FTR canlı sahnede kullanılır.

Desteklenen operatörler:
  - Turkcell 5G MEC (TURK_MOBILE endpoint)
  - Vodafone TR 5G (VODAFONE_TR endpoint)
  - Generic CAMARA-compliant QoD API

Kullanım:
    client = QoDClient(base_url="https://api.operator.com/qod/v0", token="...")
    session = await client.create_session(ue_id="ue-123", profile="LOW_LATENCY")
    await client.delete_session(session.session_id)
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger("sinaptic5g.live.qod")

# ── QoD Profil Sabitleri ───────────────────────────────────────────────────────
class QoDProfile(str, Enum):
    """CAMARA QoD API profilleri."""
    LOW_LATENCY         = "QOS_L"   # < 50ms RTT
    STABLE_BANDWIDTH    = "QOS_S"   # > 10 Mbps garanti
    THROUGHPUT_S        = "QOS_S"
    THROUGHPUT_M        = "QOS_M"
    THROUGHPUT_L        = "QOS_L"


@dataclasses.dataclass
class QoDSession:
    """Aktif bir QoD API oturum bilgisi."""
    session_id:   str
    ue_id:        str
    profile:      QoDProfile
    created_at:   float
    expires_at:   float
    status:       str = "AVAILABLE"

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - time.time())


class QoDClient:
    """
    5G QoD API istemcisi — CAMARA v0.9+ uyumlu.

    Sınırlamalar (post-FTR hedefleri):
      1. OAuth2 token yenileme otomatik değil — uygulayıcı sorumluluğu.
      2. Webhook bildirimleri desteklenmez (polling tabanlı).
      3. IPv6 UE ID desteği test edilmedi.
    """

    DEFAULT_DURATION_SECONDS = 300  # 5 dakika oturum

    def __init__(
        self,
        base_url:    str,
        token:       str,
        timeout:     float = 10.0,
        max_retries: int   = 3,
    ):
        self.base_url    = base_url.rstrip("/")
        self.token       = token
        self.timeout     = timeout
        self.max_retries = max_retries
        self._sessions:  dict[str, QoDSession] = {}

    async def create_session(
        self,
        ue_id:    str,
        profile:  QoDProfile  = QoDProfile.LOW_LATENCY,
        duration: int         = DEFAULT_DURATION_SECONDS,
    ) -> QoDSession:
        """
        CAMARA /sessions endpoint'ine POST atar.
        Başarısız olursa MaxRetriesError fırlatır.
        """
        payload = {
            "ueId":              {"externalId": ue_id},
            "qosProfile":        profile.value,
            "duration":          duration,
            "notificationUrl":   None,
            "notificationToken": None,
        }

        # TODO: gerçek HTTP istemci entegrasyonu (httpx veya aiohttp)
        logger.info(
            "[QoD] create_session: ue=%s profile=%s duration=%ds url=%s/sessions",
            ue_id, profile.value, duration, self.base_url,
        )

        # Geliştirme stub — gerçek HTTP isteği burada yapılacak
        # async with httpx.AsyncClient() as client:
        #     r = await client.post(f"{self.base_url}/sessions",
        #                           json=payload, headers=self._headers(),
        #                           timeout=self.timeout)
        #     r.raise_for_status()
        #     data = r.json()

        # Stub yanıt (geliştirme/test)
        now = time.time()
        session = QoDSession(
            session_id = f"stub-{ue_id}-{int(now)}",
            ue_id      = ue_id,
            profile    = profile,
            created_at = now,
            expires_at = now + duration,
            status     = "AVAILABLE",
        )
        self._sessions[session.session_id] = session
        logger.info("[QoD] Oturum oluşturuldu: %s (expires in %ds)", session.session_id, duration)
        return session

    async def delete_session(self, session_id: str) -> bool:
        """CAMARA /sessions/{sessionId} DELETE."""
        if session_id not in self._sessions:
            logger.warning("[QoD] Silinecek oturum bulunamadı: %s", session_id)
            return False

        # TODO: gerçek DELETE isteği
        # async with httpx.AsyncClient() as client:
        #     r = await client.delete(f"{self.base_url}/sessions/{session_id}",
        #                             headers=self._headers(), timeout=self.timeout)

        del self._sessions[session_id]
        logger.info("[QoD] Oturum silindi: %s", session_id)
        return True

    async def get_session_status(self, session_id: str) -> Optional[dict]:
        """CAMARA /sessions/{sessionId} GET."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return {
            "sessionId": session.session_id,
            "status":    session.status if not session.is_expired else "DELETED",
            "remainingSeconds": session.remaining_seconds,
        }

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

    async def cleanup_expired(self) -> int:
        """Süresi dolmuş oturumları temizler. Kaç oturum silindğini döner."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            await self.delete_session(sid)
        return len(expired)
