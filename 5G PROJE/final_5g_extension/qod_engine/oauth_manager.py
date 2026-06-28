"""
api/oauth_manager.py — Sinaptic5G OAuth 2.0 Token Yönetimi
===========================================================
Turkcell Open Gateway için OAuth 2.0 Client Credentials akışı.

Özellikler:
- Kapsam (scope) bazlı token önbellekleme
- Thread-safe asyncio.Lock kullanımı
- Sona ermeden 5 dakika önce otomatik yenileme
- Graceful hata yönetimi
"""

import os
import time
import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

TOKEN_URL     = os.getenv("TURKCELL_TOKEN_URL", "")
CLIENT_ID     = os.getenv("TURKCELL_QOD_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("TURKCELL_QOD_CLIENT_SECRET", "")


class OAuthManager:
    """
    OAuth 2.0 Client Credentials akışı ile token yönetimi.

    Token önbellekleme stratejisi:
    - Her scope için ayrı token saklanır
    - SAFETY_MARGIN = 300s (5 dakika) — sona ermeden önce yenilenir
    - asyncio.Lock ile eş zamanlı isteklerde çift token sorunu önlenir

    Token dict formatı:
        {
          "qod:sessions:write": {"token": "Bearer ...", "expires_at": 1745000000},
        }
    """

    SAFETY_MARGIN = 300  # 5 dakika

    def __init__(
        self,
        client_id: str     = None,
        client_secret: str = None,
        token_url: str     = None,
    ):
        self._client_id     = client_id or CLIENT_ID
        self._client_secret = client_secret or CLIENT_SECRET
        self._token_url     = token_url or TOKEN_URL

        self._tokens: dict[str, dict] = {}  # scope → {token, expires_at}
        self._lock = asyncio.Lock()

    async def get_token(self, scope: str) -> str:
        """
        Belirtilen scope için geçerli bir access token döner.

        Önbellekte geçerli token varsa önbellekten döner.
        Yoksa veya süresi dolmuşsa yeni token alır.

        Parametreler:
            scope: OAuth kapsamı (örn: "qod:sessions:write")

        Dönüş:
            access_token string'i

        Raises:
            Exception: Token alınamadıysa
        """
        async with self._lock:
            cached = self._tokens.get(scope)

            # Önbellekte geçerli token var mı?
            if (
                cached
                and time.time() < cached["expires_at"] - self.SAFETY_MARGIN
            ):
                logger.debug(f"Önbellekten token döndü: scope={scope}")
                return cached["token"]

            # Yeni token al
            logger.info(f"Yeni token alınıyor: scope={scope}")
            token_data = await self._fetch_token(scope)
            self._tokens[scope] = token_data
            return token_data["token"]

    async def _fetch_token(self, scope: str) -> dict:
        """
        Turkcell OAuth sunucusundan yeni token alır.

        İstek formatı:
            POST /oauth/token
            Content-Type: application/x-www-form-urlencoded
            Authorization: Basic base64(client_id:client_secret)
            Body: grant_type=client_credentials&scope=<scope>

        Dönüş dict formatı:
            {"token": "abc...", "expires_at": 1745000000.0}
        """
        if not self._client_id or not self._client_secret or not self._token_url:
            raise ValueError(
                "TURKCELL token URL ile QoD client kimlik bilgileri "
                "sunucu ortamında ayarlanmalıdır.\n"
                "Sağlayıcı onboarding kimlik bilgilerini sunucu ortamında ayarlayın."
            )

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    self._token_url,
                    data={
                        "grant_type": "client_credentials",
                        "scope": scope,
                    },
                    auth=(self._client_id, self._client_secret),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"OAuth sunucusuna bağlanılamadı: {self._token_url}\n"
                    "TURKCELL_TOKEN_URL ve sağlayıcı kimlik bilgilerini doğrulayın."
                ) from e

            if response.status_code != 200:
                raise Exception(
                    f"OAuth token alınamadı: HTTP {response.status_code} — "
                    f"{response.text}"
                )

            data = response.json()
            expires_in = data.get("expires_in", 3600)

            logger.debug(
                f"OAuth token alındı: scope={scope}, expires_in={expires_in}s"
            )

            return {
                "token":      data["access_token"],
                "expires_at": time.time() + expires_in,
            }

    def invalidate(self, scope: Optional[str] = None):
        """
        Token önbelleğini temizler.
        Test ortamı veya hata kurtarma için kullanılır.

        Parametreler:
            scope: Belirtilirse yalnızca o scope temizlenir.
                   None verilirse tüm önbellek temizlenir.
        """
        if scope:
            self._tokens.pop(scope, None)
            logger.debug(f"Token önbelleği temizlendi: scope={scope}")
        else:
            self._tokens.clear()
            logger.debug("Tüm token önbelleği temizlendi")

    @property
    def cached_scopes(self) -> list[str]:
        """Önbellekteki aktif scope listesi."""
        now = time.time()
        return [
            scope for scope, data in self._tokens.items()
            if time.time() < data["expires_at"] - self.SAFETY_MARGIN
        ]


if __name__ == "__main__":
    import asyncio

    async def test():
        manager = OAuthManager()
        print("OAuthManager oluşturuldu")
        print(f"Token URL: {manager._token_url}")
        print(f"Client ID: {'ayarlı' if manager._client_id else 'EKSİK'}")

    asyncio.run(test())
