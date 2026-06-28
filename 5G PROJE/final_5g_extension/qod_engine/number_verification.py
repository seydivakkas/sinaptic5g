"""
api/number_verification.py — Sinaptic5G Number Verification API İstemcisi
=========================================================================
GSMA Open Gateway / Turkcell Number Verification API entegrasyonu.

Bu API, kullanıcının SIM kartını SMS OTP gerektirmeksizin sessizce doğrular.
Mobil veri bağlantısı (5G/4G) üzerinden ağ operatörü SIM eşleşmesini yapar.

⚠️ ÖNEMLİ: WiFi üzerinden çağrı yapıldığında HTTP 422 döner.
            Yalnızca mobil veri bağlantısında çalışır.

Referanslar:
    - https://developer.turkcell.com.tr
    - https://github.com/camaraproject/NumberVerification
"""

import re
import uuid
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

E164_RE = re.compile(r"^\+[1-9][0-9]{7,14}$")


class NumberVerificationClient:
    """
    Turkcell Open Gateway — Number Verification API istemcisi.

    Doğrulama akışı:
    1. Android, OIDC Authorization Code + PKCE ile kullanıcı tokenı alır.
    2. Bu istemci token ile POST /verify çağrısını yapar.
    3. Yanıt: {"devicePhoneNumberVerified": true/false}

    HTTP durum kodları:
    - 200 → SIM eşleşme sonucu (verified: true/false)
    - 403 → SIM eşleşmedi (farklı numara)
    - 422 → WiFi üzerinden çağrı (mobil veri gerekli)
    - 5xx → API hatası

    Gecikme ve başarı oranı ancak ham test kaydıyla raporlanır.
    """

    def __init__(
        self,
        verify_url: Optional[str] = None,
    ):
        import os
        self._verify_url = verify_url or os.getenv(
            "TURKCELL_NUMBER_VERIFY_URL",
            ""
        )

    async def verify(
        self,
        phone_number: str,
        access_token: str,
        timeout: float = 5.0,
    ) -> bool:
        """
        Telefon numarasını SIM eşleşmesiyle sessizce doğrular.

        Parametreler:
            phone_number: E.164 formatında numara (+905xxxxxxxxx)
            access_token: Android OIDC Authorization Code + PKCE akışından
                          alınan tek kullanımlık, kısa ömürlü kullanıcı tokenı.
            timeout:      İstek zaman aşımı (saniye)

        Dönüş:
            True:  SIM eşleşti → numara doğrulandı
            False: SIM eşleşmedi veya hata

        ⚠️ Uyarı:
            Bu API mobil veri bağlantısı (5G/4G) üzerinden çağrılmalıdır.
            WiFi bağlantısında çalışmaz (HTTP 422 döner).
        """
        if not access_token:
            raise ValueError("Number Verification requires a user-bound access token")
        if not E164_RE.fullmatch(phone_number):
            raise ValueError("phone number must be normalized E.164")
        if not self._verify_url:
            raise RuntimeError("Number Verification provider URL is not configured")
        token = access_token

        headers = {
            "Authorization":  f"Bearer {token}",
            "Content-Type":   "application/json",
            "X-Correlator":   str(uuid.uuid4()),  # İstek takibi için
        }

        payload = {"phoneNumber": phone_number}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self._verify_url,
                    headers=headers,
                    json=payload,
                )

            return self._handle_response(response, phone_number)

        except httpx.TimeoutException:
            logger.error(
                f"Number Verification: Zaman aşımı ({timeout}s) — "
                f"{phone_number[:6]}***"
            )
            return False

        except httpx.ConnectError as e:
            logger.error("Number Verification connection failed: %s", type(e).__name__)
            return False

        except Exception as e:
            logger.error("Number Verification unexpected failure: %s", type(e).__name__)
            return False

    def _handle_response(self, response: httpx.Response, phone_number: str) -> bool:
        """HTTP yanıtını işler ve doğrulama sonucunu döner."""
        masked_number = phone_number[:4] + "***" + phone_number[-4:]

        if response.status_code == 200:
            data     = response.json()
            verified = data.get("devicePhoneNumberVerified", False)
            if verified:
                logger.info(f"✅ Number Verification: {masked_number} doğrulandı")
            else:
                logger.info(f"❌ Number Verification: {masked_number} doğrulanamadı")
            return bool(verified)

        elif response.status_code == 403:
            logger.warning(
                f"Number Verification: SIM eşleşmedi ({masked_number}) — HTTP 403"
            )
            return False

        elif response.status_code == 422:
            logger.warning(
                "Number Verification: WiFi üzerinden çağrı yapıldı. "
                "Mobil veri bağlantısı (5G/4G) gereklidir. — HTTP 422"
            )
            return False

        elif response.status_code == 401:
            logger.error(
                "Number Verification: Geçersiz token — HTTP 401. "
                "Token yenileniyor..."
            )
            return False

        else:
            logger.error(
                f"Number Verification: HTTP {response.status_code} — "
                f"{response.text[:200]}"
            )
            return False
