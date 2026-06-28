"""
src/live/mobile_sync.py
========================
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

Mobil Cihaz Senkronizasyon Modülü
------------------------------------
Araç içi kabinden gelen FTR tespitlerini mobil uygulamaya
düşük gecikmeli WebSocket veya SSE üzerinden gönderir.

Özellikler (post-FTR hedefleri):
  - JSON payload: results.json formatıyla uyumlu
  - Bant genişliği uyarlamalı gönderim (5G QoD ile entegre)
  - Bağlantı kesildiğinde otomatik yeniden bağlantı
  - FTR offline modunda hiçbir ağ bağlantısı açılmaz (kritik)

Kullanım:
    sync = MobileSync(host="0.0.0.0", port=8765)
    await sync.start()
    await sync.broadcast(detection_result)
    await sync.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

logger = logging.getLogger("sinaptic5g.live.mobile_sync")

# FTR offline modunda ağ tamamen devre dışı
_FTR_OFFLINE_MODE = True  # Docker --network none ile uyumlu


@dataclass
class SyncPayload:
    """Mobil uygulamaya gönderilecek minimal tespit paketi."""
    timestamp:        float
    video_id:         str
    detection_count:  int
    risk_score:       float
    alerts:           list[str]   # FTR whitelist etiketleri
    plate_number:     Optional[str] = None


class MobileSync:
    """
    WebSocket tabanlı gerçek zamanlı senkronizasyon sunucusu.

    FTR NOT: Bu sınıf yalnızca `_FTR_OFFLINE_MODE = False` olduğunda
    ağ bağlantısı açar. Docker --network none ile çalışırken tamamen
    pasif kalır.
    """

    def __init__(
        self,
        host:             str   = "0.0.0.0",
        port:             int   = 8765,
        max_clients:      int   = 10,
        broadcast_hz:     float = 5.0,   # saniyede 5 mesaj
        reconnect_delay:  float = 2.0,
    ):
        self.host            = host
        self.port            = port
        self.max_clients     = max_clients
        self.broadcast_hz    = broadcast_hz
        self.reconnect_delay = reconnect_delay

        self._clients:       set       = set()
        self._pending:       list[dict]= []
        self._running:       bool      = False
        self._server                   = None

    async def start(self) -> None:
        """WebSocket sunucusunu başlatır. FTR offline modunda NOP."""
        if _FTR_OFFLINE_MODE:
            logger.info("[MobileSync] FTR offline modunda -- ag baglantisi acilmadi")
            return

        # TODO: websockets kütüphanesi entegrasyonu
        # self._server = await websockets.serve(
        #     self._handler, self.host, self.port,
        #     max_size=2**20,  # 1MB
        # )
        self._running = True
        logger.info("[MobileSync] Sunucu baslatildi: ws://%s:%d", self.host, self.port)

    async def broadcast(self, payload: SyncPayload) -> int:
        """
        Bağlı tüm istemcilere tespit paketini gönderir.
        FTR offline modunda sadece buffer'a yazar.
        Döndürülen değer: gönderilen istemci sayısı.
        """
        msg = json.dumps({
            "type":      "detection",
            "ts":        payload.timestamp,
            "video_id":  payload.video_id,
            "count":     payload.detection_count,
            "risk":      round(payload.risk_score, 3),
            "alerts":    payload.alerts,
            "plate":     payload.plate_number,
        }, ensure_ascii=True)

        if _FTR_OFFLINE_MODE or not self._clients:
            self._pending.append(json.loads(msg))
            return 0

        # TODO: gerçek WebSocket broadcast
        # dead = set()
        # for ws in self._clients:
        #     try: await ws.send(msg)
        #     except Exception: dead.add(ws)
        # self._clients -= dead

        logger.debug("[MobileSync] Broadcast: %d istemci", len(self._clients))
        return len(self._clients)

    async def stop(self) -> None:
        """Sunucuyu kapatır."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._running = False
        logger.info("[MobileSync] Sunucu durduruldu. Tampon: %d mesaj", len(self._pending))

    def get_pending_buffer(self) -> list[dict]:
        """FTR offline modunda biriken mesajları döner (test için)."""
        buf = list(self._pending)
        self._pending.clear()
        return buf
