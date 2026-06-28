"""
src/live/mec_offload.py
========================
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

MEC (Multi-access Edge Computing) Offload Modülü
--------------------------------------------------
Yoğun çıkarım görevlerini (büyük ONNX modeli, CRNN OCR, temporal LSTM)
5G MEC sunucusuna yükü aktarır. Düşük latency ve yüksek bant genişliği
garantisi için QoDClient ile birlikte çalışır.

Mimari:
  Araç -> 5G Kablosuz -> MEC Düğümü -> Sonuç -> Araç

FTR NOT: Bu modül FTR Docker konteynerinde çalışmaz.
         Yalnızca `live_main.py` tarafından başlatılır.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger("sinaptic5g.live.mec_offload")


class OffloadStrategy(str, Enum):
    """MEC yük aktarma stratejisi."""
    ALWAYS_LOCAL  = "local"       # MEC yok, yerel çıkarım
    ALWAYS_REMOTE = "remote"      # Tüm çıkarımı MEC'e gönder
    ADAPTIVE      = "adaptive"    # Gecikmeye göre karar ver
    HEAVY_MODELS  = "heavy"       # Yalnızca büyük modelleri offload et


@dataclass
class OffloadDecision:
    """Tek bir çıkarım görevi için offload kararı."""
    task_id:        str
    model_name:     str
    strategy:       OffloadStrategy
    use_mec:        bool
    estimated_ms:   float
    reason:         str


class MECOffloadManager:
    """
    Adaptif MEC Offload Yöneticisi.

    Karar mantığı:
      - Yerel RTT < mec_rtt_threshold_ms → yerel çıkarım
      - Yerel RTT >= eşik → MEC offload
      - MEC bağlantı başarısız → yerel fallback

    Desteklenen görevler:
      - `coco_detector`: YOLOv8 COCO araç tespiti
      - `custom_detector`: Davranış/plaka tespiti
      - `crnn_ocr`: Plaka OCR
      - `temporal_lstm`: Davranış sınıflama
    """

    # Offload edilebilir modeller ve tahmini yerel gecikme (ms)
    OFFLOADABLE_MODELS = {
        "coco_detector":    120.0,
        "custom_detector":   80.0,
        "crnn_ocr":          50.0,
        "temporal_lstm":     30.0,
    }

    def __init__(
        self,
        mec_endpoint:          str   = "https://mec.operator.com/inference/v1",
        mec_rtt_threshold_ms:  float = 80.0,
        strategy:              OffloadStrategy = OffloadStrategy.ADAPTIVE,
        max_pending:           int   = 32,
    ):
        self.mec_endpoint           = mec_endpoint
        self.mec_rtt_threshold_ms   = mec_rtt_threshold_ms
        self.strategy               = strategy
        self.max_pending            = max_pending

        self._mec_available:        bool  = False
        self._mec_rtt_ms:           float = float("inf")
        self._pending_tasks:        dict  = {}
        self._stats = {"local": 0, "mec": 0, "fallback": 0}

    async def ping_mec(self) -> float:
        """
        MEC sunucusuna ping atar ve RTT ölçer.
        Başarısız olursa float('inf') döner.
        """
        # TODO: gerçek HTTP ping
        # t = time.perf_counter()
        # async with httpx.AsyncClient() as c:
        #     r = await c.get(f"{self.mec_endpoint}/health", timeout=2.0)
        # return (time.perf_counter() - t) * 1000

        # Stub: test ortamında MEC yok → offline
        logger.debug("[MEC] Ping: stub — MEC mevcut degil")
        self._mec_available = False
        self._mec_rtt_ms    = float("inf")
        return float("inf")

    def decide(self, model_name: str) -> OffloadDecision:
        """
        Belirli bir model için offload kararı verir.
        FTR offline modunda her zaman `use_mec=False` döner.
        """
        local_ms = self.OFFLOADABLE_MODELS.get(model_name, 50.0)

        if self.strategy == OffloadStrategy.ALWAYS_LOCAL or not self._mec_available:
            reason = "ALWAYS_LOCAL" if self.strategy == OffloadStrategy.ALWAYS_LOCAL else "MEC erisilemiyor"
            return OffloadDecision(
                task_id=f"t{int(time.time()*1000)}",
                model_name=model_name,
                strategy=self.strategy,
                use_mec=False,
                estimated_ms=local_ms,
                reason=reason,
            )

        if self.strategy == OffloadStrategy.ALWAYS_REMOTE:
            return OffloadDecision(
                task_id=f"t{int(time.time()*1000)}",
                model_name=model_name,
                strategy=self.strategy,
                use_mec=True,
                estimated_ms=self._mec_rtt_ms,
                reason="ALWAYS_REMOTE",
            )

        # Adaptif karar
        use_mec = self._mec_rtt_ms < local_ms and self._mec_available
        return OffloadDecision(
            task_id=f"t{int(time.time()*1000)}",
            model_name=model_name,
            strategy=self.strategy,
            use_mec=use_mec,
            estimated_ms=self._mec_rtt_ms if use_mec else local_ms,
            reason=f"adaptive: mec={self._mec_rtt_ms:.1f}ms local={local_ms:.1f}ms",
        )

    async def run_inference(
        self,
        model_name: str,
        input_data: Any,
        local_fn:   Any,  # callable
    ) -> Any:
        """
        Offload kararına göre yerel veya MEC üzerinde çıkarım yapar.
        MEC başarısız olursa yerel fallback.
        """
        decision = self.decide(model_name)

        if not decision.use_mec:
            self._stats["local"] += 1
            return await asyncio.get_event_loop().run_in_executor(None, local_fn, input_data)

        # TODO: gerçek MEC HTTP çıkarım isteği
        # try:
        #     async with httpx.AsyncClient() as c:
        #         r = await c.post(f"{self.mec_endpoint}/{model_name}",
        #                          content=serialize(input_data), timeout=5.0)
        #         r.raise_for_status()
        #         self._stats["mec"] += 1
        #         return deserialize(r.content)
        # except Exception as e:
        #     logger.warning("[MEC] Offload basarisiz, fallback: %s", e)

        # Fallback
        self._stats["fallback"] += 1
        return await asyncio.get_event_loop().run_in_executor(None, local_fn, input_data)

    def stats(self) -> dict:
        """Çıkarım istatistiklerini döner."""
        total = sum(self._stats.values()) or 1
        return {
            "local_pct":    round(self._stats["local"]    / total * 100, 1),
            "mec_pct":      round(self._stats["mec"]      / total * 100, 1),
            "fallback_pct": round(self._stats["fallback"] / total * 100, 1),
            "raw":          dict(self._stats),
        }
