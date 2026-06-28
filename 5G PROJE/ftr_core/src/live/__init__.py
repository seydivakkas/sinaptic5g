"""
src/live/__init__.py
=====================
Sinaptic5G — Canli 5G/MEC Genisleme Modulleri

Bu paket, FTR offline cekirdeginin ustune eklenen canli yayin,
5G QoD API entegrasyonu ve MEC offload yeteneklerini icerir.

NOT: Bu modüller FTR değerlendirmesiyle ILGISIZDIR.
     Yalnizca yarisma sonrasi / canli sahne modunda aktiftir.
"""
__version__ = "0.1.0-alpha"
__all__ = ["qod_client", "webrtc_streamer", "mobile_sync", "mec_offload"]
