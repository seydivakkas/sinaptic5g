"""Sinaptic5G — Modüler Kaynak Kodu Paketi.

Alt paketler:
    src.models  — YOLOv8, Zero-DCE, Temporal modeller
    src.data    — Dataset, augmentation, dataloader
"""

from src import models, data

__all__ = ["models", "data"]
