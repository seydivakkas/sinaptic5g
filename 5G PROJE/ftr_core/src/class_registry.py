# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
#
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
#
# Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
# amaçlı olarak paylaşılmıştır.
#
# YASAKLAR:
#   1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
#   2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
#   3. Alt lisanslanamaz, satılamaz veya devredilemez.
#   4. Tersine mühendislik yapılamaz.
#
# İZİN VERİLEN KULLANIM:
#   - GitHub üzerinde görüntüleme ve okuma.
#   - Kişisel öğrenim amacıyla kodu inceleme (kopyalamadan).
#
# YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
# İzin talepleri için: GitHub @seydivakkas

"""Merkezi sınıf kaydı (Class Registry).

Proje genelinde kullanılan tüm model sınıf tanımlarının tek yetkili kaynağı.

İki farklı model bağlamı vardır:

1. **Canlı Pipeline (LIVE_CLASS_MAP)** — 6 sınıf
   - ``ai_pipeline.py`` ve ``data.yaml`` tarafından kullanılır.
   - Canlı GPU model (YOLOv8n_tripwire) eğitim/inference sınıfları.

2. **FTR Pipeline (FTR_CLASS_MAP)** — 9 sınıf
   - ``ftr_main.py OnnxCustomDetector`` tarafından kullanılır.
   - Çevrimdışı yarışma modeli (YOLOv8m özel) sınıfları.
   - competition_contract.py ile eşlenen Türkçe kanonik etiketler.

Bu iki harita bilinçli olarak farklıdır; farklı modeller farklı etiket
sözleşmeleri kullanır. Uyumsuzluk bu modül aracılığıyla belgelenir ve
kontrol altında tutulur.
"""

from __future__ import annotations

from typing import Dict


# ══════════════════════════════════════════════════════════════════════
# CANLI PİPELİNE — 6 sınıf (İngilizce etiketler)
# Kullanım: ai_pipeline.py, data.yaml, config.py ModelConfig
# Model: YOLOv8n_tripwire.pt / .onnx
# ══════════════════════════════════════════════════════════════════════

LIVE_CLASS_MAP: Dict[int, str] = {
    0: "license_plate",
    1: "phone",
    2: "cigarette",
    3: "toy",
    4: "togg",
    5: "driver_face",
}

LIVE_CLASS_NAMES: list[str] = [LIVE_CLASS_MAP[i] for i in range(len(LIVE_CLASS_MAP))]

# İngilizce ad → sınıf ID eşlemesi (ai_pipeline.py uyumu)
LIVE_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in LIVE_CLASS_MAP.items()}

# Dikkat dağıtıcı nesneler (canlı GPU görev yönlendirmesi)
LIVE_DISTRACTOR_CLASSES = {"phone", "cigarette", "toy"}

# Renk haritası (BGR formatı — OpenCV uyumlu)
LIVE_CLASS_COLORS: Dict[str, tuple[int, int, int]] = {
    "license_plate": (0, 255, 0),     # Yeşil
    "phone":         (0, 0, 255),     # Kırmızı
    "cigarette":     (0, 165, 255),   # Turuncu
    "toy":           (255, 0, 255),   # Magenta
    "togg":          (255, 255, 0),   # Cyan
    "driver_face":   (255, 255, 255), # Beyaz
}


# ══════════════════════════════════════════════════════════════════════
# FTR PİPELİNE — 9 sınıf (Türkçe kanonik etiketler)
# Kullanım: ftr_main.py OnnxCustomDetector
# Model: detector.onnx / detector_optimized.onnx (YOLOv8m özel)
# ══════════════════════════════════════════════════════════════════════

FTR_CLASS_MAP: Dict[int, str] = {
    0: "telefonla_konusma",
    1: "su_icme",
    2: "arkaya_bakma",
    3: "esneme",
    4: "sigara_icme",
    5: "emniyet_kemeri_ihlali",
    6: "teknocan",
    7: "bilgisayar",
    8: "license_plate",
}

FTR_NUM_CLASSES: int = len(FTR_CLASS_MAP)

FTR_CLASS_NAMES: list[str] = [FTR_CLASS_MAP[i] for i in range(FTR_NUM_CLASSES)]

# Türkçe ad → sınıf ID eşlemesi
FTR_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in FTR_CLASS_MAP.items()}


# ══════════════════════════════════════════════════════════════════════
# SINIF-SPESİFİK GÜVEN EŞİKLERİ (FTR Pipeline)
# Threshold kalibrasyon çalışmasından türetilmiştir.
# Rapor: reports/threshold_calibration_study.csv
# ══════════════════════════════════════════════════════════════════════

FTR_CLASS_THRESHOLDS: Dict[str, float] = {
    "telefonla_konusma":      0.30,  # Recall artırmak için düşük tutuldu
    "su_icme":                0.40,  # Dengeli precision/recall
    "arkaya_bakma":           0.25,  # Düşük precision → düşük eşikle daha fazla yakalama
    "esneme":                 0.25,  # Düşük precision → düşük eşikle daha fazla yakalama
    "sigara_icme":            0.45,  # Yüksek FP → yüksek eşik ile filtreleme
    "emniyet_kemeri_ihlali":  0.30,  # Güvenlik-kritik → düşük eşik
    "teknocan":               0.35,  # Varsayılan (düşük destek)
    "bilgisayar":             0.30,  # Düşük destek → düşük eşik
    "license_plate":          0.40,  # İyi performans → orta eşik
}

# Varsayılan eşik (yeni/bilinmeyen sınıflar için)
FTR_DEFAULT_THRESHOLD: float = 0.35


# ══════════════════════════════════════════════════════════════════════
# ZERO-DCE KOŞULLU ÇALIŞTIRMA PARAMETRELERİ
# ══════════════════════════════════════════════════════════════════════

ZERO_DCE_BRIGHTNESS_THRESHOLD: int = 80  # Ortalama parlaklık bu değerin altındaysa Zero-DCE çalışır


# ══════════════════════════════════════════════════════════════════════
# CANLI ↔ FTR EŞLEMESİ
# Canlı pipeline İngilizce etiketlerinin FTR Türkçe karşılıkları.
# competition_contract.py LABEL_ALIASES ile tutarlıdır.
# ══════════════════════════════════════════════════════════════════════

LIVE_TO_FTR_LABEL: Dict[str, str] = {
    "phone":         "telefonla_konusma",
    "cigarette":     "sigara_icme",
    "toy":           "teknocan",
    "license_plate": "license_plate",
    # driver_face ve togg FTR etiketlerinde doğrudan karşılıksızdır
}
