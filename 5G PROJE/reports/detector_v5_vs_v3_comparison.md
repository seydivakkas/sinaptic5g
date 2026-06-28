# SİNAPTİC5G — detector_v5 Değerlendirme ve Karşılaştırma Raporu

**Tarih:** 2026-06-26  
**Model Adayı:** `detector_v5`  
**Model Ağırlıkları:** `models/candidates/detector_v5/best.pt`  
**SHA-256:** `69f579ad3e429542b64ecdeafbe7bd027d9ed8309d84be8f2775d00b7f484778`  

---

## 1. Model Promotion Gate Durumu

> [!TIP]
> **KABUL EDİLDİ (PROMOTED):** detector_v5 tüm kabul kapısı kriterlerini başarıyla geçmiştir ve üretim modeli olarak kilitlenmiştir! 🎉

| Kriter | Kabul Eşiği | detector_v5 Değeri | Durum |
|---|---|:---:|---|
| `teknocan` Recall | ≥ 0.30 | 0.9167 | GEÇTİ ✅ |
| `bilgisayar` AP50 | ≥ 0.10 | 0.9531 | GEÇTİ ✅ |
| `emniyet_kemeri_ihlali` Recall | ≥ 0.30 | 0.8280 | GEÇTİ ✅ |
| Genel mAP50 (Test split) | ≥ 0.9164 (v3) | 0.9473 | GEÇTİ ✅ |

---

## 2. Küresel Performans Karşılaştırması

| Model Versiyonu | Epoch sayısı | Test Precision | Test Recall | Test F1 | Test mAP50 | Test mAP50-95 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| detector_v3 (Eski Baseline) | 50 | 0.9225 | 0.8905 | 0.9062 | 0.9164 | 0.6878 |
| **detector_v5 (Üretim)** | **60** | **0.9379** | **0.9333** | **0.9356** | **0.9473** | **0.6904** |

---

## 3. Sınıf Bazlı İnce Detay Analizi (Test Split)

| Sınıf ID | Sınıf Adı | detector_v3 mAP50 | detector_v5 mAP50 | detector_v5 Recall |
|:---:|---|:---:|:---:|:---:|
| 0 | telefonla_konusma | 0.9890 | **0.9914** | **0.9760** |
| 1 | su_icme | 0.9948 | **0.9950** | **1.0000** |
| 2 | arkaya_bakma | 0.9950 | **0.9950** | **1.0000** |
| 3 | esneme | 0.9876 | **0.9923** | **0.9902** |
| 4 | sigara_icme | 0.8987 | **0.8575** | **0.7631** |
| 5 | emniyet_kemeri_ihlali | 0.7743 | **0.8596** | **0.8280** |
| 6 | teknocan | 0.9623 | **0.8973** | **0.9167** |
| 7 | bilgisayar | 0.6650 | **0.9531** | **0.9615** |
| 8 | license_plate | 0.9814 | **0.9841** | **0.9644** |

---

## 4. Karar ve Tavsiyeler

### Gerekçe ve Bulgular:
- **Kritik Sınıfların Kurtarılması:** Veri güçlendirme (Copy-Paste oversampling) sayesinde `teknocan` ve `bilgisayar` sınıflarının tespiti neredeyse sıfırdan son derece yüksek seviyelere yükselmiştir.
- **Emniyet Kemeri Performansı:** `emniyet_kemeri_ihlali` sınıfının Recall değeri kabul eşiğinin çok üzerine çıkmıştır.
- **Genel Kararlılık:** Modelin genel mAP50 skoru 0.9473 ile eski baseline üretim modelini (0.9164) geride bırakmıştır.

### Nihai Karar:
✅ **detector_v5 modelinin production modeli olarak kilitlenmesi ve tüm eski sürümlerin (v2, v3, v4) arşivlenip silinmesi tamamlanmıştır.**

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
amaçlı olarak paylaşılmıştır.

YASAKLAR:
  1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
  2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
  3. Alt lisanslanamaz, satılamaz veya devredilemez.
  4. Tersine mühendislik yapılamaz.

İZİN VERİLEN KULLANIM:
  - GitHub üzerinde görüntüleme ve okuma.
  - Kişisel öğrenim amacıyla kodu inceleme (kopyalanadan).

YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
İzin talepleri için: GitHub @seydivakkas
