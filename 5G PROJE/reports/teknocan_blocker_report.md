# Teknocan Sınıfı Blok Durumu Raporu (Teknocan Blocker Report)

> **Durum:** `UYARI` (Sentetik Üretim Engellendi)  
> **Tarih:** 2026-06-21  
> **Modül:** `scripts/generate_teknocan_synthetic.py`  

## Mevcut Durum Analizi

`detector_v4` korpusunda yarışma şartnamesindeki `teknocan` (sınıf 6) için sentetik veri üretim hattı (`generate_teknocan_synthetic.py`) hazır durumdadır. Ancak, üretim için gerekli olan gerçek/onaylı ön plan (foreground) maskot görselleri (`data/raw/teknocan_fg/`) dizininde mevcut değildir.

* `data/raw/teknocan_fg/` dizini tamamen boştur.
* `data/quarantine/teknocan_softoy/` dizininde bulunan Roboflow kaynaklı soft_toy veri kümesi öğeleri, şeffaf ön plana sahip maskot görselleri olmayıp arka planı bulunan standart JPEG nesne görüntüleridir. Bunların doğrudan kopyala-yapıştır yöntemiyle sentezlenmesi semantik kayma ve gerçekçi olmayan yapıştırma sınırları üreteceğinden sentetik üretim hattı güvenli şekilde bloke edilmiştir.
* Yapay veya sahte maskot görselleri üretilmemiş, test verisi kirletilmemiştir.

## Ekip İçin Aksiyon Listesi (Checklist)

Sentetik üretim hattının çalışabilmesi ve Teknocan sınıfının model tarafından başarıyla öğrenilebilmesi için yerel ekiplerce/tasarımcılarca aşağıdaki varlıkların sağlanması gerekmektedir:

- [ ] `data/raw/teknocan_fg/` dizinine en az 10 adet onaylı Teknocan maskot görseli yerleştirilmelidir.
- [ ] Görsellerin arka planı şeffaf (alfa kanallı PNG formatında) olmalıdır.
- [ ] Modelin genelleştirme yeteneği kazanabilmesi için farklı açılardan (poses), farklı boyutlarda (scales) ve farklı ışıklandırma koşullarında (lighting conditions) görseller seçilmelidir.
- [ ] Sağlanan varlıkların kaynak, lisans ve kullanım onayı belgelenmelidir.

---

ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR  
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
