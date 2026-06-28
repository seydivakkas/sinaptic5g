# SİNAPTİC5G — Uygulama / Implementasyon Planı

**Amaç:** Mevcut geliştirme önerilerini uygulanabilir bir sıraya koymak; hangi işin önce yapılacağını, hangi dosyalara dokunulacağını, hangi kanıtların üretileceğini ve hangi kabul kriterleri sağlanmadan sonraki faza geçilmeyeceğini netleştirmek.

**Ana ilke:** Üretim tesliminde `detector_v3` kilidi korunur. `detector_v4_full`, YOLOv8l/YOLOv11, gelişmiş OCR ve temporal modeller **aday/deneysel kanal** olarak yürütülür. Aday model yalnızca ölçümlerle `detector_v3` performansını ve teslim kısıtlarını geçtiğinde üretime alınır.

---

## 0. Uygulama Mantığı

Bu plan dört paralel kulvara ayrılır:

| Kulvar | Hedef | Neden önemli? |
|---|---|---|
| A — Çekirdek tutarlılık | Sınıf isimleri, split oranları, pipeline davranışı ve bellek yönetimini tek standarda bağlamak | Farklı dosyalardaki uyumsuzluklar model doğru olsa bile teslim çıktısını bozabilir |
| B — Veri kalitesi | Az temsil edilen sınıfları güçlendirmek, etiketi temizlemek, augmentasyonu iyileştirmek | Teknocan, bilgisayar ve emniyet kemeri sınıflarında asıl sorun veri desteği |
| C — Model ve OCR | `detector_v4_full`, OCR ve temporal modeli ölçülebilir şekilde geliştirmek | Yeni modelin gerçekten iyi olup olmadığını kanıtlamak gerekir |
| D — Test, kanıt ve teslim | Her değişikliği raporlanabilir kanıta dönüştürmek | Yarışma raporunda “yapıldı” değil “ölçüldü ve kanıtlandı” denmelidir |

---

## 1. Ön Kabul Kuralları

Uygulamaya başlamadan önce aşağıdaki kurallar proje standardı kabul edilmelidir:

1. **Production model kilidi:** `models/detector.onnx` mevcut `detector_v3` kilidi olarak kalır.
2. **Aday modeller ayrı dizinde tutulur:** `models/candidates/detector_v4_full/`, `models/candidates/yolov8l/`, `models/candidates/yolov11m/`.
3. **Tek kaynak sınıf sistemi:** Sınıf isimleri, indeksleri ve yarışma çıktısı tek bir `src/class_registry.py` dosyasından okunur.
4. **Tek kaynak veri split oranı:** `config.py`, `dataset_split.py`, `data.yaml` ve rapor aynı oranı kullanır.
5. **Her faz sonunda kanıt dosyası üretilir:** `.json`, `.csv`, `.md`, ekran görüntüsü, confusion matrix, latency benchmark veya test raporu.
6. **Model değiştirme kapısı:** Aday model, `detector_v3` değerlerini geçmeden üretime alınmaz.

---

## 2. Faz 0 — Repo Sabitleme ve Güvenli Başlangıç

**Süre:** 0.5 gün  
**Amaç:** Değişikliklere başlamadan önce mevcut çalışan sistemi bozmamak.

| Adım | İş | Dosya / Dizin | Çıktı | Kabul kriteri |
|---|---|---|---|---|
| 0.1 | Yeni branch aç | Git | `feature/final-implementation-plan` | Ana branch bozulmadan çalışma başlar |
| 0.2 | Mevcut model hash doğrula | `models/detector.onnx` | `reports/model_lock_check.md` | Hash eski kabul edilen hash ile aynı |
| 0.3 | Mevcut testleri çalıştır | `tests/` | `reports/baseline_pytest.txt` | Tüm testler geçer veya mevcut hatalar kayıt altına alınır |
| 0.4 | Mevcut E2E benchmark al | `ftr_main.py`, örnek video | `reports/baseline_e2e_latency.json` | Teslim bütçesine göre başlangıç noktası netleşir |
| 0.5 | Kanıt klasör yapısını oluştur | `reports/final_evidence/` | Standart klasörler | Sonraki fazların çıktısı kaybolmaz |

**Komut iskeleti:**

```bash
git checkout -b feature/final-implementation-plan
python scripts/verify_model_lock.py --model models/detector.onnx --out reports/model_lock_check.md
pytest -q | tee reports/baseline_pytest.txt
python scripts/benchmark_e2e.py --input samples/test_video.mp4 --out reports/baseline_e2e_latency.json
mkdir -p reports/final_evidence/{data,model,pipeline,ocr,tracking,5g,tests,figures}
```

---

## 3. Faz 1 — Acil Çekirdek Düzeltmeler

**Süre:** 1–3 gün  
**Hedef:** Teslim çıktısını bozabilecek kritik tutarsızlıkları kapatmak.

### 3.1. İş Sırası

| Öncelik | Görev | Dokunulacak dosyalar | Beklenen çıktı | Kabul kriteri |
|---|---|---|---|---|
| P0 | `CLASS_MAP` tutarsızlığını gider | `src/class_registry.py`, `ai_pipeline.py`, `competition_contract.py`, `data.yaml` | Tek sınıf kaydı | 9 resmi sınıf tek indeks düzeniyle her yerde aynı |
| P0 | Split oranını standardize et | `config.py`, `dataset_split.py`, `data.yaml`, rapor | `reports/split_consistency_report.md` | Train/Val/Test oranı ve seed tek kaynakta |
| P1 | Bellek cleanup ekle | `ftr_main.py`, tracker yardımcıları | Track buffer cleanup | Uzun video testinde bellek büyümesi sınırlı |
| P1 | Sınıf bazlı confidence threshold uygula | `config.py`, `ftr_main.py` | `CLASS_THRESHOLDS` | Threshold test raporu üretilir |
| P2 | Zero-DCE koşullu çalıştır | `ftr_main.py`, `enhancement.py` | Parlaklık eşikli enhancer | Aydınlık karelerde gereksiz işlem yapılmaz |

### 3.2. Önerilen `class_registry.py` İskeleti

```python
from enum import IntEnum

class DriverBehavior(IntEnum):
    TELEFONLA_KONUSMA = 0
    SU_ICME = 1
    ARKAYA_BAKMA = 2
    ESNEME = 3
    SIGARA_ICME = 4
    EMNIYET_KEMERI_IHLALI = 5
    TEKNOCAN = 6
    BILGISAYAR = 7
    LICENSE_PLATE = 8

CLASS_NAMES_TR = {
    DriverBehavior.TELEFONLA_KONUSMA: "telefonla_konusma",
    DriverBehavior.SU_ICME: "su_icme",
    DriverBehavior.ARKAYA_BAKMA: "arkaya_bakma",
    DriverBehavior.ESNEME: "esneme",
    DriverBehavior.SIGARA_ICME: "sigara_icme",
    DriverBehavior.EMNIYET_KEMERI_IHLALI: "emniyet_kemeri_ihlali",
    DriverBehavior.TEKNOCAN: "teknocan",
    DriverBehavior.BILGISAYAR: "bilgisayar",
    DriverBehavior.LICENSE_PLATE: "license_plate",
}

COMPETITION_LABEL_MAP = {
    DriverBehavior.TELEFONLA_KONUSMA: ("surucu_davranisi", "telefonla_konusma"),
    DriverBehavior.SU_ICME: ("surucu_davranisi", "su_icme"),
    DriverBehavior.ARKAYA_BAKMA: ("surucu_davranisi", "arkaya_bakma"),
    DriverBehavior.ESNEME: ("surucu_davranisi", "esneme"),
    DriverBehavior.SIGARA_ICME: ("surucu_davranisi", "sigara_icme"),
    DriverBehavior.EMNIYET_KEMERI_IHLALI: ("surucu_davranisi", "emniyet_kemeri_ihlali"),
    DriverBehavior.TEKNOCAN: ("nesne", "teknocan"),
    DriverBehavior.BILGISAYAR: ("nesne", "bilgisayar"),
    DriverBehavior.LICENSE_PLATE: ("plaka", "license_plate"),
}
```

### 3.3. Faz 1 Sonu Kanıtları

| Kanıt | Dosya |
|---|---|
| Sınıf kaydı doğrulama çıktısı | `reports/final_evidence/pipeline/class_registry_validation.md` |
| Split oranı ve sınıf dağılımı | `reports/final_evidence/data/split_consistency_report.md` |
| Memory cleanup stress çıktısı | `reports/final_evidence/pipeline/memory_cleanup_test.json` |
| Threshold karşılaştırması | `reports/final_evidence/model/class_threshold_ablation.csv` |
| Zero-DCE koşullu çalışma benchmark | `reports/final_evidence/pipeline/conditional_enhancement_benchmark.json` |

---

## 4. Faz 2 — Veri Seti Güçlendirme

**Süre:** 1–2 hafta  
**Hedef:** Kritik düşük destekli sınıfları gerçek veri, sentetik veri ve temiz etiketle güçlendirmek.

### 4.1. Veri Hedefleri

| Sınıf | Minimum hedef | Ana veri yöntemi | Not |
|---|---:|---|---|
| `teknocan` | 500+ gerçek görüntü | Araç içi oyuncak çekimi, farklı kamera açıları | Sentetik destek olabilir ama gerçek veri şart |
| `bilgisayar` | 500+ gerçek görüntü | Kabin içi laptop/tablet kullanım senaryoları | Telefonla karışma riski yüksek |
| `emniyet_kemeri_ihlali` | 1500+ görüntü | Farklı ışık, açı, sürücü ve kıyafet | Kemer/kemersiz ayrımı net olmalı |
| `telefonla_konusma` | 1500+ görüntü | Farklı telefon tutuş pozisyonları | El-yüz yakınlığına dikkat |
| `arkaya_bakma` | 500+ görüntü | Gövde ve baş dönüş açısı çeşitliliği | Normal baş hareketinden ayrılmalı |

### 4.2. Veri İş Akışı

1. **Anotasyon rehberi oluştur:** Her sınıf için pozitif/negatif örnek, box sınırı ve edge-case açıklaması yazılır.
2. **Yeni veri topla:** Eksik sınıflar için kısa videolar çekilir veya lisansı uygun kaynaklardan görüntü alınır.
3. **Anahtar kare çıkar:** Optik akış / scene change ile benzer kareler elenir.
4. **İlk etiketleme yap:** CVAT/Roboflow veya mevcut anotasyon aracı kullanılır.
5. **Etiket kalite audit’i çalıştır:** Model tahmini ile GT karşılaştırılır; IoU < 0.3 veya sınıf uyuşmazlığı şüpheli etiket olarak listelenir.
6. **Augmentasyon uygula:** Albumentations gelişmiş pipeline eklenir.
7. **Copy-paste üret:** Teknocan ve laptop gibi nesneler için segment/crop tabanlı sentez yapılır.
8. **Lisans envanterini güncelle:** Her kaynak için URL, lisans, kullanım amacı ve rapor referansı yazılır.

### 4.3. Faz 2 Kabul Kriterleri

| Kriter | Minimum eşik |
|---|---:|
| Her kritik sınıfta test desteği | En az 100 örnek |
| Her kritik sınıfta eğitim desteği | En az 500 örnek |
| Lisans durumu | Teslimde kullanılan tüm kaynaklar doğrulanmış |
| Etiket audit sonucu | Şüpheli etiket listesi gözden geçirilmiş |
| Split tutarlılığı | Kişi/oturum sızıntısı kontrol edilmiş |

---

## 5. Faz 3 — Model Eğitimi ve Aday Model Kapısı

**Süre:** 1–2 hafta, GPU gerekir  
**Hedef:** Yeni modelleri kontrollü şekilde eğitmek ve üretime alınıp alınamayacağını ölçmek.

### 5.1. Eğitim Sırası

| Sıra | Deney | Amaç | Çıktı | Üretime alma şartı |
|---|---|---|---|---|
| 1 | `detector_v4_full` 50 epoch | Eksik eğitimi tamamlamak | `runs/detect/detector_v4_full_50e/` | v3 mAP/F1 değerlerine yaklaşmalı veya geçmeli |
| 2 | Class-balanced / focal deneme | Az sınıfları güçlendirmek | `reports/model/balanced_loss_ablation.csv` | Kritik sınıflarda AP artışı olmalı |
| 3 | YOLOv8l veya YOLOv11m deneme | Doğruluk/hız kıyaslamak | `reports/model/backbone_comparison.md` | Latency ve boyut sınırını aşmamalı |
| 4 | CRNN Türk plaka fine-tune | OCR hedefini ölçüme çevirmek | `reports/ocr/ocr_accuracy_report.md` | Exact match ve CER eşiği sağlanmalı |
| 5 | CNN-LSTM iyileştirme | Temporal davranışları güçlendirmek | `reports/model/temporal_model_report.md` | E2E karmaşıklığına değer katmalı |

### 5.2. Model Promotion Gate

Aday model üretime ancak şu kapılardan geçerse alınır:

| Kapı | Beklenen durum |
|---|---|
| Genel mAP@0.50 | `detector_v3` değerinden düşük olmamalı veya kritik sınıf kazanımı gerekçelendirilmiş olmalı |
| mAP@0.50:0.95 | Aşırı düşüş olmamalı |
| Precision / Recall / F1 | F1 genel olarak korunmalı; kritik sınıflarda iyileşme aranmalı |
| Sınıf bazlı AP | `teknocan`, `bilgisayar`, `emniyet_kemeri_ihlali`, `sigara_icme` özel incelenmeli |
| E2E latency | Teslim zaman bütçesini aşmamalı |
| Docker boyutu | Yarışma limitini aşmamalı |
| JSON sözleşmesi | `results.json` schema validasyonunu geçmeli |
| Regresyon testleri | Mevcut testler bozulmamalı |

**Karar:** Bu kapılardan biri başarısızsa model `models/candidates/` altında kalır, production `detector_v3` olarak devam eder.

---

## 6. Faz 4 — Pipeline Optimizasyonu

**Süre:** 3–5 gün  
**Hedef:** Sistemi hız, bellek, takip kararlılığı ve OCR doğruluğu açısından iyileştirmek.

| Görev | Dosya | Beklenen etki | Test |
|---|---|---|---|
| Batch inference | `ftr_main.py`, detector wrapper | GPU verimliliği | E2E latency benchmark |
| FP16 ONNX aday | `scripts/export_fp16.py`, model dosyaları | CUDA hız artışı | FP32/FP16 doğruluk farkı |
| Adaptif stride | `ftr_main.py` | Riskli sahnede sık, risksiz sahnede seyrek analiz | Risk senaryosu simülasyonu |
| Hungarian matching | `tracking_pipeline.py` | Daha iyi ID tutarlılığı | ID switch ölçümü |
| Temporal plaka voting | `plate_ocr.py` | Plaka kararlılığı | OCR exact match + temporal accuracy |
| Track cleanup | `ftr_main.py`, tracker state | Bellek kontrolü | Uzun video stress testi |

### 6.1. Pipeline Değişiklik Sırası

1. Önce ölçüm altyapısı sabitlenir.
2. Bellek cleanup uygulanır.
3. Threshold ve conditional enhancement eklenir.
4. Adaptif stride eklenir.
5. Batch inference ve FP16 yalnızca benchmark ile açılır.
6. Tracking ve OCR iyileştirmeleri ayrı feature flag ile entegre edilir.

**Feature flag önerisi:**

```bash
--enable-class-thresholds
--enable-conditional-enhancement
--enable-adaptive-stride
--enable-batch-inference
--enable-fp16
--enable-hungarian-tracker
--enable-temporal-plate-voting
```

---

## 7. Faz 5 — 5G / Canlı Sistem Katmanı

**Süre:** 1–2 hafta, operatör sandbox erişimine bağlı  
**Hedef:** FTR offline başarısını bozmadan canlı mimariyi raporlanabilir hale getirmek.

| Alt sistem | İş | Çıktı | Kabul kriteri |
|---|---|---|---|
| WebRTC | Adaptif kalite profilleri | `reports/5g/adaptive_quality_demo.md` | RTT, packet loss ve QoD durumuna göre profil değişir |
| QoD karar motoru | Ağ kalite skoru + risk skoru | `reports/5g/qod_decision_trace.json` | QoD kararları açıklanabilir olur |
| Android edge | Küçük TFLite model adayı | `reports/5g/android_model_benchmark.md` | Mobilde makul FPS ve boyut |
| Offline cache | Bağlantı kopunca event saklama | Android Room / local DB | Bağlantı gelince batch sync |
| MEC API | Uçtan uca istek/yanıt sözleşmesi | OpenAPI / JSON örnekleri | Backend ve mobil aynı şemada konuşur |

**Not:** 5G canlı entegrasyon, operatör SIM/sandbox koşullarına bağlıysa raporda “canlı entegrasyon hazır mimari + sandbox bekleyen durum” olarak ayrılmalıdır.

---

## 8. Faz 6 — Test, Doğrulama ve Final Kanıt Paketi

**Süre:** Sürekli, final öncesi 2–3 gün yoğunlaştırılır  
**Hedef:** Her teknik iddiayı kanıt dosyasına bağlamak.

### 8.1. Test Paketi

| Test | Dosya | Amaç | Minimum kabul |
|---|---|---|---|
| Sınıf bazlı regresyon | `tests/test_per_class_regression.py` | AP düşüşlerini yakalamak | Kritik sınıflarda minimum eşik korunur |
| OCR doğruluk testi | `tests/test_ocr_accuracy.py` | Plaka okuma başarısını ölçmek | 300+ crop ile exact match ve CER raporu |
| Uzun video stress | `tests/test_stress.py` | Süre ve bellek sınırı | 10 dk video bütçe içinde biter |
| JSON schema | `tests/test_results_schema.py` | Yarışma çıktı uyumu | `additionalProperties=false` geçer |
| Docker boyutu | CI/CD script | Teslim paketi limiti | Limit altında kalır |
| T4 CUDA benchmark | Dış/GPU ortamı | Yarışma benzeri hız | Raporlanabilir latency |

### 8.2. Final Kanıt Paketi Dizini

```text
reports/final_evidence/
  data/
    split_consistency_report.md
    class_distribution_before_after.csv
    annotation_guideline.md
    label_audit_report.csv
    license_inventory.md
  model/
    detector_v3_baseline_metrics.json
    detector_v4_full_50e_metrics.json
    candidate_model_comparison.md
    confusion_matrix.png
    threshold_calibration.csv
  pipeline/
    class_registry_validation.md
    memory_cleanup_test.json
    e2e_latency_benchmark.json
    conditional_enhancement_benchmark.json
  ocr/
    ocr_test_set_manifest.csv
    ocr_accuracy_report.md
    plate_temporal_voting_report.md
  tracking/
    id_switch_report.md
    hungarian_ablation.csv
  5g/
    adaptive_quality_demo.md
    qod_decision_trace.json
    api_contract_examples.json
  tests/
    pytest_final.txt
    docker_size_check.txt
    json_schema_validation.txt
    stress_test_report.md
```

---

## 9. Gün Gün Önerilen Uygulama Takvimi

| Gün | Yapılacak ana iş | Gün sonu çıktısı |
|---:|---|---|
| 0 | Branch, model lock, baseline test/benchmark | Mevcut durum kanıtları |
| 1 | `class_registry.py`, sınıf map düzeltmeleri | Class registry validation |
| 2 | Split oranı standardizasyonu, rapor eşitleme | Split consistency report |
| 3 | Memory cleanup, threshold, conditional Zero-DCE | Pipeline patch + benchmark |
| 4–5 | Anotasyon rehberi, veri toplama planı, lisans envanteri | Data collection checklist |
| 6–10 | Kritik sınıflar için veri toplama/etiketleme/augmentasyon | Yeni dataset versiyonu |
| 11 | Label audit ve split üretimi | Temizlenmiş split |
| 12–14 | `detector_v4_full` tam eğitim | Eğitim logları ve metrikler |
| 15 | Balanced loss / threshold ablation | Ablation raporu |
| 16–17 | OCR test seti ve CRNN fine-tune | OCR accuracy report |
| 18 | Pipeline optimizasyonları | E2E benchmark karşılaştırması |
| 19 | Uzun video stress ve Docker kontrol | Stress + size raporu |
| 20 | Final quality gate ve rapor entegrasyonu | Final evidence index |

---

## 10. Risk ve Geri Alma Planı

| Risk | Etki | Önlem | Geri alma |
|---|---|---|---|
| `CLASS_MAP` değişikliği eski sonuçları bozar | Yüksek | Önce mapping testleri yaz | Eski map ile compatibility layer |
| `detector_v4` düşük kalır | Orta | Production v3 kilidini koru | v4 aday klasörde kalır |
| Yeni veri lisansı belirsiz | Yüksek | Lisans envanteri zorunlu | Belirsiz kaynak teslimden çıkarılır |
| FP16 doğruluk düşürür | Orta | FP32/FP16 karşılaştır | FP32’ye dön |
| Batch inference bellek artırır | Orta | Batch size 2/4 ablation | Tek kare inference’a dön |
| OCR test seti yetersiz kalır | Orta | 300+ crop hedefle | OCR iddiasını “hedef” yerine “sınırlı ölçüm” diye yaz |
| 5G sandbox yetişmez | Orta | Mock + sözleşme + mimari kanıt üret | Raporda canlı entegrasyon bağımlılığı açık yazılır |

---

## 11. Final Raporuna Yazılacak Kısa Formül

Rapor dilinde her iş şu kalıpla yazılmalıdır:

> “Mevcut sistemde ___ problemi tespit edilmiştir. Bu problemi gidermek için ___ uygulanmıştır. Uygulama sonrasında ___ testi yapılmış, sonuçlar ___ dosyasında kanıtlanmıştır. Elde edilen değer ___ olup, önceki değer ___ ile karşılaştırıldığında ___ sonucuna ulaşılmıştır.”

Örnek:

> “Mevcut sistemde sınıf tanımlarının `data.yaml`, `ai_pipeline.py` ve yarışma çıktı sözleşmesi arasında farklılaştığı tespit edilmiştir. Bu problemi gidermek için tüm sınıf indeksleri `src/class_registry.py` altında tek kaynak haline getirilmiştir. Uygulama sonrasında sınıf kayıt doğrulama testi çalıştırılmış, sonuçlar `reports/final_evidence/pipeline/class_registry_validation.md` dosyasında kanıtlanmıştır.”

---

## 12. Yapılmayacaklar Listesi

1. `detector_v4_full` yalnızca 2 epoch sonucu ile production’a alınmayacak.
2. Lisansı belirsiz yerel veya dış veri final teslim veri seti olarak gösterilmeyecek.
3. `data.yaml`, inference class map ve yarışma sözleşmesi ayrı ayrı elle güncellenmeyecek; hepsi `class_registry.py` üzerinden üretilecek.
4. OCR başarısı ölçülmeden “yüksek doğruluk” iddiası yazılmayacak.
5. T4/CUDA benchmark olmadan GPU performansı kesin ifade edilmeyecek.
6. Canlı 5G entegrasyon operatör bağımlılığı saklanmayacak; mock, sözleşme ve mimari kanıt ile ayrıştırılacak.

---

## 13. En Hızlı Başlangıç İçin İlk 5 İş

1. `src/class_registry.py` oluştur ve tüm sınıf map’lerini buraya bağla.
2. Split oranını tek değere indir ve `split_consistency_report.md` üret.
3. `ftr_main.py` içinde track buffer / debounce / plate cleanup eksiklerini kapat.
4. Sınıf bazlı threshold değerlerini config’e al ve ablation CSV üret.
5. Teknocan + bilgisayar + emniyet kemeri için veri toplama ve OCR test seti üretimini başlat.

Bu 5 iş tamamlanmadan daha büyük model denemelerine geçilmemelidir; çünkü veri ve sözleşme tutarsızken model iyileştirmesi güvenilir ölçülemez.
