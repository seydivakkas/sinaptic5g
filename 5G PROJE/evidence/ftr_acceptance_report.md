# Offline FTR Kabul Test Raporu (Offline FTR Acceptance Report)

* **Tarih:** 2026-06-21
* **Girdi Videosu:** `tests/smoke_input/smoke_test.mp4`
* **Çıktı JSON Yolu:** `tests/final_smoke_output/results.json`

## 1. Bağımlılık ve Çevrimdışı Çalışma Doğrulaması
* **Ağ Bağımsızlığı:** `TAMAMLANDI` (İnternet veya CAMARA çağrısı yapılmamıştır)
* **Harici Servis Bağımsızlığı:** `TAMAMLANDI` (Redis veya Android bağımlılığı yoktur)

## 2. JSON Şema Doğrulaması
* **Şema Sürümü:** JSON Schema Draft 2020-12
* **additionalProperties=false Kontrolü:** `TAMAMLANDI`
* **Durum:** TAMAMLANDI (Doğrulama başarılı)

## 3. Pytest Sonuçları
* **Pytest Durumu:** TAMAMLANDI

```
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0
rootdir: D:\SİNAPTİC\5G PROJE
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.13.0, asyncio-1.3.0, cov-7.1.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 81 items

tests\test_all_components.py .....                                       [  6%]
tests\test_auth_utils.py ..                                              [  8%]
tests\test_bff_media_boundary.py ..                                      [ 11%]
tests\test_competition_contract.py ......                                [ 18%]
tests\test_dataset_corpus.py ...                                         [ 22%]
tests\test_latest_frame.py .                                             [ 23%]
tests\test_model_manifest.py ..                                          [ 25%]
tests\test_ocr_accuracy.py ........                                      [ 35%]
tests\test_per_class_regression.py .......................               [ 64%]
tests\test_qod_orchestrator.py ........                                  [ 74%]
tests\test_signaling_api.py .......                                      [ 82%]
tests\test_stress.py ......                                              [ 90%]
tests\test_telemetry_hub.py .                                            [ 91%]
tests\test_tracking_pipeline.py ..                                       [ 93%]
tests\test_webrtc_signaling.py .....                                     [100%]

======================= 81 passed in 114.61s (0:01:54) ========================

```

## 4. Kaydedilen Sonuç İçeriği (Özet)
```json
{
  "video_id": "smoke_test.mp4",
  "arac_bilgisi": {
    "tip": "sedan",
    "plaka": "01A0000",
    "renk": "gri",
    "confidence_score": 0.0
  },
  "tespitler": []
}
```
