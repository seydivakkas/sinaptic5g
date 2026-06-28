# SİNAPTİC5G Adım Adım Komut Rehberi

Bu dosya, `SINAPTIC5G_UYGULAMA_DOSYASI.md` içindeki sonraki aşama listesini Windows PowerShell ortamında uygulanabilir komutlara dönüştürür.

Komutları `D:\SİNAPTİC\5G PROJE` dizininde çalıştır. Canlı 5G adımlarında gerçek endpoint, SIM, Turkcell Open Gateway yetkisi ve HTTPS/WSS servisleri gerekir. FTR Docker adımları bu dış erişime bağlı değildir.

---

## 0. Başlangıç

```powershell
cd "D:\SİNAPTİC\5G PROJE"

python --version
docker --version
nvidia-smi
```

Beklenen:

- Python çalışmalı.
- Docker çalışmalı.
- GPU test edilecekse `nvidia-smi` NVIDIA sürücüsünü göstermeli.

---

## 1. FTR Python Bağımlılık ve Şema Kontrolü

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-ftr.txt

python -m py_compile `
  ftr_main.py `
  src\competition_contract.py `
  src\competition_adapter.py `
  src\tracking_pipeline.py
```

JSON şemasının parse edilebilir olduğunu kontrol et:

```powershell
python - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator

schema_path = Path("schemas/results.schema.json")
schema = json.loads(schema_path.read_text(encoding="utf-8"))
Draft202012Validator.check_schema(schema)
print("OK: results.schema.json geçerli")
PY
```

PowerShell burada heredoc sorun çıkarırsa şu alternatifi kullan:

```powershell
python -c "import json; from pathlib import Path; from jsonschema import Draft202012Validator; s=json.loads(Path('schemas/results.schema.json').read_text(encoding='utf-8')); Draft202012Validator.check_schema(s); print('OK: results.schema.json geçerli')"
```

---

## 2. Pytest Kabul Koşusu

Tüm testleri çalıştır:

```powershell
python -m pytest -q
```

Daha hızlı sözleşme odaklı kontrol için:

```powershell
python -m pytest -q `
  tests\test_competition_contract.py `
  tests\test_model_manifest.py `
  tests\test_bff_media_boundary.py `
  tests\test_webrtc_signaling.py `
  tests\test_qod_orchestrator.py
```

Beklenen:

- Kabul raporundaki referans koşuda `81 passed` görülmüş.
- Bu aşamada hata olursa Docker build'e geçmeden önce düzeltilmeli.

---

## 3. Model Lock ve Hash Doğrulaması

Model kilit dosyasını ve kritik model dosyalarını kontrol et:

```powershell
Get-Item `
  model_lock.json, `
  yolov8n.onnx, `
  models\detector_optimized.onnx, `
  models\lprnet.onnx, `
  models\crnn.onnx
```

Hash değerlerini kaydet:

```powershell
Get-FileHash -Algorithm SHA256 `
  model_lock.json, `
  yolov8n.onnx, `
  models\detector_optimized.onnx, `
  models\lprnet.onnx, `
  models\crnn.onnx |
  Format-Table -AutoSize
```

FTR entrypoint model lock doğrulamasını çalıştırma sırasında otomatik yapar. Ayrı bir smoke girdi videosu varsa yerel test:

```powershell
New-Item -ItemType Directory -Force tests\local_output | Out-Null

python ftr_main.py `
  --video tests\smoke_input\smoke_test.mp4 `
  --output tests\local_output\results.json

python scripts\validate_results_schema.py `
  tests\local_output\results.json
```

---

## 4. FTR Docker Build

Kanonik FTR imajını oluştur:

```powershell
docker build -t teknofest/sinaptic5g:latest .
```

İmaj boyutunu kontrol et:

```powershell
$imageBytes = docker image inspect teknofest/sinaptic5g:latest --format '{{.Size}}'
"Docker image size: {0:N2} GB" -f ([int64]$imageBytes / 1GB)
```

Beklenen:

- İmaj 8 GB sınırının altında olmalı.
- Base image `nvidia/cuda:12.1.0-base-ubuntu22.04` olmalı.

Base image kontrolü:

```powershell
docker history teknofest/sinaptic5g:latest --no-trunc
```

---

## 5. FTR Docker Run ve Results Üretimi

Önce giriş/çıkış klasörünü hazırla:

```powershell
New-Item -ItemType Directory -Force data\input,data\output | Out-Null
Remove-Item -LiteralPath data\output\results.json -Force -ErrorAction SilentlyContinue
```

Eğer `data\input\video.mp4` yoksa smoke videoyu kopyala:

```powershell
Copy-Item -LiteralPath tests\smoke_input\smoke_test.mp4 -Destination data\input\video.mp4 -Force
```

Docker içinde ağ kapalı çalıştır:

```powershell
Measure-Command {
  docker run --rm --gpus all --network none --cpus 4 --memory 16g --shm-size 2g `
    -v "${PWD}\data\input\video.mp4:/app/data/input/video.mp4:ro" `
    -v "${PWD}\data\output:/app/data/output:rw" `
    teknofest/sinaptic5g:latest
}
```

GPU olmayan makinede yalnız sözleşme smoke için CPU fallback denenebilir:

```powershell
Measure-Command {
  docker run --rm --network none --cpus 4 --memory 16g --shm-size 2g `
    -v "${PWD}\data\input\video.mp4:/app/data/input/video.mp4:ro" `
    -v "${PWD}\data\output:/app/data/output:rw" `
    teknofest/sinaptic5g:latest
}
```

Not: Resmi yarışma hedefi Tesla T4 GPU olduğu için final provada `--gpus all` ile test tercih edilmeli.

---

## 6. Results JSON Kontrolü

Çıktının oluştuğunu kontrol et:

```powershell
Get-Item data\output\results.json
Get-Content -Raw -Encoding UTF8 data\output\results.json
```

Şema doğrulaması:

```powershell
python scripts\validate_results_schema.py `
  data\output\results.json
```

Hızlı JSON anahtar kontrolü:

```powershell
python -c "import json; d=json.load(open('data/output/results.json', encoding='utf-8')); print(d.keys()); print(d['arac_bilgisi'].keys()); print(len(d['tespitler']))"
```

Beklenen kök anahtarlar:

- `video_id`
- `arac_bilgisi`
- `tespitler`

---

## 7. Docker İmajını Teslim İçin Kaydetme

```powershell
New-Item -ItemType Directory -Force ..\output | Out-Null

docker save teknofest/sinaptic5g:latest -o ..\output\sinaptic5g-ftr-latest.tar

Get-FileHash -Algorithm SHA256 ..\output\sinaptic5g-ftr-latest.tar
Get-Item ..\output\sinaptic5g-ftr-latest.tar | Select-Object FullName,Length,LastWriteTime
```

Başka bilgisayarda yükleme provası:

```powershell
docker load -i ..\output\sinaptic5g-ftr-latest.tar
docker images | Select-String "sinaptic5g"
```

---

## 8. Android Model Asset Lock Kontrolü

Android klasörüne geç:

```powershell
cd "D:\SİNAPTİC\5G PROJE\android"
```

Java yolunu ayarla:

```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
```

Model manifest ve lock dosyalarını kontrol et:

```powershell
Get-Item `
  model_lock.json, `
  app\src\main\assets\model_manifest.json, `
  app\src\main\assets\yolov8n.tflite
```

Gradle preBuild sırasında `verifyModelAsset` otomatik çalışır. Ayrı kontrol:

```powershell
.\gradlew.bat :app:preBuild
```

Kök klasöre dön:

```powershell
cd "D:\SİNAPTİC\5G PROJE"
```

---

## 9. Android Debug APK Build

Gerçek API adresleri henüz yoksa build yalnız derleme doğrulaması içindir:

```powershell
cd "D:\SİNAPTİC\5G PROJE\android"
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"

.\gradlew.bat clean :app:assembleDebug `
  -PAPI_BASE_URL=https://api.example `
  -PMEDIA_BASE_URL=https://media.example
```

Canlı demo için gerçek endpointlerle yeniden build:

```powershell
.\gradlew.bat clean :app:assembleDebug `
  -PAPI_BASE_URL=https://GERCEK-BFF-ADRESI `
  -PMEDIA_BASE_URL=https://GERCEK-MEDYA-ADRESI `
  -POIDC_AUTHORIZATION_ENDPOINT=https://GERCEK-OIDC-AUTH-ENDPOINT `
  -POIDC_TOKEN_ENDPOINT=https://GERCEK-OIDC-TOKEN-ENDPOINT `
  -POIDC_CLIENT_ID=GERCEK_PUBLIC_CLIENT_ID `
  -POIDC_REDIRECT_URI=com.sinaptic.tripwire:/oauth2redirect
```

APK konumu:

```powershell
Get-Item app\build\outputs\apk\debug\app-debug.apk
Get-FileHash -Algorithm SHA256 app\build\outputs\apk\debug\app-debug.apk
```

Kök klasöre dön:

```powershell
cd "D:\SİNAPTİC\5G PROJE"
```

---

## 10. APK Sır ve Model Taraması

```powershell
python scripts\scan_android_apk.py
```

Beklenen:

- APK içinde tam olarak 1 adet TFLite.
- APK içinde 0 adet ONNX.
- Gömülü secret, token, private credential bulunmamalı.

APK'yı çıktı klasörüne kopyala:

```powershell
New-Item -ItemType Directory -Force ..\output\android | Out-Null
Copy-Item `
  -LiteralPath android\app\build\outputs\apk\debug\app-debug.apk `
  -Destination ..\output\android\SINAPTIC5G-debug.apk `
  -Force

Get-FileHash -Algorithm SHA256 ..\output\android\SINAPTIC5G-debug.apk
```

---

## 11. Canlı BFF, Redis ve Medya Servisini Ayağa Kaldırma

Bu bölüm gerçek canlı demo içindir. `.env` dosyası teslim paketine girmemelidir.

Gerekli ortam değişkenleri örnekleri:

```powershell
cd "D:\SİNAPTİC\5G PROJE"

$env:REDIS_URL = "redis://localhost:6379/0"
$env:TURKCELL_TOKEN_URL = "https://GERCEK-TOKEN-ENDPOINT"
$env:TURKCELL_NUMBER_VERIFY_URL = "https://GERCEK-NUMBER-VERIFY-ENDPOINT"
$env:TURKCELL_QOD_BASE_URL = "https://GERCEK-QOD-ENDPOINT"
$env:TURKCELL_QOD_CLIENT_ID = "GERCEK_CLIENT_ID"
$env:TURKCELL_QOD_CLIENT_SECRET = "GERCEK_CLIENT_SECRET"
$env:QOD_PROFILE = "OPERATOR_ONBOARDING_QOS_PROFILE"
$env:APP_SERVER_IPV4 = "SUNUCU_IPV4"
$env:APP_SERVER_PORT = "443"
$env:DEVICE_PORT = "443"
$env:ICE_SERVERS_JSON = '[{"urls":["stun:stun.l.google.com:19302"]}]'
```

Redis başlatma örneği:

```powershell
docker run --rm -p 6379:6379 --name sinaptic5g-redis redis:7-alpine
```

Ayrı PowerShell penceresinde BFF:

```powershell
cd "D:\SİNAPTİC\5G PROJE"
python server.py
```

Ayrı PowerShell penceresinde medya servisi:

```powershell
cd "D:\SİNAPTİC\5G PROJE"
python media_service.py
```

Sağlık kontrolü:

```powershell
Invoke-RestMethod http://localhost:8000/health/live
Invoke-RestMethod http://localhost:8000/health/ready
Invoke-RestMethod http://localhost:8001/health
```

Portlar ortam dosyana göre farklıysa komutlardaki URL'leri güncelle.

---

## 12. Number Verification Denemesi

Bu adım mobil veri, gerçek SIM ve sağlayıcı erişimi ister. Wi-Fi üzerinden çağrıda başarısızlık beklenebilir.

Android uygulama üzerinden:

1. APK'yı telefona kur.
2. Telefonu mobil veri bağlantısına al.
3. Uygulamayı aç.
4. Telefon numarasını `+905xxxxxxxxx` formatında gir.
5. `NUMARAYI DOĞRULA` butonuna bas.
6. Arayüzde `NUMARA DOĞRULANDI` durumunu bekle.

Logcat izleme:

```powershell
adb logcat | Select-String "TripWire|NumberVerification|QoDManager|WebRtcMedia"
```

APK kurulum:

```powershell
adb install -r "D:\SİNAPTİC\output\android\SINAPTIC5G-debug.apk"
```

---

## 13. QoD Start / Stop Testi

QoD uygulamada otomatik tetiklenir. Manuel API testi için önce geçerli app session token gerekir; bunu normalde Android Number Verification akışı üretir.

Uygulama üzerinden beklenen akış:

1. Numara doğrulandı.
2. Kamera canlı.
3. Araç kadraja girdi.
4. Araç yaklaşma sinyali oluştu.
5. `5G QoD AKTİF` veya `5G BEST EFFORT — QoD AÇILMADI` durumu gösterildi.

BFF loglarında beklenenler:

- `/auth/verify`
- `/qod/start`
- Provider başarısızsa Best Effort fallback.
- Başarılıysa `session_id`.

QoD stop, uygulamada 300 saniye sonunda otomatik denenir. Yarışma demosunda daha kısa süre gerekiyorsa kod tarafında süre değiştirilmeden manuel API ile kapatma tercih edilir.

---

## 14. Android - WebRTC - GPU Uçtan Uca Test

BFF ve medya servisi çalışırken:

1. Android uygulamada numarayı doğrula.
2. Telemetri bağlantısının `BAĞLI` olduğunu kontrol et.
3. WebRTC durumunun `P2P BAĞLI` olduğunu kontrol et.
4. Kameraya araç veya test sahnesi göster.
5. Sağ panelde plaka, araç tipi/renk, risk, olay zamanı ve hız alanlarını izle.

Logcat:

```powershell
adb logcat | Select-String "WEBRTC|TELEMETRI|TripWire|FrameProcessor"
```

BFF sağlık:

```powershell
Invoke-RestMethod http://localhost:8000/stats
```

Medya servisi sağlık:

```powershell
Invoke-RestMethod http://localhost:8001/health
```

Sorun yaşanırsa hızlı ayrım:

| Belirti | Kontrol |
|---|---|
| Numara doğrulanmıyor | Mobil veri, OIDC endpoint, Number Verification URL |
| Telemetri bağlanmıyor | `MEDIA_BASE_URL`, WSS sertifikası, token |
| WebRTC bağlanmıyor | ICE/TURN config, UDP/TCP izinleri, BFF SDP route |
| Görüntü var sonuç yok | GPU model path, media service log, ONNX provider |
| QoD açılmıyor | Yaklaşma sinyali, benefit gate, provider credential |

---

## 15. Teslim Paketi Son Hash ve Boyut Kaydı

Kök dizine dön:

```powershell
cd "D:\SİNAPTİC"
```

Kritik dosyaların hash değerlerini kaydet:

```powershell
Get-FileHash -Algorithm SHA256 `
  "SINAPTIC5G_UYGULAMA_DOSYASI.md", `
  "SINAPTIC5G_ADIM_ADIM_KOMUT_REHBERI.md", `
  "5G PROJE\Dockerfile", `
  "5G PROJE\ftr_main.py", `
  "5G PROJE\model_lock.json", `
  "5G PROJE\schemas\results.schema.json", `
  "output\sinaptic5g-ftr-latest.tar", `
  "output\android\SINAPTIC5G-debug.apk" |
  Tee-Object -FilePath "output\teslim_hash_kaydi.txt"
```

Boyut kontrolü:

```powershell
Get-Item `
  "output\sinaptic5g-ftr-latest.tar", `
  "output\android\SINAPTIC5G-debug.apk" |
  Select-Object FullName,Length,LastWriteTime |
  Tee-Object -FilePath "output\teslim_boyut_kaydi.txt"
```

Teslim öncesi gizli dosya kontrolü:

```powershell
Get-ChildItem -Recurse -Force |
  Where-Object {
    $_.Name -match '^\.env$|secret|token|credential|private|\.pem$|\.key$'
  } |
  Select-Object FullName,Length,LastWriteTime
```

Bu liste gerçek secret dosyası gösterirse teslim paketinden çıkar.

---

## 16. En Kısa Yarışma Günü Komut Dizisi

Sadece FTR teslim provası için minimum sıra:

```powershell
cd "D:\SİNAPTİC\5G PROJE"

python -m pytest -q tests\test_competition_contract.py tests\test_model_manifest.py

docker build -t teknofest/sinaptic5g:latest .

New-Item -ItemType Directory -Force data\input,data\output | Out-Null
Copy-Item -LiteralPath tests\smoke_input\smoke_test.mp4 -Destination data\input\video.mp4 -Force
Remove-Item -LiteralPath data\output\results.json -Force -ErrorAction SilentlyContinue

docker run --rm --gpus all --network none --cpus 4 --memory 16g --shm-size 2g `
  -v "${PWD}\data\input\video.mp4:/app/data/input/video.mp4:ro" `
  -v "${PWD}\data\output:/app/data/output:rw" `
  teknofest/sinaptic5g:latest

python scripts\validate_results_schema.py data\output\results.json

docker save teknofest/sinaptic5g:latest -o ..\output\sinaptic5g-ftr-latest.tar
Get-FileHash -Algorithm SHA256 ..\output\sinaptic5g-ftr-latest.tar
```

---

## 17. Başarılı Tamamlama Kriteri

Bu rehberin FTR kısmı tamamlandığında şu çıktılar elde edilmiş olmalı:

- `5G PROJE\data\output\results.json`
- `output\sinaptic5g-ftr-latest.tar`
- `output\teslim_hash_kaydi.txt`
- `output\teslim_boyut_kaydi.txt`

Canlı uygulama kısmı tamamlandığında ek olarak:

- `output\android\SINAPTIC5G-debug.apk`
- Android arayüzünde numara doğrulama durumu
- Telemetri bağlantısı
- WebRTC bağlantısı
- QoD Best Effort veya aktif durum kaydı
- GPU servis loglarında çıkarım akışı
