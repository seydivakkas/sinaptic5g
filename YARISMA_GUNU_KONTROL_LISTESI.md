# SİNAPTİC5G — Yarışma Günü Kontrol Listesi

Bu liste final canlı gösterimi ile çevrimdışı FTR Docker teslimini birlikte kapsar. Kutuları sırayla işaretleyin; yarışma sabahı yeni bağımlılık, sürücü veya model güncellemesi yapmayın.

## 1. Görev dağılımı

- [ ] Sunumu yapacak kişi: ____________________
- [ ] Android/canlı demo operatörü: ____________________
- [ ] Docker–GPU ve arıza kurtarma sorumlusu: ____________________
- [ ] Her rolün en az bir yedeği belirlendi.
- [ ] Jüri sorularında yalnız ölçülmüş değerlerin söyleneceği ekipçe teyit edildi.

## 2. Yarışmadan bir gün önce

- [ ] Depo donduruldu; çalışan kopya ayrıca iki farklı diske/USB belleğe alındı.
- [ ] [Final rapor PDF'i](output/pdf/SINAPTIC5G_Final_Tasarim_Raporu.pdf) açıldı ve 7 sayfanın tamamı kontrol edildi.
- [ ] Kaynak kod, PDF, APK, Docker imajı ve örnek video çevrimdışı olarak ikinci bilgisayara kopyalandı.
- [ ] `.env`, API anahtarı, token ve özel sertifikalar teslim paketine eklenmedi.
- [ ] İki telefon, iki veri kablosu, şarj cihazları, powerbank, HDMI/USB-C dönüştürücü, Ethernet kablosu, araç/telefon tutucu ve yedek hotspot hazırlandı.
- [ ] Sunumun PDF kopyası hem çevrimdışı hem USB bellekte hazırlandı.

### Kod ve sözleşme doğrulaması

`D:\SİNAPTİC\5G PROJE` klasöründe:

```powershell
python -m pytest -q
python -m py_compile ftr_main.py src/tracking_pipeline.py
```

- [ ] Test sonucu `81 passed`.
- [ ] Python derleme kontrolü hatasız.
- [ ] Model hash değerleri `model_lock.json` ile eşleşiyor.

### FTR Docker provası

```powershell
docker build -t sinaptic5g-ftr:canonical .

$imageBytes = docker image inspect sinaptic5g-ftr:canonical --format '{{.Size}}'
"{0:N2} GB" -f ([int64]$imageBytes / 1GB)

Remove-Item -LiteralPath data/output/results.json -Force -ErrorAction SilentlyContinue
Measure-Command {
  docker run --rm --gpus all --network none --cpus 4 --memory 16g --shm-size 2g `
    -v "${PWD}/data/input:/app/data/input:ro" `
    -v "${PWD}/data/output:/app/data/output" `
    sinaptic5g-ftr:canonical
}
python src/qod_atexit_patch.py data/output/results.json
docker save sinaptic5g-ftr:canonical -o ..\output\sinaptic5g-ftr-canonical.tar
```

- [ ] İmaj `nvidia/cuda:12.1.0-base-ubuntu22.04` tabanlı.
- [ ] İmaj boyutu 8 GB'dan küçük.
- [ ] Test bilgisayarında Tesla T4 veya eşdeğer CUDA GPU görünür (`nvidia-smi`).
- [ ] Girdi yalnız `data/input/video.mp4`; çıktı yalnız `data/output/results.json`.
- [ ] Çalışma ağ kapalıyken tamamlandı ve 10 dakika sınırının altında kaldı.
- [ ] Çıktı atomik üretildi, JSON geçerli ve olay zamanları video süresi içinde.
- [ ] Docker imajı `.tar` olarak yedeklendi.

### Android ve canlı sistem provası

```powershell
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
cd android
.\gradlew.bat clean :app:assembleDebug `
  -PAPI_BASE_URL=https://GERCEK-BFF-ADRESI `
  -PMEDIA_BASE_URL=https://GERCEK-MEDYA-ADRESI
cd ..
python scripts/scan_android_apk.py
```

- [ ] Gerçek HTTPS/WSS adresleriyle yeni APK üretildi; örnek `example` adresi kalmadı.
- [ ] APK içinde tam olarak bir TFLite model, sıfır ONNX ve sıfır gömülü sır bulundu.
- [ ] BFF ve medya servislerinin `/health` uçları yeşil.
- [ ] Android → BFF → WebRTC → GPU → telemetri akışı gerçek iki uçla denendi.
- [ ] ICE/TURN bağlantısı farklı operatör veya hotspot üzerinden denendi.
- [ ] Number Verification ve QoD yetkileri varsa gerçek hesapla; yoksa açıkça “harici erişim bekliyor” olarak gösterildi.
- [ ] Uygulama izinleri, ekran açık kalma ayarı, kamera yönü ve telefon sabitlemesi kontrol edildi.
- [ ] Bildirim/sessiz mod, otomatik güncelleme ve pil tasarrufu demo süresince kapatıldı.

## 3. Yarışma sabahı

- [ ] Kod, model, Android Studio, CUDA, Docker veya telefon işletim sistemi güncellenmedi.
- [ ] Bilgisayar ve telefonlar %100 şarj edildi; en az 20 GB boş disk alanı bırakıldı.
- [ ] Bilgisayar saatleri otomatik ve aynı zaman diliminde.
- [ ] VPN/proxy kapalı; gerekli güvenlik duvarı kuralları önceden doğrulandı.
- [ ] Teslim artefaktlarının SHA-256 değerleri kaydedildi:

```powershell
Get-FileHash -Algorithm SHA256 `
  ..\SİNAPTİC5G_FINAL_TASARIM_RAPORU.md, `
  ..\output\pdf\SINAPTIC5G_Final_Tasarim_Raporu.pdf, `
  model_lock.json, models\detector.onnx, yolov8n.onnx
```

- [ ] Son çalışan APK ve Docker `.tar` dosyası iki ayrı medyada.
- [ ] Örnek gündüz, gece, dikkat dağınıklığı ve plaka videoları çevrimdışı hazır.
- [ ] Sunum ve demo en az iki kez kronometreyle prova edildi.

## 4. Alana varınca

- [ ] Elektrik, sahne ekranı, çözünürlük, HDMI sesi ve internet test edildi.
- [ ] Yarışma GPU'sunda `nvidia-smi`, Docker GPU erişimi ve disk alanı kontrol edildi.
- [ ] Girdi videosunun adı tam olarak `video.mp4` yapıldı; orijinal dosya değiştirilmedi.
- [ ] Ayrı ve boş bir `data/output` klasörü kullanıldı.
- [ ] Canlı demo için kamera kadrajı, aydınlatma, telefon açısı ve bağlantı kalitesi kontrol edildi.
- [ ] Ekip içi “canlı demo / çevrimdışı yedek” geçiş cümlesi kararlaştırıldı.

## 5. Jüri anlatım sırası

- [ ] Problem: sürücü, yol ve araç risklerini aynı zaman çizgisinde birleştirme.
- [ ] Çözüm: Android kamera, WebRTC, 5G QoD, GPU çıkarımı, takip/agregasyon ve şema-doğrulanmış çıktı.
- [ ] 5G katkısı: düşük gecikmeli uplink, QoD talebi ve kopmada yerel/fallback davranışı.
- [ ] Veri: 14.352 görüntü, 18.590 anotasyon; kaynak ve lisans envanteri mevcut.
- [ ] Ölçülmüş sonuç: mAP50 `0,7457`, mAP50-95 `0,4710`, F1 `0,7714`.
- [ ] Sınırlar dürüstçe söylendi: bazı sınıflarda bağımsız test desteği yok; Türk plaka OCR CER ve T4 gecikmesi ayrıca ölçülmeli.
- [ ] Demo sonunda `results.json` ve olayların video üzerindeki karşılığı gösterildi.

## 6. Arıza planı

| Sorun | Hemen uygulanacak yol |
|---|---|
| Salon interneti/TURN çalışmıyor | Yerel hotspotu deneyin; canlı akış yerine kayıtlı video ile çevrimdışı FTR demosuna geçin. |
| Number Verification/QoD erişimi yok | Mock başarı göstermeyin; entegrasyon kodunu ve harici erişim bağımlılığını açıklayın. |
| Android telefon veya kablo arızası | Yedek telefon/kabloyu ve önceden kurulmuş APK'yı kullanın. |
| Docker imajı yeniden kurulamıyor | `sinaptic5g-ftr-canonical.tar` dosyasını `docker load -i` ile yükleyin. |
| GPU görünmüyor | Docker/NVIDIA runtime'ı kontrol edin; CPU fallback yalnız süre ölçülerek kullanılmalı. |
| Canlı model sonuç vermiyor | Model hash, giriş tensoru, kamera izni ve `/health` uçlarını sırayla kontrol edin. |
| Sunum bilgisayarı arızası | İkinci bilgisayar ve USB'deki PDF ile devam edin. |

## 7. Teslimden hemen önce

- [ ] Yalnız istenen kaynaklar ve artefaktlar teslim paketinde; veri seti, `.env`, cache, eğitim koşuları ve kişisel dosyalar pakete girmedi.
- [ ] PDF adı, takım adı `SinapticLink5G`, takım numarası `989764` ve başvuru numarası `5205583` doğru.
- [ ] Docker komutu, volume yolları, kaynak sınırları ve çıktı yolu birebir doğru.
- [ ] Teslim dosyası başka bilgisayarda açıldı; hash ve dosya boyutu kaydedildi.
- [ ] Portal yüklemesi tamamlandıktan sonra ekran görüntüsü ve zaman damgası saklandı.
- [ ] Teslim edilen son pakette hızlı bir virüs ve sır taraması yapıldı.

## 8. Değiştirilmemesi gereken kanonik dosyalar

- `SİNAPTİC5G_FINAL_TASARIM_RAPORU.md`
- `output/pdf/SINAPTIC5G_Final_Tasarim_Raporu.pdf`
- `5G PROJE/Dockerfile`
- `5G PROJE/ftr_main.py`
- `5G PROJE/model_lock.json`
- `5G PROJE/schemas/results.schema.json`
- `5G PROJE/models/detector.onnx`
- `5G PROJE/yolov8n.onnx`
- `5G PROJE/reports/training_dataset_manifest.yaml`
- `5G PROJE/dataset/LICENSE_INVENTORY.md`
