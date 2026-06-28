# Demo Kanıt Tablosu

| Kanıt | Sonuç |
|---|---|
| Python test paketi | 32/32 geçti |
| Android `assembleDebug` | Geçti — 93.354.690 bayt |
| Model build doğrulaması | Geçti |
| APK TFLite sayısı | 1 |
| APK ONNX sayısı | 0 |
| APK yapılandırılmış secret eşleşmesi | 0 |
| Kaynak PT SHA-256 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| TFLite SHA-256 | `a5b9e6133a5e9af3357dd49bfd059408acaeabcf37f4e31ab3914c8dead1f501` |
| TFLite tensor | input `[1,640,640,3]` float32; output `[1,84,8400]` float32 |
| GPU ONNX smoke | input `[1,3,640,640]`; output `[1,84,8400]`; boş kare 55.2 ms |
| Gerçek Android↔GPU RTP ağ testi | HEDEF — dış TURN/deploy gerekli |
