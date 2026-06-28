# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)
#
# Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
# amaçlı olarak paylaşılmıştır.

# Resmî değerlendirme ortamıyla aynı CUDA tabanı; PyTorch gerekmez.
# Dockerfile: Teslim repo root dizininde yer alır.
# hakem komutu: docker build -t teknofest/<takim>:latest .
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libglib2.0-0 libgl1 libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılıkları önce kopyala (Docker layer cache için)
COPY 5G\ PROJE/requirements-ftr.txt /tmp/requirements-ftr.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements-ftr.txt \
    && rm -rf /tmp/requirements-ftr.txt

# Kaynak dosyalar — seçici COPY (image boyutunu minimize eder)
COPY 5G\ PROJE/ftr_main.py /app/main.py
COPY 5G\ PROJE/src/ /app/src/
COPY 5G\ PROJE/schemas/results.schema.json /app/schemas/results.schema.json
COPY 5G\ PROJE/model_lock.json /app/model_lock.json
COPY 5G\ PROJE/plate_ocr.py /app/plate_ocr.py
COPY 5G\ PROJE/driver_analyzer.py /app/driver_analyzer.py

# Model ağırlıkları (FTR çıkarımı için zorunlu)
COPY 5G\ PROJE/yolov8n.onnx /app/models/coco.onnx
COPY 5G\ PROJE/models/detector_optimized.onnx /app/models/detector_optimized.onnx
COPY 5G\ PROJE/models/lprnet.onnx /app/models/lprnet.onnx
COPY 5G\ PROJE/models/crnn.onnx /app/models/crnn.onnx

# Çalışma dizinleri oluştur ve izinleri ayarla
RUN mkdir -p /app/data/input /app/data/output \
    && chmod -R a+rX /app \
    && chmod a+rwX /app/data/output

# Hakem docker run komutu:
# docker run --rm --gpus all --network none \
#   -v <video.mp4>:/app/data/input/video.mp4 \
#   -v <output-dir>:/app/data/output \
#   teknofest/<takim>:latest
ENTRYPOINT ["python3", "/app/main.py"]
