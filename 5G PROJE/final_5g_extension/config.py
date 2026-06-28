"""
config.py — Sinaptic5G Merkezi Konfigürasyon Modülü
====================================================
Tüm parametreler, eşikler ve API yapılandırmaları bu modülden yönetilir.
Ortam değişkenleri .env dosyasından otomatik yüklenir.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# ─── Proje Kök Dizini ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent


# ─── YZ Modeli Konfigürasyonu ─────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """YOLOv8 ve diğer YZ model parametreleri."""

    # Model yolları
    yolo_light_path: str = os.getenv(
        "YOLO_LIGHT_MODEL_PATH", "models/yolov8n_tripwire.pt"
    )
    yolo_heavy_path: str = os.getenv(
        "YOLO_HEAVY_MODEL_PATH", "models/yolov8l_tripwire.pt"
    )
    zero_dce_path: str = "models/zero_dce.tflite"
    zero_dce_full_path: str = "models/zero_dce_saved_model"

    # Tespit eşikleri
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45

    # QoD tetikleme eşikleri
    plate_pixel_threshold: int = 50       # Plaka piksel genişliği eşiği
    distractor_conf_threshold: float = 0.6  # Dikkat dağıtıcı nesne güven eşiği
    togg_conf_threshold: float = 0.7      # TOGG araç tespit eşiği

    # Çıkarım parametreleri
    input_size: int = 640
    device: str = "cpu"  # "cpu", "0", "cuda"

    # Fine-tuning parametreleri
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    warmup_epochs: int = 3
    freeze_layers_nano: int = 5
    freeze_layers_large: int = 10

    # TripWire özel sınıfları (merkezi kaynak: src/class_registry.py)
    # Canlı pipeline: 6 sınıf (İngilizce)
    # FTR pipeline: 9 sınıf (Türkçe) — src.class_registry.FTR_CLASS_MAP
    class_names: list = field(default_factory=lambda: [
        "license_plate",  # 0
        "phone",          # 1
        "cigarette",      # 2
        "toy",            # 3
        "togg",           # 4
        "driver_face",    # 5
    ])

    # FTR pipeline sınıf-spesifik güven eşikleri
    # (src/class_registry.py FTR_CLASS_THRESHOLDS ile senkronize)
    ftr_class_thresholds: dict = field(default_factory=lambda: {
        "telefonla_konusma":      0.30,
        "su_icme":                0.40,
        "arkaya_bakma":           0.25,
        "esneme":                 0.25,
        "sigara_icme":            0.45,
        "emniyet_kemeri_ihlali":  0.30,
        "teknocan":               0.35,
        "bilgisayar":             0.30,
        "license_plate":          0.40,
    })
    ftr_default_threshold: float = 0.35



# ─── Sürücü Analizi Konfigürasyonu ───────────────────────────────────────────
@dataclass
class DriverConfig:
    """MediaPipe ve sürücü davranış analizi parametreleri."""

    # EAR (Eye Aspect Ratio) — Uyuklama tespiti
    ear_threshold: float = 0.25          # Bu değerin altı → göz kapandı
    drowsy_frame_count: int = 30         # Kaç kare uyuklamak = uyuklama uyarısı

    # MAR (Mouth Aspect Ratio) — Esneme tespiti
    mar_threshold: float = 0.60          # Bu değerin üstü → ağız açık

    # MediaPipe parametreleri
    max_num_hands: int = 2
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    max_num_faces: int = 1
    refine_landmarks: bool = True

    # El konumu eşiği (görüntünün üst %30'unda el varsa dikkat)
    face_level_threshold: float = 0.3

    # EMA için pencere boyutu
    ear_history_size: int = 10


# ─── Hız Tahmini Konfigürasyonu ───────────────────────────────────────────────
@dataclass
class SpeedConfig:
    """Farneback optik akış hız tahmini parametreleri."""

    fps: float = 30.0
    calibration_path: str = os.getenv("CAMERA_CALIBRATION_PATH", "")
    smoothing_window: int = 10

    # Farneback parametreleri
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2

    # Hız sınırı (risk skoru için)
    default_speed_limit_kmh: float = 50.0


# ─── Risk Skoru Konfigürasyonu ────────────────────────────────────────────────
@dataclass
class RiskConfig:
    """Risk skoru ağırlıkları ve eşikleri."""

    weights: dict = field(default_factory=lambda: {
        "speed_excess": 35,   # Hız ihlali (>20 km/s fazla)
        "phone":        30,   # Telefon kullanımı
        "drowsy":       25,   # Uyuklama
        "cigarette":    15,   # Sigara içme
        "no_plate":     10,   # Plaka tespit edilmedi
        "yawning":       8,   # Esneme
        "toy":          10,   # Yabancı nesne (oyuncak)
    })

    # Risk seviyeleri
    low_threshold: float = 30.0    # 0-30: DÜŞÜK
    medium_threshold: float = 70.0 # 30-70: ORTA
    # 70+: KRİTİK

    # İlk tespit bağışlama süresi (kare sayısı)
    grace_frames: int = 8


# ─── Sunucu Konfigürasyonu ────────────────────────────────────────────────────
@dataclass
class ServerConfig:
    """FastAPI sunucu parametreleri."""

    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))
    secret_key: str = os.getenv("SECRET_KEY", "")
    workers: int = 1  # WebSocket için tek worker
    reload: bool = False
    # CORS
    allowed_origins: list = field(default_factory=lambda: [
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGIN", "").split(",")
        if origin.strip()
    ])

    # WebSocket
    ws_max_size: int = 10 * 1024 * 1024  # 10 MB maksimum mesaj boyutu

    # Rate limiting
    rate_limit_auth: str = "10/minute"
    rate_limit_ws: int = 30  # Frame/saniye limiti


# ─── Veritabanı Konfigürasyonu ────────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    """PostgreSQL ve Redis bağlantı parametreleri."""

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://tripwire:tripwire123@localhost:5432/tripwire_db"
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Redis TTL değerleri
    qod_session_ttl: int = 300  # 5 dakika
    token_cache_ttl: int = 3600  # 1 saat

    # PostgreSQL batch buffer
    batch_buffer_size: int = 150
    batch_flush_interval: float = 5.0  # saniye


@dataclass
class MediaConfig:
    """WebRTC signaling and GPU media-service settings."""

    service_token: str = os.getenv("MEDIA_SERVICE_TOKEN", "")
    signaling_ttl_seconds: int = int(os.getenv("WEBRTC_SIGNALING_TTL_SECONDS", "60"))
    media_base_url: str = os.getenv("MEDIA_BASE_URL", "")
    ice_servers_json: str = os.getenv("WEBRTC_ICE_SERVERS_JSON", "[]")


# ─── 5G CAMARA API Konfigürasyonu ─────────────────────────────────────────────
@dataclass
class CamaraConfig:
    """Turkcell Open Gateway CAMARA API parametreleri."""

    # Confidential credentials stay on the backend QoD adapter. Number
    # Verification uses a short-lived user token supplied by the Android app.
    client_id: str = os.getenv("TURKCELL_QOD_CLIENT_ID", "")
    client_secret: str = os.getenv("TURKCELL_QOD_CLIENT_SECRET", "")

    token_url: str = os.getenv("TURKCELL_TOKEN_URL", "")
    number_verify_url: str = os.getenv("TURKCELL_NUMBER_VERIFY_URL", "")
    qod_base_url: str = os.getenv("TURKCELL_QOD_BASE_URL", "")

    # QoS Profilleri
    qos_profile: str = os.getenv("TURKCELL_QOD_PROFILE", "")
    application_server_ipv4: str = os.getenv("QOD_APPLICATION_SERVER_IPV4", "")
    application_server_port: int = int(os.getenv("QOD_APPLICATION_SERVER_PORT", "443"))
    device_port: int = int(os.getenv("QOD_DEVICE_PORT", "0"))
    transport_protocol: str = os.getenv("QOD_TRANSPORT_PROTOCOL", "TCP")
    notification_url: str = os.getenv("QOD_NOTIFICATION_URL", "")
    qod_duration_seconds: int = int(os.getenv("QOD_DURATION_SECONDS", "300"))
    benefit_model_path: str = os.getenv(
        "QOD_BENEFIT_MODEL_PATH", "configs/qod_benefit_model.json"
    )

    # OAuth token yönetimi
    oauth_safety_margin: int = 300  # 5 dakika önce yenile
    max_retries: int = 3
    initial_retry_delay: float = 1.0  # saniye
    max_retry_delay: float = 16.0    # saniye

    # Zaman aşımı
    api_timeout: float = 10.0
    verify_timeout: float = 5.0


# ─── Veri Seti Konfigürasyonu ─────────────────────────────────────────────────
@dataclass
class DatasetConfig:
    """Eğitim veri seti parametreleri."""

    base_dir: str = "dataset"
    images_dir: str = "dataset/images"
    labels_dir: str = "dataset/labels"
    data_yaml: str = "dataset/data.yaml"

    # Frame çıkarma
    sample_rate: int = 5          # Her 5. kareyi al
    target_size: tuple = (640, 640)
    jpeg_quality: int = 95

    # Veri bölme
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Augmentation
    augmentation: dict = field(default_factory=lambda: {
        "rotation": {"min": -15, "max": 15},
        "brightness": {"min": -40, "max": 40},
        "blur": {"max_pixels": 2},
        "hue_shift": {"min": -15, "max": 15},
        "flip": "horizontal",
        "cutout": {"count": 3, "percent": 10},
    })


# ─── Çıktı Konfigürasyonu ────────────────────────────────────────────────────
@dataclass
class OutputConfig:
    """Çıktı dosyaları ve loglama parametreleri."""

    output_dir: str = "output"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Video çıktısı
    output_fps: float = 30.0
    output_codec: str = "mp4v"


# ─── Ana Konfigürasyon Nesnesi ────────────────────────────────────────────────
class Config:
    """Tüm konfigürasyon bileşenlerini bir araya getirir."""

    def __init__(self):
        self.model = ModelConfig()
        self.driver = DriverConfig()
        self.speed = SpeedConfig()
        self.risk = RiskConfig()
        self.server = ServerConfig()
        self.database = DatabaseConfig()
        self.media = MediaConfig()
        self.camara = CamaraConfig()
        self.dataset = DatasetConfig()
        self.output = OutputConfig()

    def validate(self) -> list[str]:
        """
        Kritik yapılandırma değerlerini doğrular.
        Sorunları string listesi olarak döner.
        """
        issues = []

        if len(self.server.secret_key) < 32:
            issues.append("SECRET_KEY eksik veya 32 karakterden kısa — oturum üretimi kapalı")

        if "*" in self.server.allowed_origins:
            issues.append("ALLOWED_ORIGIN=* üretim için uygun değil — açık origin listesi kullanın")

        if len(self.media.service_token) < 32:
            issues.append("MEDIA_SERVICE_TOKEN eksik veya 32 karakterden kısa — WebRTC signaling kapalı")

        if not 1 <= self.media.signaling_ttl_seconds <= 300:
            issues.append("WEBRTC_SIGNALING_TTL_SECONDS 1-300 aralığında olmalı")

        if not self.camara.token_url or not self.camara.qod_base_url:
            issues.append("Turkcell token/QoD onboarding URL'leri eksik — QoD devre dışı")

        if not self.camara.number_verify_url:
            issues.append("TURKCELL_NUMBER_VERIFY_URL eksik — numara doğrulama devre dışı")

        if not self.camara.client_id:
            issues.append(
                "TURKCELL_QOD_CLIENT_ID eksik — QoD backend adaptörü devre dışı"
            )

        if not self.camara.client_secret:
            issues.append(
                "TURKCELL_QOD_CLIENT_SECRET eksik — QoD backend adaptörü devre dışı"
            )

        if not self.camara.qos_profile:
            issues.append(
                "TURKCELL_QOD_PROFILE eksik — onboarding profili olmadan QoD açılmaz"
            )

        if not self.camara.application_server_ipv4:
            issues.append("QOD_APPLICATION_SERVER_IPV4 eksik — QoD flow tanımı oluşturulamaz")

        light_path = Path(self.model.yolo_light_path)
        if not light_path.exists():
            issues.append(
                f"Hafif YZ modeli bulunamadı: {self.model.yolo_light_path} — "
                "hash ve model kartıyla doğrulanmış artefaktı models/ dizinine kurun"
            )

        return issues

    def __str__(self) -> str:
        return (
            f"Sinaptic5G Config | "
            f"Server: {self.server.host}:{self.server.port} | "
            f"Model: {self.model.yolo_light_path}"
        )


# Global konfigürasyon instance'ı
cfg = Config()


if __name__ == "__main__":
    import json
    issues = cfg.validate()
    print("=" * 60)
    print("  Sinaptic5G — Konfigürasyon Doğrulama")
    print("=" * 60)

    if issues:
        print("\n⚠️  Yapılandırma Uyarıları:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✅ Tüm yapılandırmalar doğru!")

    print(f"\n📋 {cfg}")
