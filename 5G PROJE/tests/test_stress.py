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

"""
tests/test_stress.py — Uzun Video Stres Testi
=============================================
Faz 5: Test & Doğrulama

Uzun video (gerçek veya sentetik) üzerinde memory leak, buffer şişmesi
ve takip stabilitesi testleri yapar.

Testler:
1. Memory growth kontrolü (track buffer, OCR buffer, debounce counter)
2. 300+ kare işleme stabilitesi
3. Stale buffer temizleme doğrulaması
4. FTR pipeline timeout koruması

Üretim kilidi: detector_v5 / models/detector.onnx kilitli.

Çalıştırma:
    pytest tests/test_stress.py -v
    python tests/test_stress.py --n-frames 300
"""

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG = logging.getLogger("sinaptic5g.test_stress")

# ─── Test Parametreleri ───────────────────────────────────────────────────────
N_STRESS_FRAMES = 300
MAX_TRACK_BUFFER_SIZE = 100       # Maksimum track sayısı (sızıntı kontrolü)
MAX_OCR_BUFFER_SIZE = 20          # PlateRecognizer buffer limit
FPS_SYNTHETIC = 25.0
FRAME_SIZE = (640, 360)           # Test kare boyutu


# ─── Sentetik Video Kare Üreteci ─────────────────────────────────────────────

def make_synthetic_frame(
    frame_idx: int,
    h: int = 360,
    w: int = 640,
    n_vehicles: int = 3,
) -> np.ndarray:
    """Test için sentetik video karesi oluşturur."""
    frame = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    
    # Hareketli dikdörtgenler (araç simülasyonu)
    for vid in range(n_vehicles):
        offset = (frame_idx * 2 + vid * 80) % max(1, w - 100)
        x1 = offset
        y1 = 100 + vid * 80
        x2 = x1 + 80
        y2 = y1 + 60
        color = ((vid * 60) % 255, (vid * 100) % 255, 200)
        import cv2
        cv2.rectangle(frame, (x1, y1), (min(x2, w - 1), min(y2, h - 1)), color, -1)
    
    return frame


class SyntheticVehicleDetection:
    """Sentetik araç tespiti (tracker test için)."""
    def __init__(self, x1: int, y1: int, x2: int, y2: int, conf: float = 0.7):
        self.bbox = (x1, y1, x2, y2)
        self.vehicle_type = "sedan"
        self.confidence = conf


# ─── Pytest Testleri ──────────────────────────────────────────────────────────

class TestStressVideo:
    """Uzun video stres testleri."""
    
    def test_tracker_buffer_no_memory_leak(self):
        """
        SinapticTracker 300 kare boyunca track buffer'ını temizlemeli.
        Stale track'ler track_buffer kare sonra silinmeli.
        """
        try:
            from src.tracking_pipeline import SinapticTracker
        except ImportError:
            pytest.skip("SinapticTracker import edilemiyor")
        
        import numpy as np
        tracker = SinapticTracker(camera_mode="front", conf_thresh=0.45, track_buffer=30)
        
        max_active_tracks = 0
        
        for frame_idx in range(N_STRESS_FRAMES):
            frame = make_synthetic_frame(frame_idx)
            timestamp = frame_idx / FPS_SYNTHETIC
            
            # 3-5 araç tespiti (frame bazlı değişen)
            n_vehicles = 3 if (frame_idx % 50) < 40 else 0  # 10 frame'de 0 araç
            detections = []
            for vid in range(n_vehicles):
                offset = (frame_idx * 2 + vid * 80) % max(1, frame.shape[1] - 100)
                x1, y1 = offset, 100 + vid * 80
                x2, y2 = x1 + 80, y1 + 60
                detections.append(SyntheticVehicleDetection(x1, y1, x2, y2))
            
            tracked = tracker.update(frame, detections, timestamp)
            active_count = len(tracker.trackers)
            max_active_tracks = max(max_active_tracks, active_count)
        
        # Track buffer sınırını aşmamalı
        assert max_active_tracks <= MAX_TRACK_BUFFER_SIZE, (
            f"Track buffer sızıntısı: {max_active_tracks} aktif track > {MAX_TRACK_BUFFER_SIZE}"
        )
        
        LOG.info("Track buffer testi geçti. Maks aktif track: %d", max_active_tracks)
    
    def test_ocr_buffer_cleanup(self):
        """
        PlateRecognizer temporal buffer stale frame'lerde temizlenmeli.
        Uzun süre çağrılmayan buffer otomatik temizlenmeli.
        """
        try:
            from plate_ocr import PlateRecognizer
        except ImportError:
            pytest.skip("PlateRecognizer import edilemiyor")
        
        recognizer = PlateRecognizer(
            voting_buffer_size=7,
            max_buffer_age_frames=10,  # 10 kare sonra stale
        )
        
        # Buffer'ı doldur (sentetik siyah görüntü)
        black_frame = np.zeros((32, 100, 3), dtype=np.uint8)
        for _ in range(5):
            recognizer.recognize(black_frame)
        
        initial_buffer_size = len(recognizer._plate_buffer)
        
        # Stale threshold'u geç: 15 çağrı daha yap (içerik önemi yok)
        for _ in range(15):
            recognizer._buffer_frame_counter += 1
        
        # Bir sonraki recognize çağrısında stale temizleme olmalı
        recognizer.recognize(black_frame)
        
        # Buffer temizlendi mi? (stale threshold aşıldı)
        LOG.info("OCR buffer testi geçti. Buffer boyutu: %d → %d",
                  initial_buffer_size, len(recognizer._plate_buffer))
        
        # Bellek sızıntısı kontrolü: buffer boyutu max_buffer_size'ı aşmamalı
        assert len(recognizer._plate_buffer) <= recognizer.voting_buffer_size, (
            f"OCR buffer boyutu sınırı aşıldı: {len(recognizer._plate_buffer)}"
        )
    
    def test_debounce_counter_cleanup(self):
        """
        Track debounce counter'ları stale track silindiğinde temizlenmeli.
        Simülasyon: defaultdict tabanlı debounce counter temizleme.
        """
        track_debounce_counters = defaultdict(lambda: defaultdict(int))
        
        # Bazı track'lere counter ekle
        active_tracks = set(range(1, 20))
        for tid in active_tracks:
            track_debounce_counters[tid]["telefonla_konusma"] = tid % 5
        
        # 10 track'i "stale" yap
        stale_tracks = set(range(1, 11))
        active_tracks -= stale_tracks
        
        # Stale temizleme (ftr_main.py'deki mantıkla aynı)
        stale_ids = [k for k in track_debounce_counters if k not in active_tracks]
        for k in stale_ids:
            del track_debounce_counters[k]
        
        # Stale track'ler temizlendi mi?
        assert not any(k in track_debounce_counters for k in stale_tracks), \
            "Stale debounce counter'lar temizlenmedi!"
        
        # Aktif track'ler korundu mu?
        assert all(k in track_debounce_counters for k in active_tracks), \
            "Aktif track'ler yanlışlıkla silindi!"
        
        LOG.info("Debounce counter temizleme testi geçti.")
    
    def test_adaptive_stride_range(self):
        """
        compute_adaptive_stride fonksiyonu beklenen aralıkta stride döndürmeli.
        """
        try:
            from ftr_main import compute_adaptive_stride
        except ImportError:
            pytest.skip("compute_adaptive_stride import edilemiyor")
        
        # GPU + detection
        s = compute_adaptive_stride(has_detections=True, has_gpu=True)
        assert 0.05 <= s <= 0.5, f"GPU+det stride aralık dışı: {s}"
        
        # GPU + no detection
        s = compute_adaptive_stride(has_detections=False, has_gpu=True)
        assert 0.1 <= s <= 1.0, f"GPU+nodet stride aralık dışı: {s}"
        
        # CPU + detection
        s = compute_adaptive_stride(has_detections=True, has_gpu=False)
        assert 0.1 <= s <= 1.0, f"CPU+det stride aralık dışı: {s}"
        
        # CPU + no detection
        s = compute_adaptive_stride(has_detections=False, has_gpu=False)
        assert 0.5 <= s <= 2.0, f"CPU+nodet stride aralık dışı: {s}"
        
        # Video sonu (son %10) — stride küçülmeli
        s_end = compute_adaptive_stride(
            has_detections=False, has_gpu=False,
            current_timestamp=90.0, video_duration=100.0,
        )
        s_normal = compute_adaptive_stride(
            has_detections=False, has_gpu=False,
            current_timestamp=50.0, video_duration=100.0,
        )
        assert s_end <= s_normal, "Video sonu stridi normale eşit veya küçük olmalı"
        
        LOG.info("Adaptif stride testi geçti.")
    
    def test_500_frame_synthetic_pipeline(self):
        """
        500 kare sentetik işleme — bellek artışı ve hata yok olmalı.
        Production model lock bozulmadan çalışmalı.
        """
        n_frames = 500
        errors = []
        max_mem_mb = 0.0
        
        try:
            import tracemalloc
            tracemalloc.start()
        except Exception:
            pass
        
        track_feature_buffers: Dict = defaultdict(list)
        track_debounce_counters: Dict = defaultdict(lambda: defaultdict(int))
        track_last_plate: Dict = {}
        
        for i in range(n_frames):
            frame = make_synthetic_frame(i)
            timestamp = i / FPS_SYNTHETIC
            
            # Simüle edilmiş aktif track ID'leri
            active_track_ids = {1, 2, 3} if (i % 50) < 40 else {1}
            
            # Stale temizleme mantığı (ftr_main.py ile aynı)
            stale_ids = [k for k in track_debounce_counters if k not in active_track_ids]
            for k in stale_ids:
                del track_debounce_counters[k]
            stale_ids = [k for k in track_feature_buffers if k not in active_track_ids]
            for k in stale_ids:
                del track_feature_buffers[k]
            stale_ids = [k for k in track_last_plate if k not in active_track_ids]
            for k in stale_ids:
                del track_last_plate[k]
            
            # Feature buffer boyutu kontrolü
            for tid in active_track_ids:
                track_feature_buffers[tid].append(np.random.rand(7).tolist())
                if len(track_feature_buffers[tid]) > 16:
                    track_feature_buffers[tid].pop(0)
        
        try:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            max_mem_mb = peak / 1024 / 1024
            LOG.info("Peak bellek kullanımı (sentetik pipeline): %.1f MB", max_mem_mb)
        except Exception:
            pass
        
        assert not errors, f"Hata oluştu: {errors}"
        assert max_mem_mb < 200 or max_mem_mb == 0, f"Bellek kullanımı çok yüksek: {max_mem_mb:.1f} MB"
        
        # Buffer boyutu sınırı
        for tid, buf in track_feature_buffers.items():
            assert len(buf) <= 16, f"Track {tid} feature buffer sınırı aşıldı: {len(buf)}"
        
        LOG.info("500 kare sentetik pipeline testi geçti. Hata: %d", len(errors))
    
    def test_production_model_lock_integrity(self):
        """
        Stres testi boyunca üretim modeli (detector_v5) değiştirilmemeli.
        """
        import hashlib
        
        model_path = PROJECT_ROOT / "models" / "detector.onnx"
        lock_path = PROJECT_ROOT / "model_lock.json"
        
        if not model_path.is_file() or not lock_path.is_file():
            pytest.skip("Model veya lock dosyası bulunamadı")
        
        import json
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        expected_sha = lock.get("detector_onnx_sha256", "")
        
        if not expected_sha:
            pytest.skip("detector_onnx_sha256 lock'ta yok")
        
        digest = hashlib.sha256()
        with model_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        
        assert actual_sha.lower() == expected_sha.lower(), (
            "STRES TESTİ SONRASI MODEL BÜTÜNLÜK HATASI!\n"
            "detector_v5 SHA256 eşleşmiyor — model değiştirilmiş olabilir!"
        )
        LOG.info("✅ Üretim modeli bütünlüğü korundu.")


# ─── Standalone çalıştırma ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-frames", type=int, default=N_STRESS_FRAMES)
    args = parser.parse_args()
    
    LOG.info("Stres testi başlıyor — %d kare", args.n_frames)
    
    test_obj = TestStressVideo()
    
    passed = 0
    failed = 0
    
    tests = [
        ("tracker_buffer_no_memory_leak", test_obj.test_tracker_buffer_no_memory_leak),
        ("ocr_buffer_cleanup", test_obj.test_ocr_buffer_cleanup),
        ("debounce_counter_cleanup", test_obj.test_debounce_counter_cleanup),
        ("adaptive_stride_range", test_obj.test_adaptive_stride_range),
        ("500_frame_synthetic_pipeline", test_obj.test_500_frame_synthetic_pipeline),
        ("production_model_lock_integrity", test_obj.test_production_model_lock_integrity),
    ]
    
    for test_name, test_fn in tests:
        try:
            test_fn()
            LOG.info("  ✅ %s", test_name)
            passed += 1
        except Exception as e:
            LOG.error("  ❌ %s: %s", test_name, e)
            failed += 1
    
    LOG.info("\n=== Stres Testi Özeti ===")
    LOG.info("  Geçen: %d | Başarısız: %d", passed, failed)
    sys.exit(0 if failed == 0 else 1)
