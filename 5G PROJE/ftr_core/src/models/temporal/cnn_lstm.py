"""
cnn_lstm.py — Sinaptic5G Temporal 1D CNN + LSTM Sınıflandırıcı
===============================================================
16–32 frame'lik yörünge/özellik dizisini işleyerek sürücü
davranışını sınıflandıran hibrit 1D CNN + BiLSTM modeli.

Model Mimarisi:
    Input: [B, seq_len, feature_dim]   (örn: [8, 16, 7])
    ↓
    Conv1D Encoder (3 katman): temporal yerel örüntüler
    ↓
    LSTM Decoder (2 katman): uzun vadeli bağımlılıklar
    ↓
    Attention Pooling: hangi frame'ler önemli?
    ↓
    FC Classifier: 5 sınıf (normal / sigara / telefon / bakış / dikkatsizlik)

Giriş Özellikleri (7 boyut):
    0: EAR (Eye Aspect Ratio)
    1: MAR (Mouth Aspect Ratio)
    2: yörünge hızı (px/frame, normalize)
    3: yön değişim açısı (derece / 180)
    4: has_phone (0 veya 1)
    5: has_cigarette (0 veya 1)
    6: hız fazlası (km/s, normalize)

Kullanım:
    from src.models.temporal.cnn_lstm import TemporalCNNLSTM

    model = TemporalCNNLSTM()
    logits = model(feature_seq)  # [B, 5]
    probs  = torch.softmax(logits, dim=-1)
"""

import logging
from typing import Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    class MockTensor:
        pass
    class MockModule:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to use this class but it is not installed in this environment.")
    class MockNN:
        Module = MockModule
    class MockTorch:
        Tensor = MockTensor
        
    torch = MockTorch()
    nn = MockNN()
    F = None

logger = logging.getLogger(__name__)

# Sürücü davranış sınıf tanımları
BEHAVIOR_CLASSES = [
    "normal_surus",       # 0
    "sigara_icme",        # 1
    "telefon_konusma",    # 2
    "sag_sol_bakis",      # 3
    "dikkatsizlik",       # 4
]

# Giriş özellik boyutu
FEATURE_DIM = 7


class TemporalAttentionPool(nn.Module):
    """Temporal Attention Pooling.

    LSTM çıkışlarından hangi frame'lerin daha önemli olduğunu
    öğrenen ağırlıklı ortalama.

    Kullanım:
        pool = TemporalAttentionPool(hidden_size=128)
        context = pool(lstm_out)  # [B, 128]
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: LSTM çıkışı [B, seq_len, hidden_size].

        Returns:
            Ağırlıklı bağlam vektörü [B, hidden_size].
        """
        # Attention skorları
        scores = self.attention(lstm_out)  # [B, seq_len, 1]
        weights = F.softmax(scores, dim=1)  # [B, seq_len, 1]

        # Ağırlıklı toplam
        context = (lstm_out * weights).sum(dim=1)  # [B, hidden_size]
        return context


class TemporalCNNLSTM(nn.Module):
    """Temporal sürücü davranış sınıflandırıcısı.

    1D CNN → LSTM → Attention → FC mimarisi.
    16–32 frame'lik özellik dizisini 5 sınıfa sınıflandırır.

    ~280K parametre, eğitim: 2–5 dakika (CPU), <30 saniye (GPU).
    Çıkarım gecikmesi: <2ms (batch=1, CPU).

    Kullanım:
        model = TemporalCNNLSTM()
        x = torch.rand(8, 16, 7)  # [B=8, seq=16, feat=7]
        logits = model(x)          # [B, 5]
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        cnn_channels: list[int] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        cnn_dropout: float = 0.2,
        num_classes: int = 5,
        bidirectional: bool = False,
    ) -> None:
        """
        Args:
            feature_dim: Giriş özellik boyutu (varsayılan: 7).
            cnn_channels: CNN kanal boyutları (varsayılan: [64, 128, 256]).
            lstm_hidden: LSTM gizli durum boyutu.
            lstm_layers: LSTM katman sayısı.
            lstm_dropout: LSTM dropout oranı.
            cnn_dropout: CNN dropout oranı.
            num_classes: Çıkış sınıf sayısı.
            bidirectional: Çift yönlü LSTM kullan mı?
        """
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        self.feature_dim = feature_dim
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional

        # ── 1D CNN Encoder ────────────────────────────────────────────────────
        cnn_layers = []
        in_ch = feature_dim
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(cnn_dropout),
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_channels = cnn_channels[-1]

        # ── LSTM Decoder ──────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_size = lstm_hidden * (2 if bidirectional else 1)

        # ── Attention Pooling ─────────────────────────────────────────────────
        self.attention_pool = TemporalAttentionPool(lstm_out_size)

        # ── Classifier Head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier ağırlık başlatma."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş.

        Args:
            x: Özellik dizisi [B, seq_len, feature_dim].

        Returns:
            Sınıf logitleri [B, num_classes].
        """
        # CNN: [B, seq_len, feat] → [B, feat, seq_len] (Conv1d formatı)
        x_cnn = x.permute(0, 2, 1)        # [B, feature_dim, seq_len]
        x_cnn = self.cnn(x_cnn)           # [B, cnn_out, seq_len]
        x_cnn = x_cnn.permute(0, 2, 1)    # [B, seq_len, cnn_out]

        # LSTM
        lstm_out, _ = self.lstm(x_cnn)    # [B, seq_len, lstm_hidden]

        # Attention pooling
        context = self.attention_pool(lstm_out)  # [B, lstm_hidden]

        # Sınıflandırma
        logits = self.classifier(context)  # [B, num_classes]
        return logits

    def predict(
        self,
        features: torch.Tensor,
        return_probs: bool = True,
    ) -> tuple[int, float]:
        """Tek örnek için tahmin yapar.

        Args:
            features: [seq_len, feature_dim] veya [1, seq_len, feature_dim].
            return_probs: Olasılıkları kullanarak güven hesapla.

        Returns:
            (predicted_class_idx, confidence) tuple'ı.
        """
        self.eval()
        with torch.no_grad():
            if features.dim() == 2:
                features = features.unsqueeze(0)  # Batch boyutu ekle

            logits = self(features)

            if return_probs:
                probs = F.softmax(logits, dim=-1)
                confidence, pred_class = probs.max(dim=-1)
                return pred_class.item(), confidence.item()
            else:
                pred_class = logits.argmax(dim=-1)
                return pred_class.item(), 1.0


class FeatureExtractor:
    """Video karesinden temporal model giriş özelliklerini çıkarır.

    Her frame için 7 boyutlu özellik vektörü üretir:
        [EAR, MAR, speed_norm, angle_norm, has_phone, has_cigarette, speed_excess_norm]

    Kullanım
    """

    def __init__(
        self,
        max_speed_px: float = 30.0,
        max_angle_deg: float = 180.0,
        max_speed_excess_kmh: float = 50.0,
    ) -> None:
        self.max_speed_px = max_speed_px
        self.max_angle_deg = max_angle_deg
        self.max_speed_excess_kmh = max_speed_excess_kmh

    def extract(
        self,
        ear: float,
        mar: float,
        speed_px: float = 0.0,
        angle_deg: float = 0.0,
        has_phone: bool = False,
        has_cigarette: bool = False,
        speed_kmh: Optional[float] = None,
        speed_limit_kmh: float = 50.0,
    ) -> Any:
        """Tek frame için özellik vektörü üretir.

        Args:
            ear: Eye Aspect Ratio (0.0 – 1.0).
            mar: Mouth Aspect Ratio (0.0 – 1.0).
            speed_px: Piksel hızı (px/frame).
            angle_deg: Yön açısı (derece).
            has_phone: Telefon tespiti var mı?
            has_cigarette: Sigara tespiti var mı?
            speed_kmh: Araç hızı (km/s). None → 0.
            speed_limit_kmh: Hız limiti.

        Returns:
            [7] boyutlu float32 tensor veya numpy array.
        """
        speed_excess = 0.0
        if speed_kmh is not None:
            excess = max(0.0, speed_kmh - speed_limit_kmh)
            speed_excess = min(1.0, excess / self.max_speed_excess_kmh)

        features = [
            min(1.0, max(0.0, ear)),                              # EAR
            min(1.0, max(0.0, mar)),                              # MAR
            min(1.0, speed_px / (self.max_speed_px + 1e-6)),     # hız (normalize)
            min(1.0, angle_deg / (self.max_angle_deg + 1e-6)),   # açı (normalize)
            float(has_phone),                                      # has_phone
            float(has_cigarette),                                  # has_cigarette
            speed_excess,                                          # hız fazlası
        ]

        try:
            import torch
            return torch.tensor(features, dtype=torch.float32)
        except ImportError:
            import numpy as np
            return np.array(features, dtype=np.float32)


from pathlib import Path
from typing import Tuple, Any

class ONNXTemporalClassifier:
    """ONNX Runtime wrapper for Temporal CNN-LSTM behaviour classifier."""

    def __init__(self, model_path: str = "models/cnn_lstm.onnx") -> None:
        self.session = None
        self.input_name = ""
        resolved_path = Path(model_path)
        if not resolved_path.is_file():
            resolved_path = Path("5G PROJE") / model_path
            if not resolved_path.is_file():
                resolved_path = Path("/app") / model_path
        
        if resolved_path.is_file():
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
                lstm_opts = ort.SessionOptions()
                lstm_opts.intra_op_num_threads = 1
                lstm_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                lstm_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                self.session = ort.InferenceSession(str(resolved_path), lstm_opts, providers=providers)
                self.input_name = self.session.get_inputs()[0].name
                logger.info("Loaded temporal LSTM ONNX model: %s", resolved_path)
            except Exception as e:
                logger.error("Failed to load cnn_lstm ONNX session: %s", e)
        else:
            logger.warning("cnn_lstm ONNX model file not found: %s", model_path)

    def predict(self, feature_seq: Any) -> Optional[Tuple[str, float]]:
        """Predicts class and confidence for a [16, 7] or [1, 16, 7] sequence."""
        if self.session is None:
            return None
        try:
            import numpy as np
            arr = np.asarray(feature_seq, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=0) # [1, 16, 7]
            
            outputs = self.session.run(None, {self.input_name: arr})
            logits = outputs[0][0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            cls_idx = int(np.argmax(probs))
            conf = float(probs[cls_idx])
            
            return BEHAVIOR_CLASSES[cls_idx], conf
        except Exception as e:
            logger.error("Temporal LSTM ONNX prediction failed: %s", e)
            return None

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    logging.basicConfig(level=logging.INFO)

    # Model oluştur
    model = TemporalCNNLSTM(
        feature_dim=FEATURE_DIM,
        cnn_channels=[64, 128, 256],
        lstm_hidden=128,
        lstm_layers=2,
        num_classes=5,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parametreleri: {param_count:,}")

    # Dummy input: batch=4, seq=16, features=7
    dummy = torch.rand(4, 16, FEATURE_DIM)
    logits = model(dummy)
    print(f"Giriş: {dummy.shape} → Çıkış: {logits.shape}")

    # Softmax olasılıkları
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    print(f"Tahminler: {[BEHAVIOR_CLASSES[i] for i in pred.tolist()]}")

    # FeatureExtractor testi
    extractor = FeatureExtractor()
    feat = extractor.extract(
        ear=0.22, mar=0.18,
        speed_px=6.3, angle_deg=15.0,
        has_phone=True, has_cigarette=False,
        speed_kmh=65.0, speed_limit_kmh=50.0
    )
    print(f"Özellik vektörü: {feat.tolist()}")

    # Single predict testi
    seq = feat.unsqueeze(0).repeat(16, 1)  # 16 frame
    cls_idx, conf = model.predict(seq)
    print(f"Tahmin: {BEHAVIOR_CLASSES[cls_idx]} (güven: {conf:.2%})")
