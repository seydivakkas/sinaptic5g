"""
zero_dce.py — Sinaptic5G Zero-DCE Düşük Işık Geliştirme
=========================================================
Zero-Reference Deep Curve Estimation (Guo et al., CVPR 2020).
Parametrik ışık artırma eğrilerini piksel bazında tahmin eder.

LE Eğrisi:
    LE(I, x) = I(x) + α · I(x) · (1 − I(x))
    Burada I(x) giriş piksel yoğunluğu, α öğrenilebilir eğri parametresidir.

4 Loss Fonksiyonu:
    L_spa: Mekansal tutarlılık (komşu piksel fark karesi)
    L_exp: Pozlama kontrolü (ortalama → E=0.6)
    L_col: Renk tutarlılığı (RGB kanal dengesi)
    L_tvA: Pürüzsüzlük (Total Variation)

NOT: torch.nn.functional dışındaki deneysel PyTorch operatörleri
KULLANILMAZ — TFLite export uyumluluğu.

Kullanım:
    from src.models.low_light.zero_dce import ZeroDCENet, ZeroDCELoss

    model = ZeroDCENet()
    loss_fn = ZeroDCELoss()
    enhanced, curve_map = model(low_light_image)
    loss = loss_fn(low_light_image, enhanced, curve_map)
"""

import logging
from typing import Optional

import numpy as np
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


class ZeroDCENet(nn.Module):
    """Zero-DCE düşük ışık geliştirme ağı (Standart).

    ~79K parametre | CPU: ~39ms @ 256×256 | GPU/NPU: ~8ms
    PSNR (gece simülasyonu): 21.12 dB | Parlaklık artışı: 5.78×

    Gece testi sonuçları (50 epoch, 251 frame):
        - Karanlık baseline PSNR : 19.00 dB
        - Zero-DCE PSNR          : 21.12 dB (+2.12 dB)
        - Parlaklık artışı       : 5.78×

    Mimari:
        7 katmanlı hafif CNN → 3×n_iterations kanal eğri haritası üretir.
        Her iterasyonda LE eğrisi uygulanır.

    Edge (hız odaklı) için ZeroDCENetTiny kullanın (~20K param, ~12ms CPU).
    """

    def __init__(
        self,
        in_channels: int = 3,
        mid_channels: int = 32,
        n_iterations: int = 8,
    ) -> None:
        """
        Args:
            in_channels: Giriş kanal sayısı (RGB=3).
            mid_channels: Ara katman kanal sayısı.
            n_iterations: LE eğrisi tekrar sayısı.
        """
        super().__init__()
        self.n_iterations = n_iterations
        out_channels = in_channels * n_iterations  # 3 × 8 = 24

        # Hafif CNN — yalnızca nn.Conv2d ve nn.functional (TFLite uyumlu)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        # Skip connection ile birleştirme
        self.conv5 = nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(mid_channels * 2, out_channels, 3, 1, 1, bias=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """İleri geçiş.

        Args:
            x: Giriş görüntüsü [B, 3, H, W], değerler [0, 1] aralığında.

        Returns:
            (enhanced_image, curve_map) tuple'ı.
            enhanced_image: Geliştirilmiş görüntü [B, 3, H, W].
            curve_map: Eğri haritası [B, 24, H, W].
        """
        # Encoder
        f1 = F.relu(self.conv1(x))
        f2 = F.relu(self.conv2(f1))
        f3 = F.relu(self.conv3(f2))
        f4 = F.relu(self.conv4(f3))

        # Decoder (skip connections ile)
        f5 = F.relu(self.conv5(torch.cat([f4, f3], dim=1)))
        f6 = F.relu(self.conv6(torch.cat([f5, f2], dim=1)))

        # Eğri haritası (tanh → [-1, 1] aralığında α parametreleri)
        curve_map = torch.tanh(self.conv7(torch.cat([f6, f1], dim=1)))

        # LE eğrisini iteratif uygula
        # LE(I, x) = I(x) + α · I(x) · (1 − I(x))
        enhanced = x
        for i in range(self.n_iterations):
            alpha = curve_map[:, i * 3 : (i + 1) * 3, :, :]
            enhanced = enhanced + alpha * enhanced * (1 - enhanced)

        return enhanced, curve_map


# ─── Loss Fonksiyonları ───────────────────────────────────────────────────────


class SpatialConsistencyLoss(nn.Module):
    """L_spa — Mekansal Tutarlılık Kaybı.

    Komşu 4×4 piksel blokları arasındaki ortalama farkın karesi.
    Yapay kenar oluşumunu engeller.
    """

    def __init__(self, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        # Ortalama pooling kernel'ı
        kernel = torch.ones(1, 1, patch_size, patch_size) / (patch_size ** 2)
        self.register_buffer("kernel", kernel)

    def forward(
        self,
        original: torch.Tensor,
        enhanced: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            original: Orijinal görüntü [B, 3, H, W].
            enhanced: Geliştirilmiş görüntü [B, 3, H, W].

        Returns:
            Spatial consistency loss skaler.
        """
        # Luminans kanalı (Y = 0.299R + 0.587G + 0.114B)
        org_lum = 0.299 * original[:, 0:1] + 0.587 * original[:, 1:2] + 0.114 * original[:, 2:3]
        enh_lum = 0.299 * enhanced[:, 0:1] + 0.587 * enhanced[:, 1:2] + 0.114 * enhanced[:, 2:3]

        # Ortalama pooling
        org_pool = F.avg_pool2d(org_lum, self.patch_size, self.patch_size)
        enh_pool = F.avg_pool2d(enh_lum, self.patch_size, self.patch_size)

        # 4 komşu yönde fark
        d_org_lr = org_pool[:, :, :, :-1] - org_pool[:, :, :, 1:]  # Sol-sağ
        d_enh_lr = enh_pool[:, :, :, :-1] - enh_pool[:, :, :, 1:]

        d_org_ud = org_pool[:, :, :-1, :] - org_pool[:, :, 1:, :]  # Üst-alt
        d_enh_ud = enh_pool[:, :, :-1, :] - enh_pool[:, :, 1:, :]

        loss = (
            torch.mean((d_org_lr - d_enh_lr) ** 2)
            + torch.mean((d_org_ud - d_enh_ud) ** 2)
        )

        return loss


class ExposureControlLoss(nn.Module):
    """L_exp — Pozlama Kontrolü Kaybı.

    Bölge ortalamasının hedef E=0.6 değerine yakınsamasını sağlar.
    |Mean(Y) - E| kaybı.
    """

    def __init__(
        self,
        patch_size: int = 16,
        target_exposure: float = 0.6,
    ) -> None:
        """
        Args:
            patch_size: Bölge boyutu (16×16 piksel).
            target_exposure: Hedef pozlama değeri (E=0.6).
        """
        super().__init__()
        self.patch_size = patch_size
        self.target_exposure = target_exposure

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced: Geliştirilmiş görüntü [B, 3, H, W].

        Returns:
            Exposure control loss skaler.
        """
        # Luminans (gri tonlama ortalaması)
        lum = torch.mean(enhanced, dim=1, keepdim=True)
        # Bölge ortalamaları
        patches = F.avg_pool2d(lum, self.patch_size, self.patch_size)
        # Hedeften uzaklık
        loss = torch.mean(torch.abs(patches - self.target_exposure))
        return loss


class ColorConstancyLoss(nn.Module):
    """L_col — Renk Tutarlılığı Kaybı.

    RGB kanallarının birbirine eşitlenmesini sağlar.
    (|R-G| + |R-B| + |B-G|)
    Renk kaymasını (color shift) önler.
    """

    def forward(self, enhanced: torch.Tensor) -> torch.Tensor:
        """
        Args:
            enhanced: Geliştirilmiş görüntü [B, 3, H, W].

        Returns:
            Color constancy loss skaler.
        """
        mean_r = torch.mean(enhanced[:, 0, :, :], dim=(1, 2))
        mean_g = torch.mean(enhanced[:, 1, :, :], dim=(1, 2))
        mean_b = torch.mean(enhanced[:, 2, :, :], dim=(1, 2))

        loss = (
            torch.mean(torch.abs(mean_r - mean_g))
            + torch.mean(torch.abs(mean_r - mean_b))
            + torch.mean(torch.abs(mean_b - mean_g))
        )

        return loss


class TotalVariationLoss(nn.Module):
    """L_tvA — Total Variation Kaybı.

    Eğri haritasının (A) türevlerinin mutlak toplamı.
    Pürüzsüzlük sağlar, gürültüyü bastırır.

    NOT: Epsilon (1e-6) ile NaN önleme — backpropagation güvenliği.
    Sıfır gradyan durumunda sayısal stabilite sağlanır.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """
        Args:
            eps: Sayısal stabilite için epsilon değeri.
        """
        super().__init__()
        self.eps = eps

    def forward(self, curve_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            curve_map: Eğri haritası [B, C, H, W].

        Returns:
            Total variation loss skaler.

        NOT: Bu loss fonksiyonunun geri yayılımında NaN üretme ihtimalini
        önlemek için epsilon ile stabilize edilmiştir.
        """
        # Yatay türev (dx)
        dx = curve_map[:, :, :, :-1] - curve_map[:, :, :, 1:]
        # Dikey türev (dy)
        dy = curve_map[:, :, :-1, :] - curve_map[:, :, 1:, :]

        # Mutlak değer toplamı (epsilon ile stabilize)
        loss = (
            torch.mean(torch.sqrt(dx ** 2 + self.eps))
            + torch.mean(torch.sqrt(dy ** 2 + self.eps))
        )

        return loss


class ZeroDCELoss(nn.Module):
    """Birleşik Zero-DCE Loss fonksiyonu.

    L_total = w_spa * L_spa + w_exp * L_exp + w_col * L_col + w_tv * L_tvA

    Kullanım:
        loss_fn = ZeroDCELoss()
        loss = loss_fn(original, enhanced, curve_map)
    """

    def __init__(
        self,
        w_spa: float = 1.0,
        w_exp: float = 10.0,
        w_col: float = 5.0,
        w_tv: float = 200.0,
        target_exposure: float = 0.6,
        patch_size: int = 16,
    ) -> None:
        """
        Args:
            w_spa: Spatial loss ağırlığı.
            w_exp: Exposure loss ağırlığı.
            w_col: Color loss ağırlığı.
            w_tv: Total variation loss ağırlığı.
            target_exposure: Hedef pozlama (E=0.6).
            patch_size: Spatial/exposure patch boyutu.
        """
        super().__init__()
        self.w_spa = w_spa
        self.w_exp = w_exp
        self.w_col = w_col
        self.w_tv = w_tv

        self.l_spa = SpatialConsistencyLoss(patch_size=4)
        self.l_exp = ExposureControlLoss(patch_size=patch_size, target_exposure=target_exposure)
        self.l_col = ColorConstancyLoss()
        self.l_tv = TotalVariationLoss()

    def forward(
        self,
        original: torch.Tensor,
        enhanced: torch.Tensor,
        curve_map: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            original: Orijinal görüntü [B, 3, H, W].
            enhanced: Geliştirilmiş görüntü [B, 3, H, W].
            curve_map: Eğri haritası [B, C, H, W].

        Returns:
            (total_loss, loss_dict) tuple'ı.
            loss_dict: Her bir loss bileşeninin değeri.
        """
        loss_spa = self.l_spa(original, enhanced)
        loss_exp = self.l_exp(enhanced)
        loss_col = self.l_col(enhanced)
        loss_tv = self.l_tv(curve_map)

        total = (
            self.w_spa * loss_spa
            + self.w_exp * loss_exp
            + self.w_col * loss_col
            + self.w_tv * loss_tv
        )

        loss_dict = {
            "L_spa": loss_spa.item(),
            "L_exp": loss_exp.item(),
            "L_col": loss_col.item(),
            "L_tvA": loss_tv.item(),
            "L_total": total.item(),
        }

        return total, loss_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Model testi
    model = ZeroDCENet(n_iterations=8)
    loss_fn = ZeroDCELoss()

    # Düşük ışık test görüntüsü (batch=2, 3 kanal, 256×256)
    dummy_input = torch.rand(2, 3, 256, 256) * 0.3  # Karanlık görüntü

    enhanced, curve_map = model(dummy_input)
    total_loss, loss_breakdown = loss_fn(dummy_input, enhanced, curve_map)

    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Giriş: {dummy_input.shape} → Çıkış: {enhanced.shape}")
    print(f"Eğri haritası: {curve_map.shape}")
    print(f"Loss: {loss_breakdown}")

    # NaN kontrolü
    total_loss.backward()
    has_nan = any(
        torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None
    )
    print(f"NaN gradyan: {'⚠️ EVET' if has_nan else '✅ HAYIR'}")


# --- ZeroDCENetTiny (edge/mobile) ---
class ZeroDCENetTiny(ZeroDCENet):
    ''''' Zero-DCE Tiny: mid_channels=16, n_iterations=6 (~20K param, ~12ms CPU). '''''
    def __init__(self) -> None:
        super().__init__(in_channels=3, mid_channels=16, n_iterations=6)


import cv2
from pathlib import Path

def enhance(frame: np.ndarray, model_path: str = "models/zero_dce_lite.tflite") -> np.ndarray:
    """Enhance low-light frames using a TFLite Zero-DCE model.
    Bypassed if mean brightness >= 90 or if model/interpreter is missing.
    """
    import numpy as np
    if frame is None or frame.size == 0:
        return frame

    # Calculate mean brightness on grayscale representation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness >= 90:
        return frame

    # Try importing tflite runtime
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            tflite = None

    if tflite is None:
        logger.warning("TFLite runtime is not available; Zero-DCE bypassed")
        return frame

    # Resolve model path
    resolved_path = Path(model_path)
    if not resolved_path.is_file():
        # Fallback to alternative paths
        resolved_path = Path("5G PROJE") / model_path
        if not resolved_path.is_file():
            resolved_path = Path("/app") / model_path
            if not resolved_path.is_file():
                logger.warning("Zero-DCE TFLite model file not found: %s; bypassed", model_path)
                return frame

    try:
        interpreter = tflite.Interpreter(model_path=str(resolved_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Input shape expected is typically [1, 416, 416, 3] or [1, 3, 416, 416]
        in_shape = input_details[0]['shape']
        h_model, w_model = in_shape[1], in_shape[2]
        # In case the format is NCHW: [1, 3, H, W]
        is_nchw = (in_shape[1] == 3)
        if is_nchw:
            h_model, w_model = in_shape[2], in_shape[3]

        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, (w_model, h_model))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        input_data = rgb.astype(np.float32) / 255.0
        if is_nchw:
            input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        # Handle quantization if model expects integer type
        input_dtype = input_details[0]['dtype']
        if input_dtype in (np.int8, np.uint8):
            scale, zero_point = input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(input_dtype)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Dequantize if needed
        output_dtype = output_details[0]['dtype']
        if output_dtype in (np.int8, np.uint8):
            scale, zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - zero_point) * scale

        # Output extraction
        # Expected shape [1, H, W, 3] or [1, 3, H, W]
        if is_nchw:
            output_data = np.transpose(output_data[0], (1, 2, 0))
        else:
            output_data = output_data[0]

        # Rescale to 0-255 uint8 and convert to BGR
        output_rgb = (np.clip(output_data, 0.0, 1.0) * 255.0).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        return cv2.resize(output_bgr, (w_orig, h_orig))
    except Exception as e:
        logger.error("Zero-DCE enhancement failed: %s", e)
        return frame

