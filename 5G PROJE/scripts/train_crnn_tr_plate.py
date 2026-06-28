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
scripts/train_crnn_tr_plate.py — CRNN Türk Plaka Fine-Tune Script/Config
=========================================================================
Faz 3: Model Eğitimi Hazırlığı

Mevcut models/crnn.onnx modelini Türk plakaları için fine-tune eder.
GPU yoksa dry-run/smoke-run modu çalışır.

Mimarı:
- CRNN: CNN backbone + BiLSTM + CTC loss
- Girdi: 100×32 grayscale plaka crop
- Çıktı: Türk plaka karakterleri (CTC decoding)
- Vocabulary: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" (36 karakter + blank)

Kullanım:
    # GPU eğitimi:
    python scripts/train_crnn_tr_plate.py \
        --train-dir data/ocr_test/images \
        --label-dir data/ocr_test/labels \
        --output-dir models/runs/crnn_tr_plate \
        --epochs 30 --batch 32

    # Smoke run (GPU yok):
    python scripts/train_crnn_tr_plate.py --smoke-run --epochs 1 --batch 2

Üretim kilidi: models/crnn.onnx üzerine ASLA YAZMA. Output yolu farklı olacak.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

LOG = logging.getLogger("sinaptic5g.train_crnn")

# ─── Vocabulary ───────────────────────────────────────────────────────────────
BLANK_CHAR = "-"
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VOCABULARY = BLANK_CHAR + CHARSET  # index 0 = blank
NUM_CLASSES = len(VOCABULARY)      # 37

# ─── Model Parametreleri ──────────────────────────────────────────────────────
IMG_W = 100   # CRNN giriş genişliği
IMG_H = 32    # CRNN giriş yüksekliği
IMG_C = 1     # Grayscale

# ─── Üretim Kilidi ────────────────────────────────────────────────────────────
PRODUCTION_CRNN_PATH = Path("models/crnn.onnx")
PRODUCTION_CRNN_INT8_PATH = Path("models/crnn_int8.onnx")
LOCKED_PATHS = {PRODUCTION_CRNN_PATH, PRODUCTION_CRNN_INT8_PATH}


def _check_production_lock(output_dir: Path) -> None:
    """Üretim modellerinin üzerine yazılmasını önler."""
    output_dir_resolved = output_dir.resolve()
    for locked in LOCKED_PATHS:
        locked_resolved = locked.resolve()
        if output_dir_resolved == locked_resolved.parent:
            if locked_resolved.name in [p.name for p in output_dir.iterdir() if output_dir.exists()]:
                raise RuntimeError(
                    f"GÜVENLİK HATASI: Çıktı dizini üretim modelini içeriyor: {locked}\n"
                    f"models/runs/crnn_tr_plate/ gibi ayrı bir dizin kullanın."
                )


# ─── Veri Yükleyici ───────────────────────────────────────────────────────────

class TRPlateDataset:
    """
    Türk plaka OCR veri seti.
    Görüntü: data/ocr_test/images/*.jpg
    Etiket: data/ocr_test/labels/*.txt (tek satır plate_text)
    """
    
    def __init__(self, images_dir: Path, labels_dir: Path, augment: bool = False):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.augment = augment
        self.samples: List[Tuple[Path, str]] = []
        self._load()
    
    def _load(self) -> None:
        img_paths = sorted(
            list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        )
        for img_path in img_paths:
            label_path = self.labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            text = label_path.read_text(encoding="utf-8").strip().upper()
            # Sadece geçerli karakterler
            text = "".join(c for c in text if c in CHARSET)
            if text:
                self.samples.append((img_path, text))
        LOG.info("Dataset yüklendi: %d örnek", len(self.samples))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        
        img = cv2.resize(img, (IMG_W, IMG_H))
        
        if self.augment:
            img = self._augment(img)
        
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0  # [-1, 1]
        img = img[np.newaxis, ...]  # [1, H, W]
        return img, label
    
    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Basit augmentasyon — CLAHE, blur, noise."""
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img = clahe.apply(img)
        
        # Random blur (hafif)
        if np.random.random() < 0.3:
            k = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        
        # Random brightness
        delta = np.random.randint(-30, 30)
        img = np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)
        
        return img
    
    def text_to_labels(self, text: str) -> List[int]:
        """Metni CRNN label indekslerine çevirir."""
        return [VOCABULARY.index(c) for c in text if c in VOCABULARY]
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, List[str]]:
        """Batch döner: (images [B, 1, H, W], texts)"""
        images = []
        texts = []
        for idx in indices:
            img, text = self[idx]
            images.append(img)
            texts.append(text)
        return np.stack(images, axis=0), texts


# ─── CTC Greedy Decoder ───────────────────────────────────────────────────────

def ctc_greedy_decode(log_probs: np.ndarray) -> str:
    """
    CTC greedy decoding.
    log_probs: [seq_len, num_classes]
    """
    char_indices = np.argmax(log_probs, axis=-1)
    decoded = []
    prev = -1
    for idx in char_indices:
        if idx != 0 and idx != prev:
            decoded.append(VOCABULARY[idx])
        prev = idx
    return "".join(decoded)


def cer(pred: str, gt: str) -> float:
    """Character Error Rate (Levenshtein / len(gt))."""
    if not gt:
        return 0.0 if not pred else 1.0
    
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / m


# ─── ONNX Smoke Run ───────────────────────────────────────────────────────────

def smoke_run_onnx(crnn_path: Path, dataset: TRPlateDataset) -> Dict:
    """
    GPU olmadan ONNX modeli ile smoke test.
    Birkaç örnek üzerinde CER hesaplar.
    """
    LOG.info("Smoke run başlıyor: %s", crnn_path)
    results = {"status": "SMOKE_RUN", "model": str(crnn_path), "samples_tested": 0}
    
    if not crnn_path.is_file():
        LOG.warning("ONNX model bulunamadı: %s", crnn_path)
        results["status"] = "MODEL_NOT_FOUND"
        return results
    
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(crnn_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
    except Exception as e:
        LOG.warning("ONNX session açılamadı: %s", e)
        results["status"] = "ONNX_LOAD_FAILED"
        results["error"] = str(e)
        return results
    
    n_test = min(5, len(dataset))
    exact_matches = 0
    total_cer = 0.0
    
    for i in range(n_test):
        img, gt_text = dataset[i]
        img_batch = img[np.newaxis, ...]  # [1, 1, H, W]
        
        try:
            outputs = sess.run(None, {input_name: img_batch})
            log_probs = outputs[0]
            
            if log_probs.ndim == 3:
                if log_probs.shape[0] == 1:
                    seq = log_probs[0]
                else:
                    seq = log_probs[:, 0, :]
            else:
                seq = log_probs
            
            pred_text = ctc_greedy_decode(seq)
        except Exception as e:
            LOG.debug("Inference hatası: %s", e)
            pred_text = ""
        
        match = (pred_text == gt_text)
        c = cer(pred_text, gt_text)
        total_cer += c
        
        if match:
            exact_matches += 1
        
        LOG.info("  [%d] GT=%s PRED=%s MATCH=%s CER=%.3f", i, gt_text, pred_text, match, c)
    
    results["samples_tested"] = n_test
    results["exact_match_rate"] = exact_matches / max(n_test, 1)
    results["mean_cer"] = total_cer / max(n_test, 1)
    results["status"] = "SMOKE_PASSED"
    return results


# ─── PyTorch Eğitim İskeleti ──────────────────────────────────────────────────

def try_pytorch_training(
    dataset: TRPlateDataset,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    device_str: str = "cpu",
) -> Dict:
    """
    PyTorch CRNN fine-tune eğitim iskeleti.
    GPU yoksa sadece dry-run yapar (1 batch, 1 epoch).
    """
    results = {"status": "NOT_STARTED"}
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        LOG.warning("PyTorch bulunamadı — PyTorch eğitimi atlanıyor.")
        results["status"] = "PYTORCH_NOT_AVAILABLE"
        return results
    
    device = torch.device(device_str)
    LOG.info("PyTorch eğitim iskeleti — device=%s", device)
    
    # ─── CRNN Model Tanımı ─────────────────────────────────────────────────
    class CRNN(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
                nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
                nn.MaxPool2d((2, 1), (2, 1)),
                nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(),
            )
            self.rnn = nn.Sequential(
                nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True),
            )
            self.fc = nn.Linear(512, num_classes)
        
        def forward(self, x):
            # x: [B, 1, H, W]
            x = self.cnn(x)                         # [B, C, H', W']
            b, c, h, w = x.size()
            x = x.squeeze(2) if h == 1 else x.permute(0, 3, 2, 1).reshape(b, w, c * h)
            if x.dim() == 3:
                x, _ = self.rnn[0](x)               # [B, W, 512]
            x = self.fc(x)                          # [B, W, num_classes]
            x = x.permute(1, 0, 2)                  # [W, B, num_classes]
            return x.log_softmax(dim=2)
    
    model = CRNN(NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n = len(dataset)
    if n == 0:
        LOG.warning("Dataset boş — eğitim atlanıyor")
        results["status"] = "EMPTY_DATASET"
        return results
    
    best_cer = float("inf")
    
    for epoch in range(epochs):
        model.train()
        indices = list(range(n))
        np.random.shuffle(indices)
        
        total_loss = 0.0
        n_batches = 0
        
        for batch_start in range(0, n, batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            images, texts = dataset.get_batch(batch_indices)
            
            x = torch.FloatTensor(images).to(device)
            
            # CTC target hazırlama
            target_list = [dataset.text_to_labels(t) for t in texts]
            target_lengths = torch.IntTensor([len(t) for t in target_list])
            targets = torch.IntTensor([l for t in target_list for l in t])
            
            log_probs = model(x)
            T = log_probs.size(0)
            input_lengths = torch.full((x.size(0),), T, dtype=torch.int32)
            
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            
            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        LOG.info("Epoch [%d/%d] loss=%.4f", epoch + 1, epochs, avg_loss)
        
        # Checkpoint
        if (epoch + 1) % 5 == 0 or epoch == 0:
            ckpt_path = output_dir / f"crnn_tr_plate_ep{epoch + 1:03d}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            LOG.info("Checkpoint kaydedildi: %s", ckpt_path)
    
    # Final model kaydet
    final_path = output_dir / "crnn_tr_plate_best.pth"
    torch.save(model.state_dict(), final_path)
    LOG.info("Final model kaydedildi: %s", final_path)
    
    results["status"] = "COMPLETED"
    results["output_dir"] = str(output_dir)
    results["final_model"] = str(final_path)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="CRNN Türk Plaka Fine-Tune")
    parser.add_argument("--train-dir", type=Path, default=Path("data/ocr_test/images"))
    parser.add_argument("--label-dir", type=Path, default=Path("data/ocr_test/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/runs/crnn_tr_plate"))
    parser.add_argument("--crnn-model", type=Path, default=Path("models/crnn.onnx"),
                        help="Mevcut CRNN ONNX (smoke run için)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke-run", action="store_true",
                        help="Sadece ONNX smoke run yap (GPU gerekmez)")
    parser.add_argument("--report-path", type=Path, default=Path("reports/crnn_train_results.json"))
    
    args = parser.parse_args()
    
    # Üretim kilidi kontrolü
    _check_production_lock(args.output_dir)
    
    LOG.info("CRNN Türk Plaka Fine-Tune başlıyor...")
    LOG.info("  Üretim CRNN modeli: %s (DOKUNULMAZ)", PRODUCTION_CRNN_PATH)
    LOG.info("  Çıktı dizini: %s", args.output_dir)
    
    # Veri seti yükle
    dataset = TRPlateDataset(args.train_dir, args.label_dir, augment=not args.smoke_run)
    
    results = {}
    
    if args.smoke_run or len(dataset) == 0:
        LOG.info("Smoke run modu — ONNX model test ediliyor")
        results = smoke_run_onnx(args.crnn_model, dataset)
    else:
        LOG.info("PyTorch eğitim iskeleti çalıştırılıyor...")
        results = try_pytorch_training(
            dataset=dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            device_str=args.device,
        )
    
    # Rapor yaz
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Sonuç raporu: %s", args.report_path)
    LOG.info("Sonuç: %s", results.get("status", "UNKNOWN"))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
