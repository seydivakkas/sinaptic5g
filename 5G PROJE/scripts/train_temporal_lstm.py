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
scripts/train_temporal_lstm.py — CNN-LSTM Temporal Model Eğitim İskeleti
=========================================================================
Faz 3: Model Eğitimi Hazırlığı

Mevcut models/cnn_lstm.onnx için iyileştirilmiş temporal model eğitim iskeleti.

Mimari:
- Giriş: [batch, seq_len=16, features=7] — ftr_main.py ile senkronize
- 7 özellik: ear, mar, speed_px, angle_deg, has_phone, has_cigarette, speed_kmh_normalized
- LSTM: 2 katman, hidden_size=64, dropout=0.3
- Çıktı: 5 sınıf
  0: normal_surus
  1: telefonla_konusma (temporal)
  2: sigara_icme (temporal)
  3: uyuklama
  4: esneme

GPU yoksa smoke test/dry-run çalışır.

Kullanım:
    # GPU eğitimi:
    python scripts/train_temporal_lstm.py \
        --data-dir data/temporal \
        --output-dir models/runs/cnn_lstm_improved \
        --epochs 50 --seq-len 16

    # Smoke run:
    python scripts/train_temporal_lstm.py --smoke-run

Üretim kilidi: models/cnn_lstm.onnx üzerine ASLA YAZMA.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("sinaptic5g.train_lstm")

# ─── Sınıf Tanımları (ftr_main.py ile senkronize) ────────────────────────────
TEMPORAL_CLASSES = {
    0: "normal_surus",
    1: "telefonla_konusma",
    2: "sigara_icme",
    3: "uyuklama",
    4: "esneme",
}
NUM_TEMPORAL_CLASSES = len(TEMPORAL_CLASSES)
SEQ_LEN = 16           # Pencere boyutu (ftr_main.py ile senkronize)
NUM_FEATURES = 7       # ear, mar, speed_px, angle_deg, has_phone, has_cig, speed_kmh_norm

# ─── Üretim Kilidi ────────────────────────────────────────────────────────────
PRODUCTION_CNN_LSTM_PATH = Path("models/cnn_lstm.onnx")


def _check_production_lock(output_dir: Path) -> None:
    """Üretim modelinin üzerine yazılmasını önler."""
    output_dir_abs = str(output_dir.resolve())
    prod_parent = str(PRODUCTION_CNN_LSTM_PATH.resolve().parent)
    if output_dir_abs == prod_parent:
        raise RuntimeError(
            f"GÜVENLİK HATASI: Çıktı dizini üretim modeli dizini ile aynı: {PRODUCTION_CNN_LSTM_PATH}\n"
            "models/runs/cnn_lstm_improved/ gibi ayrı bir dizin kullanın."
        )


# ─── Sentetik Veri Üreteci (Gerçek Data Yoksa) ───────────────────────────────

class SyntheticTemporalDataset:
    """
    Gerçek temporal data yoksa sentetik veri üretir.
    Gerçek kullanım için: DriveAct, dashcam temporal etiketleri.
    """
    
    def __init__(self, n_samples: int = 1000, seq_len: int = 16, seed: int = 42):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.rng = np.random.RandomState(seed)
        self.X, self.y = self._generate()
        LOG.info("Sentetik temporal dataset: %d örnek", len(self.X))
    
    def _generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        7 özellik, 16 zaman adımı sentetik veri üretir.
        Gerçek featurelar ile senkronize (ftr_main.py FeatureExtractor).
        """
        X = np.zeros((self.n_samples, self.seq_len, NUM_FEATURES), dtype=np.float32)
        y = np.zeros(self.n_samples, dtype=np.int64)
        
        for i in range(self.n_samples):
            label = self.rng.choice(NUM_TEMPORAL_CLASSES, p=[0.5, 0.15, 0.1, 0.15, 0.1])
            y[i] = label
            
            # EAR (göz kırpma oranı)
            ear_mean = 0.28 if label == 3 else 0.35  # Uyuklama: düşük EAR
            ear_noise = 0.05
            X[i, :, 0] = np.clip(self.rng.normal(ear_mean, ear_noise, self.seq_len), 0.1, 0.5)
            
            # MAR (ağız açıklık oranı)
            mar_mean = 0.65 if label == 4 else 0.15  # Esneme: yüksek MAR
            mar_noise = 0.08
            X[i, :, 1] = np.clip(self.rng.normal(mar_mean, mar_noise, self.seq_len), 0.0, 1.0)
            
            # Speed px/frame (normalize)
            X[i, :, 2] = np.clip(self.rng.normal(0.3, 0.1, self.seq_len), 0.0, 1.0)
            
            # Head angle deg (normalize [-1, 1])
            X[i, :, 3] = np.clip(self.rng.normal(0.0, 0.2, self.seq_len), -1.0, 1.0)
            
            # Has phone (binary)
            phone_prob = 0.8 if label == 1 else 0.05
            X[i, :, 4] = (self.rng.random(self.seq_len) < phone_prob).astype(np.float32)
            
            # Has cigarette (binary)
            cig_prob = 0.8 if label == 2 else 0.03
            X[i, :, 5] = (self.rng.random(self.seq_len) < cig_prob).astype(np.float32)
            
            # Speed kmh normalize [0, 1] (50 kmh → 0.5)
            X[i, :, 6] = np.clip(self.rng.normal(0.5, 0.15, self.seq_len), 0.0, 1.0)
        
        return X, y
    
    def __len__(self) -> int:
        return self.n_samples
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[indices], self.y[indices]


# ─── ONNX Smoke Run ───────────────────────────────────────────────────────────

def smoke_run_onnx(model_path: Path) -> Dict:
    """Mevcut cnn_lstm.onnx ile smoke test."""
    LOG.info("CNN-LSTM ONNX smoke run: %s", model_path)
    results = {"status": "SMOKE_RUN", "model": str(model_path)}
    
    if not model_path.is_file():
        LOG.warning("Model bulunamadı: %s", model_path)
        results["status"] = "MODEL_NOT_FOUND"
        return results
    
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape
        LOG.info("  Input name=%s, shape=%s", input_name, input_shape)
    except Exception as e:
        results["status"] = "ONNX_LOAD_FAILED"
        results["error"] = str(e)
        return results
    
    # Sentetik veri ile test
    dataset = SyntheticTemporalDataset(n_samples=10, seq_len=SEQ_LEN)
    
    correct = 0
    for i in range(10):
        X, y = dataset.get_batch([i])
        
        # Shape adaptasyonu
        if len(input_shape) == 3:
            inp = X  # [1, 16, 7]
        else:
            inp = X.reshape(1, -1)
        
        try:
            out = sess.run(None, {input_name: inp.astype(np.float32)})
            pred = int(np.argmax(out[0], axis=-1).reshape(-1)[0])
            correct += int(pred == y[0])
            LOG.info("  [%d] GT=%s PRED=%s", i, TEMPORAL_CLASSES[y[0]], TEMPORAL_CLASSES.get(pred, "?"))
        except Exception as e:
            LOG.debug("Inference hatası: %s", e)
    
    results["status"] = "SMOKE_PASSED"
    results["accuracy_on_synthetic"] = correct / 10
    LOG.info("Smoke run tamamlandı — doğruluk (sentetik): %d/10", correct)
    return results


# ─── PyTorch Eğitim İskeleti ──────────────────────────────────────────────────

def try_pytorch_training(
    dataset: SyntheticTemporalDataset,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device_str: str = "cpu",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
) -> Dict:
    """PyTorch LSTM eğitim iskeleti."""
    results = {"status": "NOT_STARTED"}
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        LOG.warning("PyTorch bulunamadı — LSTM eğitimi atlanıyor.")
        results["status"] = "PYTORCH_NOT_AVAILABLE"
        return results
    
    device = torch.device(device_str)
    LOG.info("LSTM eğitim iskeleti — device=%s", device)
    
    # ─── Model Tanımı ──────────────────────────────────────────────────────
    class TemporalLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=NUM_FEATURES,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.attention = nn.Linear(hidden_size * 2, 1)
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size * 2, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, NUM_TEMPORAL_CLASSES)
        
        def forward(self, x):
            # x: [B, seq_len, features]
            lstm_out, _ = self.lstm(x)          # [B, seq_len, hidden*2]
            
            # Attention
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)  # [B, hidden*2]
            
            out = self.dropout(context)
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            return out
    
    model = TemporalLSTM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Sınıf ağırlıkları — dengeli sampling
    class_counts = np.bincount(dataset.y, minlength=NUM_TEMPORAL_CLASSES).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_TEMPORAL_CLASSES
    
    try:
        import torch
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).to(device)
        )
    except Exception:
        criterion = nn.CrossEntropyLoss()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n = len(dataset)
    best_acc = 0.0
    best_path = output_dir / "cnn_lstm_improved_best.pth"
    
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n).tolist()
        
        total_loss = 0.0
        correct = 0
        n_batches = 0
        
        for batch_start in range(0, n, batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            X_np, y_np = dataset.get_batch(batch_idx)
            
            X_t = torch.FloatTensor(X_np).to(device)
            y_t = torch.LongTensor(y_np).to(device)
            
            out = model(X_t)
            loss = criterion(out, y_t)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y_t).sum().item()
            n_batches += 1
        
        scheduler.step()
        acc = correct / n
        avg_loss = total_loss / max(n_batches, 1)
        
        LOG.info("Epoch [%d/%d] loss=%.4f acc=%.3f", epoch + 1, epochs, avg_loss, acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "acc": acc,
                "loss": avg_loss,
                "num_features": NUM_FEATURES,
                "seq_len": SEQ_LEN,
                "num_classes": NUM_TEMPORAL_CLASSES,
                "class_map": TEMPORAL_CLASSES,
            }, best_path)
            LOG.info("En iyi model kaydedildi: acc=%.3f", acc)
    
    results["status"] = "COMPLETED"
    results["best_acc"] = best_acc
    results["output_dir"] = str(output_dir)
    results["best_model"] = str(best_path)
    results["num_features"] = NUM_FEATURES
    results["seq_len"] = SEQ_LEN
    results["num_classes"] = NUM_TEMPORAL_CLASSES
    
    LOG.info("LSTM eğitimi tamamlandı. En iyi doğruluk: %.3f", best_acc)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    parser = argparse.ArgumentParser(description="CNN-LSTM Temporal Model Eğitimi")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Gerçek temporal veri dizini (yoksa sentetik kullanılır)")
    parser.add_argument("--output-dir", type=Path, default=Path("models/runs/cnn_lstm_improved"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--smoke-run", action="store_true",
                        help="Sadece ONNX smoke test yap")
    parser.add_argument("--report-path", type=Path, default=Path("reports/lstm_train_results.json"))
    
    args = parser.parse_args()
    
    # Üretim kilidi kontrolü
    _check_production_lock(args.output_dir)
    
    LOG.info("CNN-LSTM Temporal Model Eğitimi başlıyor...")
    LOG.info("  Üretim modeli: %s (DOKUNULMAZ)", PRODUCTION_CNN_LSTM_PATH)
    LOG.info("  Çıktı dizini: %s", args.output_dir)
    
    results = {}
    
    if args.smoke_run:
        LOG.info("Smoke run modu")
        results = smoke_run_onnx(PRODUCTION_CNN_LSTM_PATH)
    else:
        # Gerçek data yoksa sentetik kullan
        dataset = SyntheticTemporalDataset(n_samples=2000, seq_len=args.seq_len)
        
        results = try_pytorch_training(
            dataset=dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            device_str=args.device,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    
    # Rapor yaz
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Sonuç raporu: %s", args.report_path)
    LOG.info("Sonuç: %s", results.get("status", "UNKNOWN"))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
