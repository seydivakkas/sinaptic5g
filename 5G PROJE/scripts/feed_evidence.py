"""
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

scripts/feed_evidence.py
-------------------------
reports/ ve proje kökündeki mevcut kanıt dosyalarını evidence/ altına kopyalar.
SALT OKUMA + KOPYALAMA — mevcut hiçbir kaynak dosyayı değiştirmez.
evidence/ altındaki dosyalar her çalıştırmada yeniden oluşturulur.

Kullanım:
    python scripts/feed_evidence.py
    python scripts/feed_evidence.py --skip-docker   # Docker adımını atla
    python scripts/feed_evidence.py --dry-run       # Sadece ne yapılacağını göster
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR  = PROJECT_ROOT / "reports"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
MODELS_DIR   = PROJECT_ROOT / "models"


# ─────────────────────────────────────────────────────────────
# YARDIMCILAR
# ─────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def copy_if_exists(src: Path, dst: Path, label: str, dry_run: bool) -> bool:
    if src.is_file():
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        print(f"  [OK]   {src.name}  →  evidence/{dst.name}")
        return True
    else:
        print(f"  [SKIP] {label} bulunamadı: {src}")
        return False


def _docker_running() -> bool:
    try:
        r = subprocess.run(
            ["docker", "info"],
            capture_output=True, timeout=10
        )
        return r.returncode == 0
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# ADIMLAR
# ─────────────────────────────────────────────────────────────

def step_model_lock(dry_run: bool) -> None:
    print("\n[1] model_lock.json")
    copy_if_exists(PROJECT_ROOT / "model_lock.json",
                   EVIDENCE_DIR / "model_lock.json",
                   "model_lock.json", dry_run)


def step_model_hashes(dry_run: bool) -> None:
    print("\n[2] Model Hash Tablosu (model_hashes.txt)")
    onnx_files = sorted(
        list(MODELS_DIR.rglob("*.onnx")) + list(PROJECT_ROOT.glob("*.onnx"))
    )
    if not onnx_files:
        print("  [SKIP] .onnx dosyası bulunamadı")
        return

    lines = []
    for f in onnx_files:
        rel = f.relative_to(PROJECT_ROOT)
        h   = sha256(f)
        lines.append(f"{rel}: {h}")
        print(f"  [hash] {rel}")

    if not dry_run:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        (EVIDENCE_DIR / "model_hashes.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    print(f"  [OK]   {len(lines)} model  →  evidence/model_hashes.txt")


def step_latency(dry_run: bool) -> None:
    print("\n[3] Latency Benchmark")
    # Önce FTR-spesifik profil, yoksa v5 e2e benchmark
    for candidate in (
        "ftr_performance_profile.json",
        "detector_v5_e2e_latency_benchmark.json",
        "e2e_latency_benchmark.json",
    ):
        src = REPORTS_DIR / candidate
        if src.is_file():
            if not dry_run:
                EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, EVIDENCE_DIR / "latency_benchmark.json")
            print(f"  [OK]   {candidate}  →  evidence/latency_benchmark.json")
            return
    print("  [SKIP] Latency benchmark dosyası bulunamadı")


def step_robustness(dry_run: bool) -> None:
    print("\n[4] Robustness Raporu")
    copy_if_exists(REPORTS_DIR / "robustness_stress_test.md",
                   EVIDENCE_DIR / "robustness_report.md",
                   "robustness_stress_test.md", dry_run)


def step_dataset_manifest(dry_run: bool) -> None:
    print("\n[5] Dataset Manifest")
    for candidate in ("dataset_audit.md", "FINAL_EVIDENCE_INDEX.md"):
        src = REPORTS_DIR / candidate
        if src.is_file():
            if not dry_run:
                EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, EVIDENCE_DIR / "dataset_manifest.md")
            print(f"  [OK]   {candidate}  →  evidence/dataset_manifest.md")
            return
    print("  [SKIP] Dataset manifest bulunamadı")


def step_split_summary(dry_run: bool) -> None:
    print("\n[6] Split Summary")
    for candidate in (
        "detector_v5_split_summary_v2.json",
        "detector_v5_split_summary.json",
        "detector_v4_split_summary.json",
    ):
        src = REPORTS_DIR / candidate
        if src.is_file():
            if not dry_run:
                EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, EVIDENCE_DIR / "split_summary.json")
            print(f"  [OK]   {candidate}  →  evidence/split_summary.json")
            return
    print("  [SKIP] Split summary bulunamadı")


def step_ftr_acceptance_report(dry_run: bool) -> None:
    print("\n[7] FTR Acceptance Report")
    for candidate in (
        "final_ftr_acceptance_report.md",
        "final_quality_gate.md",
        "final_validation_checklist.md",
    ):
        src = REPORTS_DIR / candidate
        if src.is_file():
            if not dry_run:
                EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, EVIDENCE_DIR / "ftr_acceptance_report.md")
            print(f"  [OK]   {candidate}  →  evidence/ftr_acceptance_report.md")
            return
    print("  [SKIP] FTR acceptance report bulunamadı")


def step_error_analysis(dry_run: bool) -> None:
    print("\n[8] Error Analysis")
    copy_if_exists(REPORTS_DIR / "error_analysis_report.md",
                   EVIDENCE_DIR / "error_analysis_report.md",
                   "error_analysis_report.md", dry_run)


def step_schema_validation(dry_run: bool) -> None:
    """validate_results_schema.py'yi çalıştırıp sonucu kaydeder."""
    print("\n[9] JSON Schema + Label Kontrat Doğrulaması")

    # Test çıktısı aranacak yerlerin öncelik sırası
    candidates = [
        PROJECT_ROOT / "data" / "output" / "results.json",
        PROJECT_ROOT / "tests" / "final_smoke_output" / "results.json",
        PROJECT_ROOT / "tests" / "smoke_output" / "results.json",
    ]
    results_json: Path | None = None
    for c in candidates:
        if c.is_file():
            results_json = c
            break

    if results_json is None:
        print("  [SKIP] Test results.json bulunamadı (Docker run sonrası tekrar deneyin)")
        return

    validator = PROJECT_ROOT / "scripts" / "validate_results_schema.py"
    if not validator.is_file():
        print(f"  [SKIP] validate_results_schema.py bulunamadı: {validator}")
        return

    schema_out  = EVIDENCE_DIR / "results_schema_validation.json"
    label_out   = EVIDENCE_DIR / "label_contract_validation.json"

    if not dry_run:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [sys.executable, str(validator), str(results_json), "--out", str(schema_out)],
            capture_output=True, text=True
        )
        if schema_out.is_file():
            print(f"  [OK]   results_schema_validation.json  (exit={r.returncode})")
        else:
            print(f"  [FAIL] Schema doğrulama çıktısı oluşturulamadı")

        # label_contract_validation.json = schema doğrulaması ile aynı rapor (label da dahil)
        if schema_out.is_file():
            shutil.copy2(schema_out, label_out)
            # status'u override et
            with open(schema_out, encoding="utf-8") as f:
                doc = json.load(f)
            label_errors = [e for e in doc.get("errors", []) if "etiket" in e or "kategori" in e]
            label_doc = {
                "status": "PASS" if not label_errors else "FAIL",
                "error_count": len(label_errors),
                "errors": label_errors,
            }
            with open(label_out, "w", encoding="utf-8") as f:
                json.dump(label_doc, f, indent=2, ensure_ascii=False)
            print(f"  [OK]   label_contract_validation.json")
    else:
        print(f"  [DRY]  validate_results_schema.py {results_json}")


def step_docker_logs(dry_run: bool, skip_docker: bool) -> None:
    print("\n[10] Docker Build & Run Logları")

    if skip_docker:
        print("  [SKIP] --skip-docker bayrağı aktif")
        return

    if not _docker_running():
        print("  [SKIP] Docker Daemon çalışmıyor")
        return

    if dry_run:
        print("  [DRY]  docker build + docker run")
        return

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # Docker Build
    build_log = EVIDENCE_DIR / "docker_build_log.txt"
    print("  Çalışıyor: docker build ...")
    with open(build_log, "w", encoding="utf-8") as f:
        r = subprocess.run(
            ["docker", "build", "-t", "teknofest/sinaptic5g:ftr", "."],
            cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT
        )
    if r.returncode == 0:
        print("  [OK]   docker_build_log.txt")
    else:
        print("  [FAIL] Docker build başarısız — log kaydedildi")
        return

    # Docker Run — smoke test videosu varsa
    smoke_video = PROJECT_ROOT / "tests" / "smoke_input" / "video.mp4"
    if not smoke_video.is_file():
        print(f"  [SKIP] Smoke video bulunamadı: {smoke_video}")
        print("         docker_run_log.txt atlandı.")
        return

    run_log = EVIDENCE_DIR / "docker_run_log.txt"
    print("  Çalışıyor: docker run ...")
    with open(run_log, "w", encoding="utf-8") as f:
        r = subprocess.run([
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{smoke_video}:/app/data/input/video.mp4:ro",
            "-v", f"{EVIDENCE_DIR}:/app/data/output:rw",
            "teknofest/sinaptic5g:ftr",
        ], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode == 0:
        print("  [OK]   docker_run_log.txt")
    else:
        print("  [FAIL] Docker run başarısız — log kaydedildi")


# ─────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────

def print_summary(dry_run: bool) -> None:
    print("\n" + "─" * 60)
    if dry_run:
        print("DRY-RUN tamamlandı. Yukarıdaki [OK] satırları gerçek çalıştırmada kopyalanacak.")
    else:
        files = sorted(EVIDENCE_DIR.iterdir()) if EVIDENCE_DIR.is_dir() else []
        print(f"evidence/ klasörü: {len(files)} dosya")
        for f in files:
            size = f.stat().st_size
            print(f"  {f.name:<45} {size:>10} byte")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="evidence/ klasörünü reports/'tan besler")
    parser.add_argument("--dry-run",      action="store_true", help="Sadece ne yapılacağını göster, kopyalama yapma")
    parser.add_argument("--skip-docker",  action="store_true", help="Docker build/run adımını atla")
    args = parser.parse_args()

    dry = args.dry_run
    print(f"{'[DRY-RUN] ' if dry else ''}evidence/ besleme başlatıldı — {PROJECT_ROOT}")

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    step_model_lock(dry)
    step_model_hashes(dry)
    step_latency(dry)
    step_robustness(dry)
    step_dataset_manifest(dry)
    step_split_summary(dry)
    step_ftr_acceptance_report(dry)
    step_error_analysis(dry)
    step_schema_validation(dry)
    step_docker_logs(dry, args.skip_docker)

    print_summary(dry)


if __name__ == "__main__":
    main()
