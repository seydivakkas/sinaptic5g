"""
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

scripts/validate_results_schema.py
-----------------------------------
FTR results.json dosyasını resmi şemaya ve label kontratına göre doğrular.
SALT OKUMA — mevcut hiçbir dosyayı değiştirmez.

Kullanım:
    python scripts/validate_results_schema.py /path/to/results.json
    python scripts/validate_results_schema.py /path/to/results.json --out evidence/results_schema_validation.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# RESMİ FTR KONTRATI (kaynak: FTR_Aşaması_Teslim_D.pdf)
# ─────────────────────────────────────────────────────────────
VALID_VEHICLE_TYPES = {"sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon"}
VALID_COLORS = {"beyaz", "siyah", "gri", "kirmizi", "mavi", "sari", "yesil", "turuncu", "kahverengi"}

VALID_LABELS_BY_CATEGORY: dict[str, set[str]] = {
    "sofor_eylemi": {
        "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
        "telefonla_konusma", "slalom", "etrafa_bakinma", "emniyet_kemeri_ihlali",
    },
    "nesneler": {"teknocan", "bilgisayar"},
    "yolcular": {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"},
}

ALL_VALID_LABELS: set[str] = set()
for _labels in VALID_LABELS_BY_CATEGORY.values():
    ALL_VALID_LABELS |= _labels

PLATE_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])((\s?[a-zA-Z]\s?)(\d{4,5})|(\s?[a-zA-Z]{2}\s?)(\d{3,4})|(\s?[a-zA-Z]{3}\s?)(\d{2,3}))$"
)

FORBIDDEN_KEYS = {"hiz", "speed", "qod_status", "network_quality", "device_id",
                  "api_status", "frame_id", "bbox", "track_id", "debug", "raw_predictions"}


# ─────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────

def _is_float_in_range(value: object) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def _has_turkish_char(s: str) -> bool:
    return bool(re.search(r"[çğışöüÇĞİŞÖÜ]", s))


# ─────────────────────────────────────────────────────────────
# ANA DOĞRULAMA
# ─────────────────────────────────────────────────────────────

def validate(data: dict) -> list[str]:
    """Hataları string listesi olarak döner. Boş liste = PASS."""
    errors: list[str] = []

    # 1. Zorunlu üst düzey anahtarlar
    for key in ("video_id", "arac_bilgisi", "tespitler"):
        if key not in data:
            errors.append(f"Eksik zorunlu alan: '{key}'")

    # 2. Yasak şema dışı anahtarlar
    for key in data:
        if key in FORBIDDEN_KEYS:
            errors.append(f"Şema dışı yasak alan: '{key}'")

    # 3. arac_bilgisi doğrulaması
    ab = data.get("arac_bilgisi", {})
    if isinstance(ab, dict):
        # tip
        tip = ab.get("tip")
        if tip is None:
            errors.append("arac_bilgisi.tip eksik")
        elif tip not in VALID_VEHICLE_TYPES:
            errors.append(f"arac_bilgisi.tip geçersiz: '{tip}' | Geçerli: {sorted(VALID_VEHICLE_TYPES)}")

        # plaka
        plaka = ab.get("plaka")
        if plaka is None:
            errors.append("arac_bilgisi.plaka eksik")
        elif not isinstance(plaka, str):
            errors.append("arac_bilgisi.plaka string olmalı")
        elif _has_turkish_char(plaka):
            errors.append(f"arac_bilgisi.plaka Türkçe karakter içeriyor: '{plaka}'")
        elif plaka not in ("tespit_edilemedi",) and not PLATE_REGEX.match(plaka.strip()):
            errors.append(f"arac_bilgisi.plaka regex ile eşleşmiyor: '{plaka}'")

        # renk
        renk = ab.get("renk")
        if renk is None:
            errors.append("arac_bilgisi.renk eksik")
        elif renk not in VALID_COLORS:
            errors.append(f"arac_bilgisi.renk geçersiz: '{renk}' | Geçerli: {sorted(VALID_COLORS)}")

        # confidence_score
        cs = ab.get("confidence_score")
        if cs is None:
            errors.append("arac_bilgisi.confidence_score eksik")
        elif not _is_float_in_range(cs):
            errors.append(f"arac_bilgisi.confidence_score geçersiz: {cs}")

        # Şema dışı alt anahtarlar
        for k in ab:
            if k in FORBIDDEN_KEYS:
                errors.append(f"arac_bilgisi altında şema dışı alan: '{k}'")
    elif "arac_bilgisi" in data:
        errors.append("arac_bilgisi dict olmalı")

    # 4. tespitler doğrulaması
    tespitler = data.get("tespitler", [])
    if not isinstance(tespitler, list):
        errors.append("tespitler bir liste (array) olmalı")
    else:
        for idx, t in enumerate(tespitler):
            prefix = f"tespitler[{idx}]"
            if not isinstance(t, dict):
                errors.append(f"{prefix} dict olmalı")
                continue

            # zaman_saniye
            zaman = t.get("zaman_saniye")
            if zaman is None:
                errors.append(f"{prefix}.zaman_saniye eksik")
            elif not isinstance(zaman, (int, float)):
                errors.append(f"{prefix}.zaman_saniye sayı olmalı")

            # kategori
            kat = t.get("kategori")
            if kat is None:
                errors.append(f"{prefix}.kategori eksik")
            elif kat not in VALID_LABELS_BY_CATEGORY:
                errors.append(f"{prefix}.kategori geçersiz: '{kat}' | Geçerli: {list(VALID_LABELS_BY_CATEGORY)}")

            # etiket
            etiket = t.get("etiket")
            if etiket is None:
                errors.append(f"{prefix}.etiket eksik")
            else:
                if _has_turkish_char(etiket):
                    errors.append(f"{prefix}.etiket Türkçe karakter içeriyor: '{etiket}'")
                if etiket not in ALL_VALID_LABELS:
                    errors.append(f"{prefix}.etiket geçersiz: '{etiket}'")
                elif kat in VALID_LABELS_BY_CATEGORY and etiket not in VALID_LABELS_BY_CATEGORY[kat]:
                    errors.append(
                        f"{prefix}.etiket '{etiket}' kategori '{kat}' ile uyuşmuyor"
                    )

            # confidence_score
            cs = t.get("confidence_score")
            if cs is None:
                errors.append(f"{prefix}.confidence_score eksik")
            elif not _is_float_in_range(cs):
                errors.append(f"{prefix}.confidence_score geçersiz: {cs}")

            # Şema dışı alt anahtarlar
            for k in t:
                if k in FORBIDDEN_KEYS:
                    errors.append(f"{prefix} altında şema dışı alan: '{k}'")

    return errors


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FTR results.json schema + label contract validator"
    )
    parser.add_argument("results_json", type=Path, help="Doğrulanacak results.json dosyası")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Doğrulama sonucunun JSON olarak kaydedileceği dosya (opsiyonel)"
    )
    args = parser.parse_args()

    if not args.results_json.is_file():
        print(f"[ERROR] Dosya bulunamadı: {args.results_json}", file=sys.stderr)
        sys.exit(2)

    try:
        with open(args.results_json, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse hatası: {e}", file=sys.stderr)
        sys.exit(2)

    errors = validate(data)

    result_doc: dict = {
        "file": str(args.results_json),
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors,
    }

    print(json.dumps(result_doc, indent=2, ensure_ascii=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result_doc, f, indent=2, ensure_ascii=False)
        print(f"\n[Kaydedildi] {args.out}", file=sys.stderr)

    sys.exit(0 if not errors else 1)


if __name__ == "__main__":
    main()
