"""Canonical TEKNOFEST FTR output contract.

This module is the only source of truth for public labels, plate syntax and
``results.json`` validation. Internal models may use richer labels; only this
boundary is allowed to emit the competition document.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


VEHICLE_TYPES = (
    "sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon",
)
VEHICLE_COLORS = (
    "beyaz", "siyah", "gri", "kirmizi", "mavi", "sari", "yesil",
    "turuncu", "kahverengi",
)
DRIVER_ACTIONS = (
    "arkaya_bakma", "esneme", "sigara_icme", "su_icme",
    "telefonla_konusma", "slalom", "etrafa_bakinma",
    "emniyet_kemeri_ihlali",
)
OBJECT_LABELS = ("teknocan", "bilgisayar")
PASSENGER_LABELS = ("arka_koltuk_1", "arka_koltuk_2", "on_koltuk")

CATEGORY_BY_LABEL = {
    **{label: "sofor_eylemi" for label in DRIVER_ACTIONS},
    **{label: "nesneler" for label in OBJECT_LABELS},
    **{label: "yolcular" for label in PASSENGER_LABELS},
}

LABEL_ALIASES = {
    "telefon_konusma": "telefonla_konusma",
    "sag_sol_bakis": "etrafa_bakinma",
    "dikkatsizlik": "etrafa_bakinma",
    "toy": "teknocan",
    "laptop": "bilgisayar",
    "computer": "bilgisayar",
    "front_passenger": "on_koltuk",
    "rear_left_passenger": "arka_koltuk_1",
    "rear_right_passenger": "arka_koltuk_2",
}

PLATE_PATTERN = (
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"([A-Z][0-9]{4,5}|[A-Z]{2}[0-9]{3,4}|[A-Z]{3}[0-9]{2,3})$"
)
PLATE_RE = re.compile(PLATE_PATTERN)


def clamp_confidence(value: Any) -> float:
    """Return a finite confidence in the public ``[0, 1]`` interval."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return round(max(0.0, min(1.0, number)), 6)


def normalize_plate(value: Any) -> str | None:
    """Normalize a Turkish plate; return ``None`` when it is not contract-valid."""
    if value is None:
        return None
    plate = re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    return plate if PLATE_RE.fullmatch(plate) else None


def normalize_label(value: Any) -> tuple[str, str] | None:
    """Map an internal label to the single public category/label vocabulary."""
    label = str(value or "").strip().lower()
    label = LABEL_ALIASES.get(label, label)
    category = CATEGORY_BY_LABEL.get(label)
    return (category, label) if category else None


def default_schema_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / "results.schema.json"


def load_schema(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    schema_path = Path(path) if path else default_schema_path()
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    Draft202012Validator.check_schema(schema)
    return schema


def validate_results(
    document: Mapping[str, Any],
    schema_path: str | os.PathLike[str] | None = None,
) -> None:
    """Raise ``jsonschema.ValidationError`` for any contract violation."""
    Draft202012Validator(load_schema(schema_path)).validate(dict(document))


def atomic_write_results(
    document: Mapping[str, Any],
    output_path: str | os.PathLike[str],
    schema_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Validate, fsync and atomically replace the public results file."""
    validate_results(document, schema_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(document, handle, ensure_ascii=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination

