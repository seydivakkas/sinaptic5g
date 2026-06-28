"""Build a reproducible, leakage-audited dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml


SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_id(path: Path) -> str:
    stem = path.stem
    prefix, marker, suffix = stem.rpartition("_f")
    return prefix if marker and suffix.isdigit() else stem


def build_manifest(dataset_root: Path, license_inventory: Path) -> dict:
    records: list[dict] = []
    split_counts = Counter()
    class_histogram = Counter()
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    hashes_by_split: dict[str, dict[str, str]] = defaultdict(dict)

    for split in SPLITS:
        # Hem Ultralytics'in ``split/images`` hem de klasik
        # ``images/split`` dizilimini kanonik olarak denetle.
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.exists():
            image_dir = dataset_root / split / "images"
            label_dir = dataset_root / split / "labels"
        images = sorted(path for path in image_dir.glob("**/*") if path.is_file()) if image_dir.exists() else []
        for image in images:
            relative = image.relative_to(dataset_root).as_posix()
            digest = sha256(image)
            group = group_id(image)
            records.append({"path": relative, "sha256": digest, "split": split, "group_id": group})
            split_counts[split] += 1
            groups_by_split[split].add(group)
            hashes_by_split[split][relative] = digest

            label = label_dir / f"{image.stem}.txt"
            if label.is_file():
                for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.split()
                    if parts:
                        class_histogram[parts[0]] += 1

    group_overlap = {
        f"{left}_{right}": sorted(groups_by_split[left] & groups_by_split[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1:]
    }
    if any(group_overlap.values()):
        raise ValueError(f"group leakage detected: {group_overlap}")

    content_hash = hashlib.sha256(
        "\n".join(f"{item['path']}:{item['sha256']}:{item['split']}" for item in records).encode("utf-8")
    ).hexdigest()
    split_compliance = all(split_counts[name] > 0 for name in SPLITS)
    total = sum(split_counts.values())
    return {
        "status": "OLCULDU",
        "dataset_version": date.today().isoformat(),
        "manifest_sha256": content_hash,
        "split_seed": 42,
        "split_policy": "source splits preserved; physical directories and inferred filename groups audited",
        "split_ratio": {
            name: round(split_counts[name] / total, 4) if total else 0.0
            for name in SPLITS
        },
        "desired_group_keys": ["video_id", "capture_session", "vehicle_id", "person_id"],
        "available_group_key": "filename-derived source key",
        "group_audit_scope": "Subject/person IDs are not available for every external source; overlap result is limited to inferred filename groups.",
        "train_count": int(split_counts["train"]),
        "val_count": int(split_counts["val"]),
        "test_count": int(split_counts["test"]),
        "split_compliance": split_compliance,
        "split_compliance_note": (
            "physical train/val/test sets are present"
            if split_compliance
            else "BLOCKED: val/test are empty; KPI values cannot be promoted to OLCULDU"
        ),
        "class_histogram": {str(key): value for key, value in sorted(class_histogram.items())},
        "license_inventory": license_inventory.as_posix(),
        "group_overlap": group_overlap,
        "file_count": len(records),
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset/manifest.yaml"))
    parser.add_argument("--licenses", type=Path, default=Path("dataset/LICENSE_INVENTORY.md"))
    args = parser.parse_args()
    manifest = build_manifest(args.dataset, args.licenses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"manifest={args.output} sha256={manifest['manifest_sha256']} files={manifest['file_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
