"""Build a reproducible, leakage-resistant 70/20/10 YOLO corpus.

The builder consumes only enabled, licence-audited detection sources from
``configs/datasets.yaml``. Roboflow augmentation variants are grouped by their
pre-``.rf.`` stem, exact duplicate image bytes are removed, and no synthetic or
dummy image is ever admitted to validation or test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


PROJECT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}


@dataclass(frozen=True)
class Sample:
    source: str
    image: Path
    label: Path
    group_id: str
    mapped_lines: tuple[str, ...]
    classes: tuple[int, ...]
    image_sha256: str

    @property
    def primary_class(self) -> int:
        return min(self.classes)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_group_id(source: str, stem: str) -> str:
    """Collapse Roboflow variants and local augmentation suffixes into one group."""
    base = re.split(r"\.rf\.[0-9a-fA-F]+", stem, maxsplit=1)[0]
    base = re.sub(r"_(?:jpg|jpeg|png)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_(?:aug|copy|flip|noise|bright|dark)\d*$", "", base, flags=re.IGNORECASE)
    return f"{source}:{base.lower()}"


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
        candidate = images_dir / f"{stem}{suffix.upper()}"
        if candidate.is_file():
            return candidate
    return None


def remap_label(label: Path, label_map: dict[int, int]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    mapped: list[str] = []
    classes: list[int] = []
    for raw in label.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            source_class = int(parts[0])
            coords = [float(value) for value in parts[1:5]]
        except ValueError:
            continue
        if source_class not in label_map or not all(0.0 <= value <= 1.0 for value in coords):
            continue
        target = label_map[source_class]
        mapped.append(f"{target} " + " ".join(f"{value:.6f}" for value in coords))
        classes.append(target)
    return tuple(mapped), tuple(sorted(set(classes)))


def iter_source_pairs(name: str, cfg: dict) -> Iterable[tuple[Path, Path]]:
    root = PROJECT / cfg["root"]
    if cfg["format"] == "yolo-flat":
        labels = root / cfg["labels_dir"]
        images = root / cfg["images_dir"]
        for label in sorted(labels.glob("*.txt")):
            image = find_image(images, label.stem)
            if image:
                yield image, label
        return

    for split in cfg.get("splits", []):
        split_dir = root / split
        images = split_dir / "images"
        labels = split_dir / "labels"
        for label in sorted(labels.glob("*.txt")):
            image = find_image(images, label.stem)
            if image:
                yield image, label


def collect_samples(config: dict) -> tuple[list[Sample], dict]:
    samples: list[Sample] = []
    seen_hashes: set[str] = set()
    audit = {"skipped_unmapped": 0, "skipped_duplicates": 0, "missing_sources": []}
    for name, cfg in config["sources"].items():
        if not cfg.get("enabled") or cfg.get("task") != "detection":
            continue
        root = PROJECT / cfg["root"]
        if not root.exists():
            audit["missing_sources"].append(name)
            continue
        print(f"Scanning source: {name}...")
        label_map = {int(k): int(v) for k, v in cfg["label_map"].items()}
        count = 0
        for image, label in iter_source_pairs(name, cfg):
            mapped_lines, classes = remap_label(label, label_map)
            if not mapped_lines:
                audit["skipped_unmapped"] += 1
                continue
            image_hash = sha256_file(image)
            if image_hash in seen_hashes:
                audit["skipped_duplicates"] += 1
                continue
            seen_hashes.add(image_hash)
            samples.append(
                Sample(
                    source=name,
                    image=image,
                    label=label,
                    group_id=source_group_id(name, image.stem),
                    mapped_lines=mapped_lines,
                    classes=classes,
                    image_sha256=image_hash,
                )
            )
            count += 1
            if count % 1000 == 0:
                print(f"  Processed {count} images in {name}...")
        print(f"Source {name} scan complete. Added {count} valid samples.")
    return samples, audit


def cap_groups(samples: list[Sample], max_groups_per_class: int, seed: int) -> list[Sample]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)
    by_class: dict[int, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        by_class[min(member.primary_class for member in members)].append(group_id)
    rng = random.Random(seed)
    selected: set[str] = set()
    for class_id, group_ids in sorted(by_class.items()):
        rng.shuffle(group_ids)
        selected.update(group_ids[:max_groups_per_class])
    return [sample for sample in samples if sample.group_id in selected]


def extract_session_id(filename: str) -> str:
    stem = Path(filename).stem
    stem_no_rf = re.split(r"\.rf\.[0-9a-fA-F]+", stem, maxsplit=1)[0]
    frame_match = re.search(r"(_frame_?\d+)", stem_no_rf, flags=re.IGNORECASE)
    if frame_match:
        return stem_no_rf[:frame_match.start()]
    numeric_match = re.search(r"_(\d+)$", stem_no_rf)
    if numeric_match:
        return stem_no_rf[:numeric_match.start()]
    return stem_no_rf


def allocate_groups(samples: list[Sample], seed: int) -> dict[str, str]:
    # Group samples by session_id
    sessions: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        session_id = f"{sample.source}:{extract_session_id(sample.image.name).lower()}"
        sessions[session_id].append(sample)

    # Stratification by source and primary class
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for session_id, members in sessions.items():
        source = members[0].source
        primary_class = min(member.primary_class for member in members)
        key = (source, primary_class)
        strata[key].append(session_id)

    rng = random.Random(seed)
    session_assignment: dict[str, str] = {}
    for key, session_ids in sorted(strata.items()):
        rng.shuffle(session_ids)
        total = len(session_ids)
        n_test = round(total * SPLIT_RATIOS["test"])
        n_val = round(total * SPLIT_RATIOS["val"])
        if total >= 10:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
        if n_test + n_val >= total:
            n_test = max(0, min(n_test, total - 1))
            n_val = max(0, min(n_val, total - n_test - 1))
        for session_id in session_ids[:n_test]:
            session_assignment[session_id] = "test"
        for session_id in session_ids[n_test : n_test + n_val]:
            session_assignment[session_id] = "val"
        for session_id in session_ids[n_test + n_val :]:
            session_assignment[session_id] = "train"

    # Map group_id to session's split
    assignment: dict[str, str] = {}
    for sample in samples:
        session_id = f"{sample.source}:{extract_session_id(sample.image.name).lower()}"
        assignment[sample.group_id] = session_assignment[session_id]

    # Write session split log to reports/split_session_log.csv
    log_rows = ["session_id,split"]
    for session_id, split in sorted(session_assignment.items()):
        log_rows.append(f"{session_id},{split}")
    reports_dir = PROJECT / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "split_session_log.csv").write_text("\n".join(log_rows) + "\n", encoding="utf-8")

    return assignment



def hardlink_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def materialize(samples: list[Sample], assignment: dict[str, str], output: Path, class_names: dict) -> dict:
    if output.exists():
        shutil.rmtree(output)
    for split in SPLITS:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    counts = {split: {"images": 0, "instances": Counter(), "sources": Counter()} for split in SPLITS}
    records: list[dict] = []
    mode_counts: Counter = Counter()
    for sample in sorted(samples, key=lambda item: (assignment[item.group_id], item.source, item.image.name)):
        split = assignment[sample.group_id]
        short = hashlib.sha256(f"{sample.source}:{sample.image}".encode()).hexdigest()[:12]
        image_name = f"{sample.source}__{short}{sample.image.suffix.lower()}"
        label_name = f"{sample.source}__{short}.txt"
        image_out = output / split / "images" / image_name
        label_out = output / split / "labels" / label_name
        mode_counts[hardlink_or_copy(sample.image, image_out)] += 1
        label_out.write_text("\n".join(sample.mapped_lines) + "\n", encoding="utf-8")
        counts[split]["images"] += 1
        counts[split]["sources"][sample.source] += 1
        counts[split]["instances"].update(int(line.split()[0]) for line in sample.mapped_lines)
        records.append(
            {
                "split": split,
                "group_id": sample.group_id,
                "source": sample.source,
                "source_image": str(sample.image.relative_to(PROJECT)).replace("\\", "/"),
                "source_label": str(sample.label.relative_to(PROJECT)).replace("\\", "/"),
                "image_sha256": sample.image_sha256,
                "output_image": str(image_out.relative_to(PROJECT)).replace("\\", "/"),
                "classes": list(sample.classes),
            }
        )

    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    manifest_sha = sha256_file(manifest)
    (output / "manifest.sha256").write_text(f"{manifest_sha}  manifest.jsonl\n", encoding="ascii")
    yaml.safe_dump(
        {
            "path": str(output.resolve()).replace("\\", "/"),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "names": {int(k): v for k, v in class_names.items()},
            "nc": len(class_names),
        },
        (output / "data.yaml").open("w", encoding="utf-8"),
        sort_keys=False,
        allow_unicode=True,
    )
    serializable_counts = {
        split: {
            "images": value["images"],
            "instances": dict(sorted(value["instances"].items())),
            "sources": dict(sorted(value["sources"].items())),
        }
        for split, value in counts.items()
    }
    return {"counts": serializable_counts, "manifest_sha256": manifest_sha, "materialization": dict(mode_counts)}


def assert_no_group_leakage(records: Iterable[dict]) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        group_splits[record["group_id"]].add(record["split"])
    leaked = {group: splits for group, splits in group_splits.items() if len(splits) > 1}
    if leaked:
        raise AssertionError(f"group leakage detected: {list(leaked.items())[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/datasets.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT / "data/curated/detector_v2")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-groups-per-class", type=int, default=2500)
    args = parser.parse_args()
    args.output = args.output.resolve()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    samples, audit = collect_samples(config)
    samples = cap_groups(samples, args.max_groups_per_class, args.seed)
    assignment = allocate_groups(samples, args.seed)
    result = materialize(samples, assignment, args.output, config["canonical_detection_classes"])
    records = [json.loads(line) for line in (args.output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert_no_group_leakage(records)
    result.update({"audit": audit, "seed": args.seed, "split_ratios_target": SPLIT_RATIOS})
    dataset_name = args.output.name
    report = PROJECT / f"reports/{dataset_name}_dataset_report.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

