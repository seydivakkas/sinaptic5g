"""Map official Drive&Act temporal annotations to Sinaptic5G behaviours."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ACTIVITY_MAP = {
    "interacting_with_phone": "telefonla_konusma",
    "talking_on_phone": "telefonla_konusma",
    "drinking": "su_icme",
    "looking_back_left_shoulder": "arkaya_bakma",
    "looking_back_right_shoulder": "arkaya_bakma",
    "working_on_laptop": "bilgisayar",
    "opening_laptop": "bilgisayar",
    "closing_laptop": "bilgisayar",
    "sitting_still": "normal_surus",
}


def build(annotation_dir: Path, output: Path, split_index: int = 0) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    counts: dict[str, Counter] = {name: Counter() for name in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        source = annotation_dir / f"midlevel.chunks_90.split_{split_index}.{split}.csv"
        with source.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                canonical = ACTIVITY_MAP.get(row["activity"])
                if not canonical:
                    continue
                row["canonical_activity"] = canonical
                row["split"] = split
                row["video_relpath"] = f"kinect_color/{row['file_id']}.mp4"
                rows.append(row)
                counts[split][canonical] += 1
    fieldnames = [
        "split", "participant_id", "file_id", "video_relpath", "annotation_id",
        "frame_start", "frame_end", "activity", "canonical_activity", "chunk_id",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    summary = {split: dict(sorted(counter.items())) for split, counter in counts.items()}
    return {"rows": len(rows), "split_index": split_index, "counts": summary}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT / "data/external/drive_and_act/annotations/activities_3s/kinect_color",
    )
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/driveact_manifest.csv")
    parser.add_argument("--split-index", type=int, default=0)
    args = parser.parse_args()
    result = build(args.annotations, args.output, args.split_index)
    (args.output.with_suffix(".summary.json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
