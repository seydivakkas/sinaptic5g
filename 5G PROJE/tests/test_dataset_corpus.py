from pathlib import Path

from scripts.build_detector_corpus import allocate_groups, assert_no_group_leakage, remap_label, source_group_id, Sample


def test_roboflow_variants_share_group():
    a = source_group_id("driver", "phone_jpg.rf.abc123")
    b = source_group_id("driver", "phone_jpg.rf.def456")
    assert a == b


def test_phone_and_texting_map_to_same_canonical_class(tmp_path: Path):
    label = tmp_path / "sample.txt"
    label.write_text("1 0.5 0.5 0.2 0.2\n2 0.4 0.4 0.1 0.1\n", encoding="utf-8")
    lines, classes = remap_label(label, {1: 0, 2: 0})
    assert classes == (0,)
    assert len(lines) == 2


def test_group_split_has_no_leakage(tmp_path: Path):
    samples = []
    for index in range(30):
        group = f"source:g{index}"
        for variant in range(2):
            samples.append(
                Sample("source", tmp_path / f"{index}_{variant}.jpg", tmp_path / "x.txt", group, ("0 0.5 0.5 0.2 0.2",), (0,), f"{index}-{variant}")
            )
    assignment = allocate_groups(samples, seed=2026)
    records = [{"group_id": sample.group_id, "split": assignment[sample.group_id]} for sample in samples]
    assert_no_group_leakage(records)
    assert set(assignment.values()) == {"train", "val", "test"}

