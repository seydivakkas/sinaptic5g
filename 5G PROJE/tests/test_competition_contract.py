from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import ValidationError

from src.competition_adapter import CompetitionAdapter
from src.competition_contract import atomic_write_results, normalize_plate, validate_results


class CompetitionContractTests(unittest.TestCase):
    def test_empty_pipeline_still_emits_only_contract_values(self):
        result = CompetitionAdapter("video.mp4").finalize()
        self.assertEqual(result["tespitler"], [])
        self.assertEqual(result["arac_bilgisi"]["confidence_score"], 0.0)
        self.assertNotIn("tespit_edilemedi", json.dumps(result, ensure_ascii=False))

    def test_internal_alias_is_normalized_and_events_are_segmented(self):
        adapter = CompetitionAdapter("video.mp4", event_gap_seconds=1.0)
        adapter.observe_event(1.0, "telefon_konusma", 0.7)
        adapter.observe_event(1.4, "telefonla_konusma", 0.9)
        adapter.observe_event(4.0, "telefon_konusma", 0.8)
        result = adapter.finalize()
        self.assertEqual(len(result["tespitler"]), 2)
        self.assertTrue(all(item["etiket"] == "telefonla_konusma" for item in result["tespitler"]))
        self.assertEqual(result["tespitler"][0]["zaman_saniye"], 1.4)

    def test_primary_track_selection_prefers_valid_plate(self):
        adapter = CompetitionAdapter("video.mp4")
        adapter.observe_vehicle(
            timestamp=0.0, track_id="long", vehicle_type="suv", type_confidence=0.9,
            color="mavi", color_confidence=0.9,
        )
        adapter.observe_vehicle(
            timestamp=8.0, track_id="long", vehicle_type="suv", type_confidence=0.9,
            color="mavi", color_confidence=0.9,
        )
        adapter.observe_vehicle(
            timestamp=2.0, track_id="plate", vehicle_type="pickup", type_confidence=0.8,
            color="beyaz", color_confidence=0.7, plate="34 ABC 123", plate_confidence=0.6,
        )
        result = adapter.finalize()
        self.assertEqual(result["arac_bilgisi"]["tip"], "pickup")
        self.assertEqual(result["arac_bilgisi"]["plaka"], "34ABC123")
        self.assertEqual(result["arac_bilgisi"]["confidence_score"], 0.6)

    def test_category_and_label_must_match(self):
        invalid = CompetitionAdapter("video.mp4").finalize()
        invalid["tespitler"] = [{
            "zaman_saniye": 1.0,
            "kategori": "nesneler",
            "etiket": "esneme",
            "confidence_score": 0.5,
        }]
        with self.assertRaises(ValidationError):
            validate_results(invalid)

    def test_atomic_write_leaves_valid_utf8_json(self):
        result = CompetitionAdapter("video.mp4").finalize()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "results.json"
            atomic_write_results(result, destination)
            loaded = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(loaded, result)
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_plate_normalization(self):
        self.assertEqual(normalize_plate("34 abc 123"), "34ABC123")
        self.assertIsNone(normalize_plate("not-a-plate"))


if __name__ == "__main__":
    unittest.main()
