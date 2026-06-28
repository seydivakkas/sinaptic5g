# ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR
# Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

from runtime.competition_adapter import CompetitionAdapterWrapper

def test_competition_adapter_vocab_compliance():
    adapter = CompetitionAdapterWrapper(video_id="demo.mp4")
    
    # Observe vehicle and events
    adapter.observe_vehicle(
        timestamp=0.0,
        track_id=1,
        vehicle_type="sedan",
        type_confidence=0.95,
        color="beyaz",
        color_confidence=0.88,
        plate="34ABC123",
        plate_confidence=0.91
    )
    
    # Sigara içme -> driver action event
    adapter.observe_event(timestamp=0.5, label="sigara_icme", confidence=0.85)
    
    doc = adapter.finalize(validate=False)
    
    assert doc["video_id"] == "demo.mp4"
    assert doc["arac_bilgisi"]["tip"] == "sedan"
    assert doc["arac_bilgisi"]["plaka"] == "34ABC123"
    assert doc["arac_bilgisi"]["renk"] == "beyaz"
    
    assert len(doc["tespitler"]) == 1
    assert doc["tespitler"][0]["etiket"] == "sigara_icme"
    assert doc["tespitler"][0]["kategori"] == "sofor_eylemi"
