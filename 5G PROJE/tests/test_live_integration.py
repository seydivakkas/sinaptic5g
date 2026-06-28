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
tests/test_live_integration.py
==============================
Unit and integration tests for the live 5G/MEC expansion modules:
- QoDClient (src/live/qod_client.py)
- MobileSync (src/live/mobile_sync.py)
- MECOffloadManager (src/live/mec_offload.py)
"""

import asyncio
import unittest
import time
from src.live.qod_client import QoDClient, QoDProfile, QoDSession
from src.live.mobile_sync import MobileSync, SyncPayload
from src.live.mec_offload import MECOffloadManager, OffloadStrategy, OffloadDecision

class TestLiveIntegration(unittest.TestCase):

    def test_qod_client_lifecycle(self):
        """Tests that the QoDClient can create, query, and delete sessions correctly."""
        client = QoDClient(base_url="https://api.operator.com/qod/v0", token="test_token_123")
        
        # Create session
        session = asyncio.run(client.create_session(ue_id="ue-phone-99", profile=QoDProfile.LOW_LATENCY, duration=10))
        self.assertIsNotNone(session)
        self.assertEqual(session.ue_id, "ue-phone-99")
        self.assertEqual(session.profile, QoDProfile.LOW_LATENCY)
        self.assertFalse(session.is_expired)
        
        # Get status
        status = asyncio.run(client.get_session_status(session.session_id))
        self.assertIsNotNone(status)
        self.assertEqual(status["sessionId"], session.session_id)
        self.assertEqual(status["status"], "AVAILABLE")
        self.assertTrue(status["remainingSeconds"] > 0)
        
        # Delete session
        deleted = asyncio.run(client.delete_session(session.session_id))
        self.assertTrue(deleted)
        
        # Query deleted session should return None
        status_deleted = asyncio.run(client.get_session_status(session.session_id))
        self.assertIsNone(status_deleted)

    def test_mobile_sync_offline_buffering(self):
        """Tests that MobileSync correctly buffers payloads in offline mode without crashing."""
        sync = MobileSync(host="127.0.0.1", port=9999)
        asyncio.run(sync.start())
        
        payload = SyncPayload(
            timestamp=time.time(),
            video_id="test_video.mp4",
            detection_count=2,
            risk_score=0.45,
            alerts=["telefonla_konusma", "esneme"],
            plate_number="34ABC123"
        )
        
        # Broadcast in offline mode should buffer
        clients_notified = asyncio.run(sync.broadcast(payload))
        self.assertEqual(clients_notified, 0)
        
        # Check pending buffer
        buf = sync.get_pending_buffer()
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0]["type"], "detection")
        self.assertEqual(buf[0]["video_id"], "test_video.mp4")
        self.assertEqual(buf[0]["count"], 2)
        self.assertEqual(buf[0]["risk"], 0.45)
        self.assertEqual(buf[0]["plate"], "34ABC123")
        self.assertIn("telefonla_konusma", buf[0]["alerts"])
        
        asyncio.run(sync.stop())

    def test_mec_offload_decisions(self):
        """Tests MECOffloadManager local/remote offload strategies and fallback."""
        # Setup manager with local strategy
        manager_local = MECOffloadManager(strategy=OffloadStrategy.ALWAYS_LOCAL)
        decision_local = manager_local.decide("coco_detector")
        self.assertFalse(decision_local.use_mec)
        self.assertEqual(decision_local.reason, "ALWAYS_LOCAL")
        
        # Setup manager with remote strategy but MEC unavailable
        manager_remote_offline = MECOffloadManager(strategy=OffloadStrategy.ALWAYS_REMOTE)
        decision_remote_offline = manager_remote_offline.decide("coco_detector")
        self.assertFalse(decision_remote_offline.use_mec)
        self.assertEqual(decision_remote_offline.reason, "MEC erisilemiyor")
        
        # Force MEC online for adaptive tests
        manager_adaptive = MECOffloadManager(strategy=OffloadStrategy.ADAPTIVE)
        manager_adaptive._mec_available = True
        manager_adaptive._mec_rtt_ms = 40.0 # Fast RTT
        
        # Task local cost for coco_detector is 120ms > 40ms RTT, so should use MEC
        decision_coco = manager_adaptive.decide("coco_detector")
        self.assertTrue(decision_coco.use_mec)
        
        # Task local cost for temporal_lstm is 30ms < 40ms RTT, so should keep local
        decision_lstm = manager_adaptive.decide("temporal_lstm")
        self.assertFalse(decision_lstm.use_mec)
        
        # Run inference test
        dummy_input = "test_frame"
        def local_inference_stub(inp):
            return {"detected": True, "input": inp}
            
        res = asyncio.run(manager_adaptive.run_inference("coco_detector", dummy_input, local_inference_stub))
        self.assertEqual(res["detected"], True)
        self.assertEqual(res["input"], "test_frame")
        
        # Check stats (it fell back to local execution since MEC HTTP request is a stub)
        stats = manager_adaptive.stats()
        self.assertEqual(stats["raw"]["fallback"], 1)

if __name__ == "__main__":
    unittest.main()
