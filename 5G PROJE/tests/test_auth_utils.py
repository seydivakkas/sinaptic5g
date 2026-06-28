from __future__ import annotations

import unittest
from unittest.mock import patch

import auth_utils


class AuthUtilsTests(unittest.TestCase):
    def test_missing_secret_fails_closed(self):
        with patch.object(auth_utils, "SECRET_KEY", ""):
            self.assertIsNone(auth_utils.verify_token("anything"))
            with self.assertRaises(RuntimeError):
                auth_utils.create_access_token({"device_id": "device"})

    @unittest.skipUnless(auth_utils.JOSE_AVAILABLE, "python-jose is not installed")
    def test_short_lived_device_session_round_trip(self):
        with patch.object(auth_utils, "SECRET_KEY", "x" * 32):
            token = auth_utils.create_access_token({"device_id": "device"}, expires_in=60)
            payload = auth_utils.verify_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["device_id"], "device")


if __name__ == "__main__":
    unittest.main()
