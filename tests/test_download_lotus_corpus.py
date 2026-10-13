import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.download_lotus_corpus import download_verified, expected_payload


class Response:
    status = 200
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self, _):
        body, self.body = self.body, b""
        return body


class DownloadTests(unittest.TestCase):
    def test_verified_source_download_and_reuse(self):
        body = b"question||steps##12\n"
        expected = {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.txt"
            with patch("urllib.request.urlopen", return_value=Response(body)) as request:
                result = download_verified("https://example.invalid/pinned", path, expected)
                self.assertTrue(result["downloaded"])
                self.assertEqual(path.read_bytes(), body)
                result = download_verified("https://example.invalid/pinned", path, expected)
                self.assertTrue(result["reused_verified_file"])
                self.assertEqual(request.call_count, 1)

    def test_mismatched_payload_never_installed(self):
        expected = {"sha256": "0" * 64, "bytes": 4}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.txt"
            with patch("urllib.request.urlopen", return_value=Response(b"bad!")):
                with self.assertRaisesRegex(ValueError, "checksum/size"):
                    download_verified("https://example.invalid/pinned", path, expected)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_existing_mismatch_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.txt"
            path.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "preserved"):
                download_verified("https://example.invalid/pinned", path, {"sha256": "0" * 64, "bytes": 4})
            self.assertEqual(path.read_bytes(), b"original")

    def test_expected_payload_from_pinned_lfs_pointer(self):
        blob = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 123\n"
        with patch("subprocess.check_output", return_value=blob):
            expected = expected_payload(Path("source"), "train")
        self.assertEqual(expected["sha256"], "a" * 64)
        self.assertEqual(expected["bytes"], 123)
        self.assertEqual(expected["verification"], "pinned_git_lfs_oid")

    def test_invalid_pointer_rejected_and_raw_blob_hashed(self):
        with patch("subprocess.check_output", return_value=b"version https://git-lfs.github.com/spec/v1\n"):
            with self.assertRaises(ValueError):
                expected_payload(Path("source"), "train")
        with patch("subprocess.check_output", return_value=b"abc"):
            expected = expected_payload(Path("source"), "valid")
        self.assertEqual(expected["sha256"], hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(expected["bytes"], 3)


if __name__ == "__main__":
    unittest.main()
