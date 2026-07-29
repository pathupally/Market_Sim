import hashlib
from pathlib import Path
import tempfile
import unittest

from tools.model import fetch


class ModelFetchTests(unittest.TestCase):
    def test_moving_revision_is_rejected(self) -> None:
        entry = {
            "id": "test-model",
            "repository": "owner/model",
            "revision": "main",
            "max_download_bytes": 1,
            "files": [
                {
                    "path": "config.json",
                    "size": 1,
                    "sha256": "0" * 64,
                }
            ],
        }
        with self.assertRaises(fetch.ModelToolError):
            fetch.validate_model_entry(entry)

    def test_corrupted_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(b"corrupt")
            expected = {
                "path": "config.json",
                "size": 7,
                "sha256": hashlib.sha256(b"correct").hexdigest(),
            }
            with self.assertRaises(fetch.ModelToolError):
                fetch.verify_file(path, expected)

    def test_manifest_is_locked_and_small_by_default(self) -> None:
        manifest = fetch.load_manifest(fetch.DEFAULT_MANIFEST)
        smol = fetch.select_model(manifest, "smollm2-135m")
        qwen = fetch.select_model(manifest, "qwen2.5-0.5b-instruct")
        self.assertEqual(len(smol["revision"]), 40)
        self.assertLess(sum(item["size"] for item in smol["files"]), 300_000_000)
        tokenizer = next(
            item for item in smol["files"] if item["path"] == "tokenizer.json"
        )
        self.assertEqual(tokenizer["size"], 2_104_556)
        self.assertEqual(
            tokenizer["sha256"],
            "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        )
        self.assertEqual([item["path"] for item in qwen["files"]], ["config.json"])
        self.assertFalse(qwen["deferred_weight"]["fetch_allowed"])

    def test_duplicate_manifest_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(
                '{"schema_version": 0, "schema_version": 1}',
                encoding="utf-8",
            )
            with self.assertRaises(fetch.ModelToolError):
                fetch.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
