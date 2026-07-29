from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.model import conformance
from tools.model import fetch


class ModelConformanceTests(unittest.TestCase):
    def make_inputs(
        self, directory: str
    ) -> tuple[Path, Path, Path, dict[str, object]]:
        root = Path(directory)
        checkpoint = root / "model.safetensors"
        checkpoint.write_bytes(b"locked checkpoint")
        entry = {
            "path": "model.safetensors",
            "size": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
        fixture = root / "fixture.safetensors"
        fixture.write_bytes(b"fixture")
        fixture.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "fixture_size": fixture.stat().st_size,
                    "fixture_sha256": hashlib.sha256(
                        fixture.read_bytes()
                    ).hexdigest(),
                    "checkpoint_size": entry["size"],
                    "checkpoint_sha256": entry["sha256"],
                }
            ),
            encoding="utf-8",
        )
        manifest = root / "model-lock.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "models": [
                        {
                            "id": conformance.MODEL_ID,
                            "repository": "owner/model",
                            "revision": "a" * 40,
                            "max_download_bytes": entry["size"],
                            "files": [entry],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return checkpoint, fixture, manifest, entry

    def test_locked_checkpoint_and_fixture_are_verified_before_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, fixture, manifest, _ = self.make_inputs(directory)
            completed = mock.Mock(returncode=7)
            with mock.patch.object(
                conformance.subprocess, "run", return_value=completed
            ) as run:
                result = conformance.run_conformance(
                    Path("conformance-bin"),
                    checkpoint,
                    fixture,
                    3,
                    manifest_path=manifest,
                )
            self.assertEqual(result, 7)
            run.assert_called_once_with(
                [
                    "conformance-bin",
                    str(checkpoint),
                    str(fixture),
                    "3",
                ],
                check=False,
            )

    def test_corrupt_checkpoint_is_rejected_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, fixture, manifest, _ = self.make_inputs(directory)
            checkpoint.write_bytes(b"corrupt")
            with mock.patch.object(conformance.subprocess, "run") as run:
                with self.assertRaises(fetch.ModelToolError):
                    conformance.run_conformance(
                        Path("conformance-bin"),
                        checkpoint,
                        fixture,
                        1,
                        manifest_path=manifest,
                    )
            run.assert_not_called()

    def test_mismatched_fixture_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, fixture, manifest, entry = self.make_inputs(directory)
            fixture.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "fixture_size": fixture.stat().st_size,
                        "fixture_sha256": hashlib.sha256(
                            fixture.read_bytes()
                        ).hexdigest(),
                        "checkpoint_size": entry["size"],
                        "checkpoint_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(conformance.subprocess, "run") as run:
                with self.assertRaises(fetch.ModelToolError):
                    conformance.run_conformance(
                        Path("conformance-bin"),
                        checkpoint,
                        fixture,
                        1,
                        manifest_path=manifest,
                    )
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
