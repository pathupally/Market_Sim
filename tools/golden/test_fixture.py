import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file

from tools.golden import export_pr3_fixture


class GoldenFixtureTests(unittest.TestCase):
    def test_export_is_byte_reproducible(self) -> None:
        tensors, manifest = export_pr3_fixture.build_fixture()
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.safetensors"
            second = Path(directory) / "second.safetensors"
            for path in (first, second):
                save_file(tensors, path, metadata=manifest["metadata"])
                export_pr3_fixture.canonicalize_safetensors(path)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_versions_and_hash_match_manifest(self) -> None:
        fixture = export_pr3_fixture.DEFAULT_OUTPUT
        manifest_path = export_pr3_fixture.DEFAULT_MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["torch_version"], "2.3.1")
        self.assertEqual(manifest["transformers_version"], "4.40.1")
        self.assertEqual(
            manifest["model_repository"],
            "HuggingFaceTB/SmolLM2-135M",
        )
        self.assertEqual(
            manifest["model_revision"],
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        )
        self.assertEqual(torch.__version__, "2.3.1")
        self.assertEqual(transformers.__version__, "4.40.1")
        self.assertEqual(fixture.stat().st_size, manifest["size"])
        self.assertEqual(
            hashlib.sha256(fixture.read_bytes()).hexdigest(),
            manifest["sha256"],
        )

    def test_fixture_contains_expected_contract(self) -> None:
        with safe_open(
            export_pr3_fixture.DEFAULT_OUTPUT,
            framework="pt",
            device="cpu",
        ) as fixture:
            names = set(fixture.keys())
            self.assertEqual(len(names), 29)
            self.assertIn("expected.after_attention", names)
            self.assertIn("expected.after_mlp", names)
            self.assertIn("expected.attention_probabilities", names)
            self.assertIn("expected.key_cache", names)
            self.assertEqual(
                tuple(fixture.get_tensor("input.hidden").shape),
                (2, 3, 8),
            )
            self.assertEqual(
                fixture.metadata()["transformers_version"], "4.40.1"
            )


if __name__ == "__main__":
    unittest.main()
