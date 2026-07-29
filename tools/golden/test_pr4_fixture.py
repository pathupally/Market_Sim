from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from safetensors import safe_open


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "golden"
    / "smollm2-pr4-greedy-f32.safetensors"
)
MANIFEST = FIXTURE.with_suffix(".json")


class Pr4FixtureTests(unittest.TestCase):
    def test_manifest_provenance_hash_and_margin_contract(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["torch_version"], "2.3.1")
        self.assertEqual(manifest["transformers_version"], "4.40.1")
        self.assertEqual(manifest["safetensors_version"], "0.4.3")
        self.assertEqual(
            manifest["model_revision"],
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        )
        self.assertEqual(manifest["prompt_token_ids"], [0, 1, 2, 3])
        self.assertEqual(manifest["expected_tokens"], [198, 198, 504])
        self.assertGreaterEqual(min(manifest["logit_margins"]), 0.5)
        self.assertEqual(FIXTURE.stat().st_size, manifest["fixture_size"])
        self.assertEqual(
            hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            manifest["fixture_sha256"],
        )

    def test_fixture_contains_complete_vocabulary_logits(self) -> None:
        with safe_open(FIXTURE, framework="numpy") as fixture:
            self.assertEqual(
                set(fixture.keys()),
                {
                    "input.token_ids",
                    "expected.tokens",
                    "expected.logits",
                    "expected.margins",
                    "tolerance.logits_absolute",
                    "tolerance.logits_relative",
                },
            )
            self.assertEqual(fixture.get_tensor("input.token_ids").shape, (4,))
            self.assertEqual(fixture.get_tensor("expected.tokens").shape, (3,))
            self.assertEqual(
                fixture.get_tensor("expected.logits").shape, (3, 49_152)
            )


if __name__ == "__main__":
    unittest.main()
