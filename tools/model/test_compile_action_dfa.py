from pathlib import Path
import os
import tempfile
import unittest

from tools.model import compile_action_dfa


class ActionCatalogTests(unittest.TestCase):
    def test_language_is_complete_and_stably_ordered(self) -> None:
        language = compile_action_dfa.action_language()
        self.assertEqual(len(language), 1_585)
        self.assertEqual(language[0], ("hold", 0, 0, "HOLD\n"))
        self.assertEqual(language[1], ("buy_yes", 1, 1, "BUY YES 1 @ 1\n"))
        self.assertEqual(
            language[-1], ("sell_yes", 8, 99, "SELL YES 8 @ 99\n")
        )

    def test_renderer_is_deterministic(self) -> None:
        rows = [
            compile_action_dfa.CatalogRow("hold", 0, 0, "HOLD\n", (1, 2)),
            compile_action_dfa.CatalogRow(
                "buy_yes", 1, 2, "BUY YES 1 @ 2\n", (3, 4, 5)
            ),
        ]
        first = compile_action_dfa.render_catalog(rows)
        second = compile_action_dfa.render_catalog(list(rows))
        self.assertEqual(first, second)
        self.assertIn("actions: 2, max tokens: 3", first)

    def test_opt_in_locked_tokenizer_matches_committed_catalog(self) -> None:
        tokenizer = os.environ.get("MARKETFORGE_TOKENIZER_JSON")
        if tokenizer is None:
            self.skipTest("MARKETFORGE_TOKENIZER_JSON is not set")
        output = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "grammar"
            / "generated"
            / "smollm2_market_action_v1.inc"
        )
        expected = output.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory) / "catalog.inc"
            generated.write_text(
                compile_action_dfa.render_catalog(
                    compile_action_dfa.compile_catalog(Path(tokenizer))
                ),
                encoding="utf-8",
            )
            self.assertEqual(generated.read_text(encoding="utf-8"), expected)


if __name__ == "__main__":
    unittest.main()
