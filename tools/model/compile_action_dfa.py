#!/usr/bin/env python3
"""Compile the locked SmolLM2 market action language into C++ constants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys


TOKENIZER_BYTES = 2_104_556
TOKENIZER_SHA256 = (
    "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
)
TOKENIZERS_VERSION = "0.19.1"
VOCABULARY_SIZE = 49_152
ACTION_COUNT = 1_585
MAXIMUM_TOKENS = 12


class CatalogError(RuntimeError):
    pass


@dataclass(frozen=True)
class CatalogRow:
    kind: str
    quantity: int
    tick: int
    text: str
    tokens: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_language() -> list[tuple[str, int, int, str]]:
    actions = [("hold", 0, 0, "HOLD\n")]
    for kind in ("buy_yes", "sell_yes"):
        word = "BUY" if kind == "buy_yes" else "SELL"
        for quantity in range(1, 9):
            for tick in range(1, 100):
                actions.append(
                    (
                        kind,
                        quantity,
                        tick,
                        f"{word} YES {quantity} @ {tick}\n",
                    )
                )
    return actions


def compile_catalog(tokenizer_path: Path) -> list[CatalogRow]:
    if tokenizer_path.stat().st_size != TOKENIZER_BYTES:
        raise CatalogError("tokenizer size does not match the locked artifact")
    if sha256_file(tokenizer_path) != TOKENIZER_SHA256:
        raise CatalogError("tokenizer SHA-256 does not match the lock")

    import tokenizers
    from tokenizers import Tokenizer

    if tokenizers.__version__ != TOKENIZERS_VERSION:
        raise CatalogError(
            f"tokenizers {tokenizers.__version__} != {TOKENIZERS_VERSION}"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size(with_added_tokens=True) != VOCABULARY_SIZE:
        raise CatalogError("tokenizer vocabulary size is not 49152")

    with tokenizer_path.open("r", encoding="utf-8") as source:
        tokenizer_json = json.load(source)
    special_ids = {
        item["id"]
        for item in tokenizer_json.get("added_tokens", [])
        if item.get("special")
    }
    model_unk = tokenizer_json.get("model", {}).get("unk_token")
    unknown_id = (
        tokenizer.token_to_id(model_unk) if isinstance(model_unk, str) else None
    )

    rows: list[CatalogRow] = []
    sequences: dict[tuple[int, ...], str] = {}
    for kind, quantity, tick, text in action_language():
        encoding = tokenizer.encode(text, add_special_tokens=False)
        tokens = tuple(encoding.ids)
        if not tokens or len(tokens) > MAXIMUM_TOKENS:
            raise CatalogError(f"invalid token length for {text!r}")
        if any(token < 0 or token >= VOCABULARY_SIZE for token in tokens):
            raise CatalogError(f"out-of-range token for {text!r}")
        if any(token in special_ids for token in tokens):
            raise CatalogError(f"special token in {text!r}")
        if unknown_id is not None and unknown_id in tokens:
            raise CatalogError(f"unknown token in {text!r}")
        if tokenizer.decode(tokens, skip_special_tokens=False) != text:
            raise CatalogError(f"decode round trip failed for {text!r}")
        if tokens in sequences:
            raise CatalogError(
                f"duplicate token sequence: {sequences[tokens]!r}, {text!r}"
            )
        sequences[tokens] = text
        rows.append(CatalogRow(kind, quantity, tick, text, tokens))

    if len(rows) != ACTION_COUNT:
        raise CatalogError("canonical action count is not 1585")
    ordered = sorted((row.tokens, row.text) for row in rows)
    for index, (tokens, text) in enumerate(ordered):
        for other_tokens, other_text in ordered[index + 1 :]:
            if other_tokens[: len(tokens)] == tokens:
                raise CatalogError(
                    f"prefix collision: {text!r}, {other_text!r}"
                )
            if other_tokens[:1] != tokens[:1]:
                break
    return rows


def render_catalog(rows: list[CatalogRow]) -> str:
    flat_tokens: list[int] = []
    entries: list[str] = []
    for row in rows:
        offset = len(flat_tokens)
        flat_tokens.extend(row.tokens)
        entries.append(
            "    {ActionKind::"
            f"{row.kind}, {row.quantity}, {row.tick}, {offset}, "
            f"{len(row.tokens)}"
            "},"
        )
    token_lines = []
    for start in range(0, len(flat_tokens), 16):
        token_lines.append(
            "    "
            + ", ".join(str(value) for value in flat_tokens[start : start + 16])
            + ","
        )
    maximum = max(len(row.tokens) for row in rows)
    return "\n".join(
        [
            "// Generated by tools/model/compile_action_dfa.py; do not edit.",
            "// clang-format off",
            f"// tokenizer sha256: {TOKENIZER_SHA256}",
            f"// tokenizers: {TOKENIZERS_VERSION}",
            f"// actions: {len(rows)}, max tokens: {maximum}",
            "static constexpr token_id_t kGeneratedTokens[] = {",
            *token_lines,
            "};",
            "",
            "static constexpr GeneratedCatalogEntry kGeneratedEntries[] = {",
            *entries,
            "};",
            "// clang-format on",
            "",
        ]
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        rendered = render_catalog(compile_catalog(arguments.tokenizer))
        if arguments.check:
            if arguments.output.read_text(encoding="utf-8") != rendered:
                raise CatalogError("committed catalog is not reproducible")
            print(f"catalog is byte-identical: {arguments.output}")
        else:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(rendered, encoding="utf-8")
            print(
                f"wrote {ACTION_COUNT} actions to {arguments.output}",
                flush=True,
            )
    except (CatalogError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
