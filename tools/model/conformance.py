#!/usr/bin/env python3
"""Verify locked SmolLM2 provenance, then run compiled PR 4 conformance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from tools.model import fetch


MODEL_ID = "smollm2-135m"
CHECKPOINT_NAME = "model.safetensors"


def locked_checkpoint(
    manifest_path: Path = fetch.DEFAULT_MANIFEST,
) -> dict[str, object]:
    manifest = fetch.load_manifest(manifest_path)
    model = fetch.select_model(manifest, MODEL_ID)
    for entry in model["files"]:
        if entry["path"] == CHECKPOINT_NAME:
            return entry
    raise fetch.ModelToolError(
        f"{MODEL_ID} does not lock {CHECKPOINT_NAME}"
    )


def verify_fixture_provenance(
    fixture_path: Path,
    checkpoint_entry: dict[str, object],
) -> None:
    manifest_path = fixture_path.with_suffix(".json")
    with manifest_path.open("r", encoding="utf-8") as source:
        fixture_manifest = json.load(
            source, object_pairs_hook=fetch.reject_duplicate_keys
        )
    if not isinstance(fixture_manifest, dict):
        raise fetch.ModelToolError("fixture manifest must be an object")
    if (
        fixture_manifest.get("schema_version") != 1
        or fixture_manifest.get("fixture_size") != fixture_path.stat().st_size
        or fixture_manifest.get("fixture_sha256")
        != fetch.sha256_file(fixture_path)
        or fixture_manifest.get("checkpoint_size")
        != checkpoint_entry["size"]
        or fixture_manifest.get("checkpoint_sha256")
        != checkpoint_entry["sha256"]
    ):
        raise fetch.ModelToolError(
            "fixture provenance does not match the locked checkpoint"
        )


def run_conformance(
    executable: Path,
    checkpoint: Path,
    fixture: Path,
    repeats: int,
    *,
    manifest_path: Path = fetch.DEFAULT_MANIFEST,
) -> int:
    if repeats < 1 or repeats > 1_000:
        raise fetch.ModelToolError("repeat count must be in [1, 1000]")
    checkpoint_entry = locked_checkpoint(manifest_path)
    fetch.verify_file(checkpoint, checkpoint_entry)
    verify_fixture_provenance(fixture, checkpoint_entry)
    print(
        f"{checkpoint_entry['sha256']}  {checkpoint} "
        "(locked checkpoint verified)",
        flush=True,
    )
    completed = subprocess.run(
        [
            str(executable),
            str(checkpoint),
            str(fixture),
            str(repeats),
        ],
        check=False,
    )
    return completed.returncode


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--manifest", type=Path, default=fetch.DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        return run_conformance(
            arguments.executable,
            arguments.checkpoint,
            arguments.fixture,
            arguments.repeats,
            manifest_path=arguments.manifest,
        )
    except (fetch.ModelToolError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
