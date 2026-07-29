#!/usr/bin/env python3
"""Fetch, verify, or purge only immutable files in models/model-lock.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import urllib.parse
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "models" / "model-lock.json"
MARKER_NAME = ".marketforge-model-cache-v1"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class ModelToolError(RuntimeError):
    pass


def reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ModelToolError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as source:
        manifest = json.load(source, object_pairs_hook=reject_duplicate_keys)
    if manifest.get("schema_version") != 1:
        raise ModelToolError("unsupported model-lock schema")
    return manifest


def select_model(manifest: dict[str, object], model_id: str) -> dict[str, object]:
    models = manifest.get("models")
    if not isinstance(models, list):
        raise ModelToolError("manifest models must be an array")
    for candidate in models:
        if isinstance(candidate, dict) and candidate.get("id") == model_id:
            validate_model_entry(candidate)
            return candidate
    raise ModelToolError(f"model is not locked: {model_id}")


def validate_model_entry(model: dict[str, object]) -> None:
    model_id = model.get("id")
    if not isinstance(model_id, str) or not MODEL_ID_PATTERN.fullmatch(model_id):
        raise ModelToolError("invalid model id")
    revision = model.get("revision")
    if not isinstance(revision, str) or not COMMIT_PATTERN.fullmatch(revision):
        raise ModelToolError("model revision must be a 40-character commit")
    repository = model.get("repository")
    if (
        not isinstance(repository, str)
        or repository.count("/") != 1
        or repository.startswith("/")
        or repository.endswith("/")
    ):
        raise ModelToolError("invalid Hugging Face repository")
    files = model.get("files")
    if not isinstance(files, list) or not files:
        raise ModelToolError("model must allow at least one file")
    total_size = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise ModelToolError("invalid file entry")
        relative = entry.get("path")
        size = entry.get("size")
        digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ModelToolError(f"invalid locked file: {relative!r}")
        total_size += size
    ceiling = model.get("max_download_bytes")
    if not isinstance(ceiling, int) or total_size > ceiling:
        raise ModelToolError("locked files exceed the model download ceiling")


def ensure_external_cache(cache_root: Path, *, create: bool) -> Path:
    resolved = cache_root.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ModelToolError("cache directory must be outside the source tree")

    marker = resolved / MARKER_NAME
    if create:
        if resolved.exists() and not marker.exists() and any(resolved.iterdir()):
            raise ModelToolError(
                "refusing to initialize a non-empty unmarked cache directory"
            )
        resolved.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(
                "marketforge model cache v1\n", encoding="utf-8"
            )
    elif not marker.is_file():
        raise ModelToolError("cache marker is missing; refusing operation")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, entry: dict[str, object]) -> None:
    expected_size = entry["size"]
    expected_hash = entry["sha256"]
    if not path.is_file():
        raise ModelToolError(f"missing cached file: {path.name}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ModelToolError(
            f"size mismatch for {path.name}: {actual_size} != {expected_size}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ModelToolError(f"SHA-256 mismatch for {path.name}")


def model_directory(cache_root: Path, model: dict[str, object]) -> Path:
    return cache_root / str(model["id"]) / str(model["revision"])


def locked_destination(root: Path, relative: str) -> Path:
    candidate = root / relative
    resolved_root = root.resolve()
    resolved_parent = candidate.parent.resolve()
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as error:
        raise ModelToolError("cached path escapes the model directory") from error
    return candidate


def download_file(
    model: dict[str, object],
    entry: dict[str, object],
    destination: Path,
) -> None:
    repository = urllib.parse.quote(str(model["repository"]), safe="/")
    revision = urllib.parse.quote(str(model["revision"]), safe="")
    relative = urllib.parse.quote(str(entry["path"]), safe="/")
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{relative}"
    temporary = destination.with_name(destination.name + ".part")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)

    expected_size = int(entry["size"])
    received = 0
    digest = hashlib.sha256()
    request = urllib.request.Request(
        url, headers={"User-Agent": "MarketForge-model-fetch/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("xb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > expected_size:
                        raise ModelToolError(
                            f"response exceeds locked size for {entry['path']}"
                        )
                    output.write(chunk)
                    digest.update(chunk)
        if received != expected_size:
            raise ModelToolError(
                f"truncated response for {entry['path']}: "
                f"{received} != {expected_size}"
            )
        if digest.hexdigest() != entry["sha256"]:
            raise ModelToolError(f"SHA-256 mismatch for {entry['path']}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def fetch(cache_root: Path, model: dict[str, object]) -> None:
    destination_root = model_directory(cache_root, model)
    for entry in model["files"]:
        destination = locked_destination(
            destination_root, str(entry["path"])
        )
        if destination.exists():
            verify_file(destination, entry)
            print(f"verified cache hit {destination.name}", flush=True)
            continue
        print(
            f"fetching locked file {entry['path']} ({entry['size']} bytes)",
            flush=True,
        )
        download_file(model, entry, destination)
        verify_file(destination, entry)
    print(f"revision {model['revision']} verified at {destination_root}")


def verify(cache_root: Path, model: dict[str, object]) -> None:
    destination_root = model_directory(cache_root, model)
    for entry in model["files"]:
        path = locked_destination(destination_root, str(entry["path"]))
        verify_file(path, entry)
        print(f"{entry['sha256']}  {path}")
    print(f"revision {model['revision']} verified")


def purge(cache_root: Path, model: dict[str, object]) -> None:
    target = model_directory(cache_root, model)
    if not target.exists():
        print(f"nothing to purge: {target}")
        return
    shutil.rmtree(target)
    print(f"purged locked model revision: {target}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fetch", "verify", "purge"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        manifest = load_manifest(arguments.manifest)
        model = select_model(manifest, arguments.model)
        cache_root = ensure_external_cache(
            arguments.cache_dir, create=arguments.action == "fetch"
        )
        if arguments.action == "fetch":
            fetch(cache_root, model)
        elif arguments.action == "verify":
            verify(cache_root, model)
        else:
            purge(cache_root, model)
    except (ModelToolError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
