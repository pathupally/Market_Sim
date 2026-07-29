"""Bounded, ordered PR-6 CUDA dispatch policy.

This module is deliberately independent of the Modal SDK. Local tests and dry
runs therefore cannot import Modal or contact a remote service. The trusted
host supplies a dispatcher only after all local gates have passed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import hmac
import io
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any, Protocol

from tools.modal.modal_budget import (
    MONTHLY_BUDGET_USD,
    PROJECT_SOFT_CAP_USD,
    RESERVE_USD,
    estimate_compute_cost,
    require_project_headroom,
)

COMPILE_TIMEOUT_SECONDS = 600
GPU_TIMEOUT_SECONDS = 900
PHYSICAL_CORES = Decimal("2")
MEMORY_GIB = Decimal("4")
GPU_KIND = "L4"
MAX_CONTAINERS = 1
CONCURRENCY = 1
TRIAL_GPU_MINUTE_CEILING = Decimal("60")
TRIAL_COMPUTE_CEILING_USD = Decimal("1.00")
TRIAL_RESERVATION_GPU_MINUTES = Decimal("15")
GATE_ID = "pr6-cuda-lifecycle-v1"

COMPILE_COST_CEILING_USD = estimate_compute_cost(
    seconds=COMPILE_TIMEOUT_SECONDS,
    physical_cores=PHYSICAL_CORES,
    memory_gib=MEMORY_GIB,
)
GPU_COST_CEILING_USD = estimate_compute_cost(
    seconds=GPU_TIMEOUT_SECONDS,
    physical_cores=PHYSICAL_CORES,
    memory_gib=MEMORY_GIB,
    gpu=GPU_KIND,
)
CHAIN_COST_CEILING_USD = COMPILE_COST_CEILING_USD + GPU_COST_CEILING_USD

if COMPILE_COST_CEILING_USD != Decimal("0.0210480"):
    raise RuntimeError("locked compile cost ceiling changed")
if GPU_COST_CEILING_USD != Decimal("0.2313720"):
    raise RuntimeError("locked GPU cost ceiling changed")
if CHAIN_COST_CEILING_USD != Decimal("0.2524200"):
    raise RuntimeError("locked chain cost ceiling changed")


@dataclass(frozen=True)
class Stage:
    name: str
    gpu: str | None
    physical_cores: int
    memory_gib: int
    timeout_seconds: int
    max_containers: int
    concurrency: int
    single_use: bool
    maximum_compute_cost_usd: str


STAGES = (
    Stage(
        name="cuda_compile",
        gpu=None,
        physical_cores=2,
        memory_gib=4,
        timeout_seconds=COMPILE_TIMEOUT_SECONDS,
        max_containers=MAX_CONTAINERS,
        concurrency=CONCURRENCY,
        single_use=True,
        maximum_compute_cost_usd=f"{COMPILE_COST_CEILING_USD:.6f}",
    ),
    Stage(
        name="gpu_smoke",
        gpu=GPU_KIND,
        physical_cores=2,
        memory_gib=4,
        timeout_seconds=GPU_TIMEOUT_SECONDS,
        max_containers=MAX_CONTAINERS,
        concurrency=CONCURRENCY,
        single_use=True,
        maximum_compute_cost_usd=f"{GPU_COST_CEILING_USD:.6f}",
    ),
)


class Dispatcher(Protocol):
    def dispatch_compile(self) -> dict[str, object]: ...

    def dispatch_gpu(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class SourceBundle:
    """A clean, immutable Git archive passed identically to both stages."""

    commit: str
    sha256: str
    content: bytes


def create_source_bundle(project_root: Path) -> SourceBundle:
    """Archive HEAD only after two clean-worktree checks guard against TOCTOU."""

    root = project_root.resolve()

    def git(*arguments: str, text: bool = False) -> bytes | str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return completed.stdout

    before = git("status", "--porcelain=v1", "--untracked-files=all", text=True)
    if before:
        raise RuntimeError("source worktree must be clean before bundling")
    commit = str(git("rev-parse", "--verify", "HEAD", text=True)).strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("source commit is not a full lowercase Git object id")
    content = git("archive", "--format=tar", commit)
    if not isinstance(content, bytes) or not content:
        raise RuntimeError("Git produced an empty source bundle")
    after_commit = str(
        git("rev-parse", "--verify", "HEAD", text=True)
    ).strip()
    after_status = git(
        "status", "--porcelain=v1", "--untracked-files=all", text=True
    )
    if after_commit != commit or after_status:
        raise RuntimeError("source changed while the immutable bundle was created")
    return SourceBundle(
        commit=commit,
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )


def strict_json_loads(text: str) -> object:
    """Reject duplicate keys and JavaScript non-finite number extensions."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def source_bundle_member_content(
    bundle: SourceBundle, member_name: str
) -> bytes:
    if hashlib.sha256(bundle.content).hexdigest() != bundle.sha256:
        raise RuntimeError("source bundle content does not match its SHA-256")
    with tarfile.open(fileobj=io.BytesIO(bundle.content), mode="r:") as archive:
        member = archive.getmember(member_name)
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(f"source bundle member is not a file: {member_name}")
        return source.read()


def source_bundle_member_sha256(bundle: SourceBundle, member_name: str) -> str:
    return hashlib.sha256(
        source_bundle_member_content(bundle, member_name)
    ).hexdigest()


def extract_source_bundle(bundle: SourceBundle, destination: Path) -> None:
    if hashlib.sha256(bundle.content).hexdigest() != bundle.sha256:
        raise RuntimeError("source bundle content does not match its SHA-256")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(bundle.content), mode="r:") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (member.isdir() or member.isfile())
            ):
                raise RuntimeError(f"unsafe source bundle member: {member.name}")
        archive.extractall(destination, filter="data")


def require_unchanged_source_bundle(
    project_root: Path, expected: SourceBundle
) -> None:
    observed = create_source_bundle(project_root)
    if (
        observed.commit != expected.commit
        or observed.sha256 != expected.sha256
        or observed.content != expected.content
    ):
        raise RuntimeError("candidate commit or source bundle changed during gates")


def scan_forbidden_artifacts(project_root: Path) -> None:
    """Reject local artifacts that must never enter authorization or evidence."""

    root = project_root.resolve()
    allowed_safetensors = {
        Path("tests/fixtures/golden/smollm2-pr4-greedy-f32.safetensors"),
        Path("tests/fixtures/golden/smollm2-tiny-layer-f32.safetensors"),
    }
    forbidden: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".git" in relative.parts or ".venv" in relative.parts:
            continue
        if path.is_dir():
            if (
                path.name == "__pycache__"
                or path.name == "out"
                or path.name == "build"
                or path.name.startswith("build-")
            ):
                forbidden.append(relative)
            continue
        suffix = path.suffix.lower()
        if (
            suffix in {
                ".pyc",
                ".pyo",
                ".profraw",
                ".profdata",
                ".nsys-rep",
                ".ncu-rep",
                ".gguf",
                ".bin",
            }
            or path.name == "tokenizer.json"
            or (
                suffix == ".safetensors"
                and relative not in allowed_safetensors
            )
            or path.name
            in {
                "CMakeCache.txt",
                "build.ninja",
                "compile_commands.json",
                ".ninja_deps",
                ".ninja_log",
            }
        ):
            forbidden.append(relative)
    if forbidden:
        rendered = ", ".join(str(path) for path in sorted(forbidden))
        raise RuntimeError(f"forbidden worktree artifacts: {rendered}")


def _state_root(root: Path | None = None) -> Path:
    return (
        root
        if root is not None
        else Path.home() / ".cache" / "marketforge" / "modal-evidence"
    )


def modal_executable_for(python_executable: Path) -> Path:
    """Preserve the invoking environment instead of resolving venv symlinks."""

    return python_executable.parent / "modal"


def require_external_tokenizer(
    project_root: Path, tokenizer_json: Path
) -> Path:
    """Require the locked tokenizer to exist outside the source checkout."""

    root = project_root.resolve()
    tokenizer = tokenizer_json.resolve()
    if not tokenizer.is_file():
        raise RuntimeError(
            "the external locked tokenizer.json is mandatory for --launch"
        )
    try:
        tokenizer.relative_to(root)
    except ValueError:
        return tokenizer
    raise RuntimeError("tokenizer.json must remain outside the Git checkout")


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with os.fdopen(descriptor, "r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    finally:
        # fdopen owns and closes descriptor after a successful construction.
        pass


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    encoded = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = os.open(
        temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{secrets.token_hex(8)}.tmp"
    )
    descriptor = os.open(
        temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _state_entry_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _private_state_key(path: Path, *, create: bool) -> bytes:
    if not _state_entry_exists(path):
        if not create:
            raise RuntimeError(f"persistent state key is missing: {path.name}")
        _atomic_bytes(path, secrets.token_bytes(32))
    if path.is_symlink():
        raise RuntimeError(f"persistent state key is not a regular file: {path.name}")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise RuntimeError(f"persistent state key permissions are invalid: {path.name}")
    key = path.read_bytes()
    if len(key) != 32:
        raise RuntimeError(f"persistent state key has an invalid size: {path.name}")
    return key


class TrialLedger:
    """Serialized, conservative PR-6 trial reservations outside Git."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = _state_root(root)
        self.path = self.root / "pr6-trial-ledger.json"
        self.lock_path = self.root / "pr6-trial-ledger.lock"
        self.key_path = self.root / "pr6-trial-ledger.key"

    @staticmethod
    def _empty() -> dict[str, object]:
        return {"schema_version": 1, "reservations": []}

    @staticmethod
    def _signature(
        payload: dict[str, object], key: bytes
    ) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hmac.new(key, encoded, hashlib.sha256).hexdigest()

    def _ensure_state(self) -> bytes:
        key_exists = _state_entry_exists(self.key_path)
        ledger_exists = _state_entry_exists(self.path)
        if key_exists != ledger_exists:
            raise RuntimeError(
                "trial ledger/key state loss or tampering detected"
            )
        if key_exists:
            return _private_state_key(self.key_path, create=False)

        key = _private_state_key(self.key_path, create=True)
        self._write(self._empty(), key=key)
        return key

    def _write(
        self,
        payload: dict[str, object],
        *,
        key: bytes | None = None,
    ) -> None:
        signing_key = (
            key
            if key is not None
            else _private_state_key(self.key_path, create=False)
        )
        _atomic_json(
            self.path,
            {
                **payload,
                "signature": self._signature(payload, signing_key),
            },
        )

    def _read(self) -> dict[str, object]:
        key = self._ensure_state()
        if self.path.is_symlink():
            raise RuntimeError("trial ledger is not a regular state file")
        metadata = self.path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise RuntimeError("trial ledger state-file permissions are invalid")
        value = strict_json_loads(self.path.read_text(encoding="utf-8"))
        if type(value) is not dict or set(value) != {
            "schema_version",
            "reservations",
            "signature",
        }:
            raise ValueError("trial ledger has an invalid schema")
        signature = value["signature"]
        payload = {
            "schema_version": value["schema_version"],
            "reservations": value["reservations"],
        }
        if (
            type(signature) is not str
            or len(signature) != 64
            or any(
                character not in "0123456789abcdef"
                for character in signature
            )
            or not hmac.compare_digest(
                signature, self._signature(payload, key)
            )
        ):
            raise ValueError("trial ledger authentication failed")
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("trial ledger schema version is invalid")
        if type(value["reservations"]) is not list:
            raise ValueError("trial ledger reservations must be an array")
        required = {
            "reservation_id",
            "candidate_commit",
            "gate_id",
            "reserved_cost_usd",
            "reserved_gpu_minutes",
            "status",
        }
        reservation_ids: set[str] = set()
        for item in value["reservations"]:
            if type(item) is not dict or set(item) != required:
                raise ValueError("trial ledger reservation is malformed")
            reservation_id = item["reservation_id"]
            if (
                type(reservation_id) is not str
                or len(reservation_id) != 32
                or any(
                    character not in "0123456789abcdef"
                    for character in reservation_id
                )
                or reservation_id in reservation_ids
            ):
                raise ValueError("trial ledger reservation_id is invalid")
            reservation_ids.add(reservation_id)
            candidate_commit = item["candidate_commit"]
            if (
                type(candidate_commit) is not str
                or len(candidate_commit) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in candidate_commit
                )
            ):
                raise ValueError("trial ledger candidate_commit is invalid")
            if type(item["gate_id"]) is not str or item["gate_id"] != GATE_ID:
                raise ValueError("trial ledger gate_id is invalid")
            if (
                type(item["status"]) is not str
                or item["status"] not in {"reserved", "passed", "failed"}
            ):
                raise ValueError("trial ledger status is invalid")
            if (
                type(item["reserved_cost_usd"]) is not str
                or type(item["reserved_gpu_minutes"]) is not str
                or item["reserved_cost_usd"]
                != f"{CHAIN_COST_CEILING_USD:.6f}"
                or item["reserved_gpu_minutes"]
                != str(TRIAL_RESERVATION_GPU_MINUTES)
            ):
                raise ValueError("trial ledger reservation costs are invalid")
        return payload

    def reserve(self, *, candidate_commit: str, gate_id: str) -> str:
        if (
            len(candidate_commit) != 40
            or any(character not in "0123456789abcdef" for character in candidate_commit)
            or gate_id != GATE_ID
        ):
            raise ValueError("trial reservation provenance is invalid")
        with _exclusive_lock(self.lock_path):
            ledger = self._read()
            reservations = ledger["reservations"]
            matching_statuses = {
                item["status"]
                for item in reservations
                if item["candidate_commit"] == candidate_commit
                and item["gate_id"] == gate_id
            }
            if "passed" in matching_statuses:
                raise RuntimeError(
                    "an identical PR-6 candidate already passed permanently"
                )
            if "reserved" in matching_statuses:
                raise RuntimeError(
                    "an identical PR-6 candidate is already being dispatched"
                )
            spent = sum(
                (
                    parse_cost(item["reserved_cost_usd"])
                    for item in reservations
                ),
                Decimal("0"),
            )
            minutes = sum(
                (
                    parse_cost(item["reserved_gpu_minutes"])
                    for item in reservations
                ),
                Decimal("0"),
            )
            if spent + CHAIN_COST_CEILING_USD > TRIAL_COMPUTE_CEILING_USD:
                raise RuntimeError("PR-6 cumulative $1 trial ceiling exhausted")
            if (
                minutes + TRIAL_RESERVATION_GPU_MINUTES
                > TRIAL_GPU_MINUTE_CEILING
            ):
                raise RuntimeError(
                    "PR-6 cumulative 60-L4-minute trial ceiling exhausted"
                )
            reservation_id = secrets.token_hex(16)
            reservations.append(
                {
                    "reservation_id": reservation_id,
                    "candidate_commit": candidate_commit,
                    "gate_id": gate_id,
                    "reserved_cost_usd": f"{CHAIN_COST_CEILING_USD:.6f}",
                    "reserved_gpu_minutes": str(
                        TRIAL_RESERVATION_GPU_MINUTES
                    ),
                    "status": "reserved",
                }
            )
            self._write(ledger)
            return reservation_id

    def finish(self, reservation_id: str, *, passed: bool) -> None:
        with _exclusive_lock(self.lock_path):
            ledger = self._read()
            matches = [
                item
                for item in ledger["reservations"]
                if item["reservation_id"] == reservation_id
            ]
            if len(matches) != 1 or matches[0]["status"] != "reserved":
                raise RuntimeError("trial reservation is absent or already finished")
            matches[0]["status"] = "passed" if passed else "failed"
            self._write(ledger)

    def require_reserved(
        self,
        reservation_id: str,
        *,
        candidate_commit: str,
        gate_id: str,
    ) -> None:
        with _exclusive_lock(self.lock_path):
            ledger = self._read()
            matches = [
                item
                for item in ledger["reservations"]
                if item["reservation_id"] == reservation_id
            ]
            if (
                len(matches) != 1
                or matches[0]["status"] != "reserved"
                or matches[0]["candidate_commit"] != candidate_commit
                or matches[0]["gate_id"] != gate_id
            ):
                raise RuntimeError("authorization has no matching trial reservation")


class AuthorizationTicketStore:
    """HMAC-authenticated, one-use launcher tickets outside Git."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = _state_root(root)
        self.key_path = self.root / "authorization.key"
        self.lock_path = self.root / "authorization.lock"

    def _key(self) -> bytes:
        return _private_state_key(self.key_path, create=True)

    def _signature(self, payload: dict[str, object]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hmac.new(self._key(), encoded, hashlib.sha256).hexdigest()

    def issue(
        self,
        *,
        source_bundle: SourceBundle,
        dependency_lock_sha256: str,
        month_to_date_usd: Decimal,
        reservation_id: str,
    ) -> Path:
        with _exclusive_lock(self.lock_path):
            payload: dict[str, object] = {
                "schema_version": 1,
                "candidate_commit": source_bundle.commit,
                "source_bundle_sha256": source_bundle.sha256,
                "dependency_lock_sha256": dependency_lock_sha256,
                "month_to_date_usd": str(month_to_date_usd),
                "gate_id": GATE_ID,
                "reservation_id": reservation_id,
                "nonce": secrets.token_hex(16),
                "expires_unix": int(time.time()) + 900,
                "consumed": False,
            }
            ticket = {**payload, "signature": self._signature(payload)}
            path = self.root / f"authorization-{payload['nonce']}.json"
            _atomic_json(path, ticket)
            return path

    def consume(self, path: Path) -> dict[str, object]:
        requested = path.resolve()
        if requested.parent != self.root.resolve():
            raise RuntimeError("authorization ticket is outside the trusted state root")
        with _exclusive_lock(self.lock_path):
            value = strict_json_loads(requested.read_text(encoding="utf-8"))
            required = {
                "schema_version",
                "candidate_commit",
                "source_bundle_sha256",
                "dependency_lock_sha256",
                "month_to_date_usd",
                "gate_id",
                "reservation_id",
                "nonce",
                "expires_unix",
                "consumed",
                "signature",
            }
            if type(value) is not dict or set(value) != required:
                raise RuntimeError("authorization ticket has an invalid schema")
            payload = {
                key: value[key] for key in required if key != "signature"
            }
            if (
                type(value["schema_version"]) is not int
                or value["schema_version"] != 1
                or type(value["expires_unix"]) is not int
                or value["expires_unix"] < int(time.time())
                or type(value["consumed"]) is not bool
                or value["consumed"]
                or type(value["signature"]) is not str
                or not hmac.compare_digest(
                    value["signature"], self._signature(payload)
                )
            ):
                raise RuntimeError("authorization ticket is invalid or consumed")
            for field in (
                "candidate_commit",
                "source_bundle_sha256",
                "dependency_lock_sha256",
                "month_to_date_usd",
                "gate_id",
                "reservation_id",
                "nonce",
            ):
                if type(value[field]) is not str or not value[field]:
                    raise RuntimeError(f"authorization ticket {field} is invalid")
            if (
                len(value["candidate_commit"]) != 40
                or any(
                    character not in "0123456789abcdef"
                    for character in value["candidate_commit"]
                )
                or len(value["source_bundle_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value["source_bundle_sha256"]
                )
                or len(value["dependency_lock_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in value["dependency_lock_sha256"]
                )
                or value["gate_id"] != GATE_ID
            ):
                raise RuntimeError("authorization ticket provenance is invalid")
            parse_cost(value["month_to_date_usd"])
            consumed_payload = {**payload, "consumed": True}
            consumed = {
                **consumed_payload,
                "signature": self._signature(consumed_payload),
            }
            _atomic_json(requested, consumed)
            return payload


class EvidenceCache:
    """Small accepted-evidence cache stored outside the source checkout."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = _state_root(root)

    @staticmethod
    def _key(candidate_commit: str, gate_id: str) -> str:
        return hashlib.sha256(
            f"{candidate_commit}\0{gate_id}".encode("utf-8")
        ).hexdigest()

    def _path(self, candidate_commit: str, gate_id: str) -> Path:
        return self.root / f"{self._key(candidate_commit, gate_id)}.json"

    def load(
        self, *, candidate_commit: str, gate_id: str
    ) -> dict[str, object] | None:
        if (
            len(candidate_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in candidate_commit
            )
            or gate_id != GATE_ID
        ):
            raise ValueError("cache provenance key is invalid")
        path = self._path(candidate_commit, gate_id)
        if not path.is_file():
            return None
        value = strict_json_loads(path.read_text(encoding="utf-8"))
        if type(value) is not dict:
            raise ValueError("cached evidence record must be an object")
        expected_keys = {
            "schema_version",
            "accepted",
            "candidate_commit",
            "gate_id",
            "manifest",
        }
        if set(value) != expected_keys:
            raise ValueError("cached evidence record has an invalid schema")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["accepted"] is not True
            or value["candidate_commit"] != candidate_commit
            or value["gate_id"] != gate_id
            or type(value["manifest"]) is not dict
        ):
            raise ValueError("cached evidence does not match the requested gate")
        return value["manifest"]

    def store(
        self,
        *,
        candidate_commit: str,
        gate_id: str,
        manifest: dict[str, object],
        lock: object,
        expected_source_bundle_sha256: str,
        expected_dependency_lock_sha256: str,
    ) -> Path:
        if (
            len(candidate_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in candidate_commit
            )
            or gate_id != GATE_ID
        ):
            raise ValueError("cache provenance key is invalid")
        if (
            type(manifest) is not dict
            or type(manifest.get("schema_version")) is not int
            or manifest.get("schema_version") != 1
            or manifest.get("result") != "pass"
            or len(manifest) < 12
        ):
            raise ValueError("only complete passing evidence is cacheable")
        from tools.modal.cuda_evidence import validate_manifest

        validate_manifest(
            manifest,
            lock,
            expected_source_commit=candidate_commit,
            expected_source_bundle_sha256=expected_source_bundle_sha256,
            expected_dependency_lock_sha256=expected_dependency_lock_sha256,
        )
        record = {
            "schema_version": 1,
            "accepted": True,
            "candidate_commit": candidate_commit,
            "gate_id": gate_id,
            "manifest": manifest,
        }
        destination = self._path(candidate_commit, gate_id)
        _atomic_json(destination, record)
        return destination


def parse_cost(value: str) -> Decimal:
    """Parse a mandatory operator-supplied finite, non-negative USD value."""
    if not value or value.strip() != value:
        raise ValueError("month-to-date cost must be a nonempty canonical value")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("month-to-date cost is malformed") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("month-to-date cost must be finite and non-negative")
    return parsed


def dry_run_manifest(month_to_date_usd: Decimal) -> dict[str, object]:
    """Return the complete no-dispatch plan after the combined-chain preflight."""
    require_project_headroom(
        month_to_date_usd=month_to_date_usd,
        planned_cost_usd=CHAIN_COST_CEILING_USD,
    )
    return {
        "schema_version": 1,
        "mode": "dry-run",
        "dispatched": False,
        "ordered_stages": [asdict(stage) for stage in STAGES],
        "budget": {
            "month_to_date_usd": str(month_to_date_usd),
            "monthly_budget_usd": str(MONTHLY_BUDGET_USD),
            "project_soft_cap_usd": str(PROJECT_SOFT_CAP_USD),
            "reserve_usd": str(RESERVE_USD),
            "maximum_planned_cost_usd": f"{CHAIN_COST_CEILING_USD:.6f}",
            "trial_compute_ceiling_usd": str(TRIAL_COMPUTE_CEILING_USD),
            "trial_gpu_minute_ceiling": str(TRIAL_GPU_MINUTE_CEILING),
            "trial_reservation_gpu_minutes": str(
                TRIAL_RESERVATION_GPU_MINUTES
            ),
        },
    }


def run_ordered(
    *,
    month_to_date_usd: Decimal,
    dispatcher: Dispatcher,
    local_gates_passed: bool,
) -> dict[str, object]:
    """Dispatch compile then GPU, stopping before GPU on any failed gate."""
    plan = dry_run_manifest(month_to_date_usd)
    if not local_gates_passed:
        raise RuntimeError("all local gates must pass before remote dispatch")

    compile_result = dispatcher.dispatch_compile()
    if compile_result.get("result") != "pass":
        raise RuntimeError("cuda_compile did not pass; gpu_smoke not dispatched")
    gpu_result = dispatcher.dispatch_gpu()
    if gpu_result.get("result") != "pass":
        raise RuntimeError("gpu_smoke did not pass")
    return {
        **plan,
        "mode": "remote",
        "dispatched": True,
        "results": {
            "cuda_compile": compile_result,
            "gpu_smoke": gpu_result,
        },
    }


def run_local_gates(
    source_bundle: SourceBundle,
    python_executable: Path,
    tokenizer_json: Path,
) -> None:
    """Run the preserved matrix against the exact immutable archive."""

    tokenizer = tokenizer_json.resolve()
    if not tokenizer.is_file():
        raise RuntimeError(
            "the external locked tokenizer.json is mandatory for --launch"
        )
    environment = {
        **os.environ,
        "CUDACXX": "/definitely/not/a/cuda/compiler",
        "CTEST_OUTPUT_ON_FAILURE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MARKETFORGE_TOKENIZER_JSON": str(tokenizer),
    }

    with tempfile.TemporaryDirectory(prefix="marketforge-pr6-gates-") as temp:
        temporary = Path(temp)
        root = temporary / "source"
        extract_source_bundle(source_bundle, root)

        def checked(command: list[str], *, cwd: Path = root) -> None:
            subprocess.run(command, cwd=cwd, env=environment, check=True)

        builds: dict[str, Path] = {}
        for name, build_type, sanitizers in (
            ("debug", "Debug", False),
            ("release", "Release", False),
            ("sanitize", "Debug", True),
        ):
            build = temporary / name
            builds[name] = build
            checked(
                [
                    "cmake",
                    "-S",
                    str(root),
                    "-B",
                    str(build),
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    "-DMARKETFORGE_ENABLE_CUDA=OFF",
                    "-DMARKETFORGE_WARNINGS_AS_ERRORS=ON",
                    (
                        "-DMARKETFORGE_ENABLE_SANITIZERS=ON"
                        if sanitizers
                        else "-DMARKETFORGE_ENABLE_SANITIZERS=OFF"
                    ),
                ]
            )
            checked(["cmake", "--build", str(build), "--parallel", "2"])
            checked(
                [
                    "ctest",
                    "--test-dir",
                    str(build),
                    "--output-on-failure",
                ]
            )

        checked(
            [
                str(python_executable),
                "-B",
                "-m",
                "unittest",
                "discover",
                "-s",
                ".",
                "-p",
                "test_*.py",
            ]
        )
        checked(
            [
                str(python_executable),
                "-B",
                "-m",
                "tools.model.compile_action_dfa",
                "--tokenizer",
                str(tokenizer),
                "--output",
                str(
                    root
                    / "src/grammar/generated/smollm2_market_action_v1.inc"
                ),
                "--check",
            ]
        )

        checkpoint = (
            Path.home()
            / "Library/Caches/marketforge/smollm2-135m"
            / "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
            / "model.safetensors"
        )
        fixture = (
            root
            / "tests/fixtures/golden/smollm2-pr4-greedy-f32.safetensors"
        )
        if checkpoint.is_file():
            for build in builds.values():
                checked(
                    [
                        str(python_executable),
                        "-B",
                        "-m",
                        "tools.model.conformance",
                        "--executable",
                        str(build / "marketforge_smollm2_conformance"),
                        "--checkpoint",
                        str(checkpoint),
                        "--fixture",
                        str(fixture),
                    ]
                )

        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".cpp", ".hpp", ".cu", ".cuh"}
        )
        formatter = shutil.which("clang-format")
        if formatter is None and shutil.which("xcrun") is not None:
            located = subprocess.run(
                ["xcrun", "--find", "clang-format"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
            ).stdout.strip()
            formatter = located or None
        if formatter is None:
            raise RuntimeError("clang-format is required for PR-6 authorization")
        if files:
            checked(
                [formatter, "--dry-run", "--Werror", *map(str, files)]
            )


def _launch(
    *,
    project_root: Path,
    python_executable: Path,
    tokenizer_json: Path,
    month_to_date: Decimal,
) -> int:
    from tools.modal.cuda_evidence import validate_manifest

    scan_forbidden_artifacts(project_root)
    tokenizer = require_external_tokenizer(project_root, tokenizer_json)
    source_bundle = create_source_bundle(project_root)
    run_local_gates(source_bundle, python_executable, tokenizer)
    scan_forbidden_artifacts(project_root)
    require_unchanged_source_bundle(project_root, source_bundle)
    dependency_hash = source_bundle_member_sha256(
        source_bundle, "tools/modal/requirements.txt"
    )
    lock = strict_json_loads(
        source_bundle_member_content(
            source_bundle, "tools/modal/cuda-toolchain-lock.json"
        ).decode("utf-8")
    )

    def validate(value: dict[str, object]) -> None:
        validate_manifest(
            value,
            lock,
            expected_source_commit=source_bundle.commit,
            expected_source_bundle_sha256=source_bundle.sha256,
            expected_dependency_lock_sha256=dependency_hash,
        )

    cache = EvidenceCache()
    cached = cache.load(
        candidate_commit=source_bundle.commit, gate_id=GATE_ID
    )
    if cached is not None:
        validate(cached)
        print(json.dumps(cached, indent=2, sort_keys=True, allow_nan=False))
        return 0

    ledger = TrialLedger()
    reservation_id = ledger.reserve(
        candidate_commit=source_bundle.commit, gate_id=GATE_ID
    )
    try:
        ticket = AuthorizationTicketStore().issue(
            source_bundle=source_bundle,
            dependency_lock_sha256=dependency_hash,
            month_to_date_usd=month_to_date,
            reservation_id=reservation_id,
        )
    except Exception:
        ledger.finish(reservation_id, passed=False)
        raise
    modal_executable = modal_executable_for(python_executable)
    if not modal_executable.is_file():
        ledger.finish(reservation_id, passed=False)
        raise RuntimeError(f"Modal executable is absent: {modal_executable}")
    passed = False
    try:
        completed = subprocess.run(
            [
                str(modal_executable),
                "run",
                "-m",
                "tools.modal.cuda_modal_app",
                "--authorization-ticket",
                str(ticket),
            ],
            cwd=project_root,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"authorized Modal application exited {completed.returncode}"
            )
        cached = cache.load(
            candidate_commit=source_bundle.commit, gate_id=GATE_ID
        )
        if cached is None:
            raise RuntimeError("Modal application returned without cached evidence")
        validate(cached)
        passed = True
        return 0
    finally:
        ledger.finish(reservation_id, passed=passed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--month-to-date-usd",
        required=True,
        help="current Modal project spend; no default is permitted",
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        help=(
            "absolute path to the external locked tokenizer.json; mandatory "
            "for --launch and unused by --dry-run"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="pure local budget plan; imports no Modal code and dispatches nothing",
    )
    mode.add_argument(
        "--launch",
        action="store_true",
        help="run all local gates, reserve the trial, then start Modal",
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    try:
        month_to_date = parse_cost(args.month_to_date_usd)
        plan = dry_run_manifest(month_to_date)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.tokenizer_json is None:
        raise SystemExit("--tokenizer-json is mandatory for --launch")
    try:
        return _launch(
            project_root=Path(__file__).resolve().parents[2],
            python_executable=Path(sys.executable),
            tokenizer_json=args.tokenizer_json,
            month_to_date=month_to_date,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    raise SystemExit(main())
