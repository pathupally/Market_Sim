"""Strict schema-version-1 validator for accepted PR-6 CUDA evidence."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from typing import Any

from tools.modal.cuda_ci import (
    CHAIN_COST_CEILING_USD,
    COMPILE_TIMEOUT_SECONDS,
    GPU_TIMEOUT_SECONDS,
    TRIAL_COMPUTE_CEILING_USD,
    TRIAL_GPU_MINUTE_CEILING,
    strict_json_loads,
)
from tools.modal.modal_budget import (
    CPU_CORE_SECOND_USD,
    GPU_SECOND_USD,
    MEMORY_GIB_SECOND_USD,
)

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_LENGTHS = [0, 1, 255, 256, 257, 1025]
EXPECTED_REGISTRY_LOCK = {
    "reference": (
        "nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:"
        "392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a"
    ),
    "index_digest": (
        "sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a"
    ),
    "linux_amd64_digest": (
        "sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade"
    ),
    "operating_system": "ubuntu24.04",
    "platform": "linux/amd64",
}
EXPECTED_TOOLCHAIN_LOCK = {
    "cuda": "12.6.3",
    "cmake": "3.30.5",
    "ninja_distribution": "1.11.1.1",
    "ninja_binary": "1.11.1.git.kitware.jobserver-1",
    "cpp_standard": 20,
    "cuda_standard": 20,
    "cuda_architectures": [89],
    "compatibility_policy": {
        "host_compiler": {"identity": "gcc", "major": 13},
        "nvcc": {
            "identity": "nvcc",
            "cuda_major": 12,
            "cuda_minor": 6,
        },
        "cuda_runtime": {
            "identity": "cudart",
            "cuda_major": 12,
            "cuda_minor": 6,
        },
        "driver": {
            "identity": "nvidia-driver",
            "minimum_cuda_api_major": 12,
            "minimum_cuda_api_minor": 6,
        },
        "cublas": {"identity": "cublas", "major": 12},
    },
}
EXPECTED_MODAL_LOCK = {"sdk": "1.3.5", "gpu": "L4", "max_containers": 1}


class ValidationError(ValueError):
    pass


def _object(
    value: object, path: str, required_keys: set[str]
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValidationError(f"{path} must be an object")
    keys = set(value)
    missing = required_keys - keys
    extra = keys - required_keys
    if missing:
        raise ValidationError(f"{path} missing fields: {sorted(missing)}")
    if extra:
        raise ValidationError(f"{path} has unknown fields: {sorted(extra)}")
    return value


def _value_type(value: object, expected: type, path: str) -> None:
    if type(value) is not expected:
        raise ValidationError(f"{path} must be {expected.__name__}")


def _exact_json_equal(observed: object, expected: object) -> bool:
    """Compare recursively without Python's bool/int or int/float aliases."""

    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        if set(observed) != set(expected):
            return False
        return all(
            _exact_json_equal(observed[key], expected[key])
            for key in expected
        )
    if type(expected) is list:
        return len(observed) == len(expected) and all(
            _exact_json_equal(observed_item, expected_item)
            for observed_item, expected_item in zip(observed, expected)
        )
    return observed == expected


def _nonempty_string(value: object, path: str) -> str:
    _value_type(value, str, path)
    if not value:
        raise ValidationError(f"{path} must be nonempty")
    return value


def _sha256(value: object, path: str) -> str:
    text = _nonempty_string(value, path)
    if SHA256_PATTERN.fullmatch(text) is None:
        raise ValidationError(f"{path} must be a lowercase SHA-256")
    return text


def _digest(value: object, path: str) -> str:
    text = _nonempty_string(value, path)
    if not text.startswith("sha256:") or SHA256_PATTERN.fullmatch(text[7:]) is None:
        raise ValidationError(f"{path} must be a sha256: digest")
    return text


def _decimal(value: object, path: str) -> Decimal:
    text = _nonempty_string(value, path)
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValidationError(f"{path} must be decimal text") from error
    if not number.is_finite() or number < 0:
        raise ValidationError(f"{path} must be finite and non-negative")
    return number


def _string_list(value: object, path: str) -> list[str]:
    if type(value) is not list or not value:
        raise ValidationError(f"{path} must be a nonempty array")
    for index, item in enumerate(value):
        _nonempty_string(item, f"{path}[{index}]")
    return value


def _validate_profiler(value: object, path: str) -> None:
    profiler = _object(
        value,
        path,
        {
            "availability",
            "command",
            "version",
            "exit_status",
            "reason",
            "capture_sha256",
            "verification_command",
            "verification_exit_status",
            "evidence",
        },
    )
    availability = _nonempty_string(
        profiler["availability"], f"{path}.availability"
    )
    if availability not in {"available", "unavailable"}:
        raise ValidationError(f"{path}.availability has an unknown enum")
    _string_list(profiler["command"], f"{path}.command")
    if profiler["version"] is not None:
        _nonempty_string(profiler["version"], f"{path}.version")
    _value_type(profiler["exit_status"], int, f"{path}.exit_status")
    verification_command = _string_list(
        profiler["verification_command"], f"{path}.verification_command"
    )
    _value_type(
        profiler["verification_exit_status"],
        int,
        f"{path}.verification_exit_status",
    )
    if availability == "available":
        if (
            profiler["exit_status"] != 0
            or profiler["verification_exit_status"] != 0
            or profiler["reason"] is not None
        ):
            raise ValidationError(f"{path} has a false available result")
        _sha256(profiler["capture_sha256"], f"{path}.capture_sha256")
        evidence = _nonempty_string(profiler["evidence"], f"{path}.evidence")
    else:
        _nonempty_string(profiler["reason"], f"{path}.reason")
        if profiler["capture_sha256"] is not None:
            raise ValidationError(f"{path} unavailable result has a capture")
        if profiler["evidence"] is not None:
            raise ValidationError(f"{path} unavailable result has evidence")

    command = profiler["command"]
    expected_tool = "nsys" if path.endswith(".nsys") else "ncu"
    if command[0] != expected_tool:
        raise ValidationError(f"{path} command does not invoke {expected_tool}")
    if expected_tool == "nsys" and "profile" not in command:
        raise ValidationError(f"{path} did not attempt a timeline capture")
    if expected_tool == "nsys":
        if (
            verification_command[:2] != ["nsys", "stats"]
            or "cuda_gpu_kern_sum" not in verification_command
        ):
            raise ValidationError(f"{path} did not verify a CUDA timeline")
        if availability == "available" and "timeline" not in evidence:
            raise ValidationError(f"{path} has no verified timeline evidence")
    if expected_tool == "ncu":
        if (
            "--export" not in command
            or "--metrics" not in command
            or "sm__cycles_elapsed.avg" not in command
            or verification_command[:2] != ["ncu", "--import"]
        ):
            raise ValidationError(f"{path} did not verify a numeric counter")
        if availability == "available" and re.fullmatch(
            r"sm__cycles_elapsed\.avg=[1-9][0-9]*(?:\.[0-9]+)?",
            evidence,
        ) is None:
            raise ValidationError(f"{path} counter evidence is not numeric")


def _validate_stage(
    value: object,
    path: str,
    *,
    gpu: str | None,
    timeout: int,
    cost: str,
) -> None:
    stage = _object(
        value,
        path,
        {
            "result",
            "gpu",
            "physical_cores",
            "memory_gib",
            "timeout_seconds",
            "max_containers",
            "concurrency",
            "single_use",
            "maximum_compute_cost_usd",
            "wall_seconds",
        },
    )
    expected = {
        "result": "pass",
        "gpu": gpu,
        "physical_cores": 2,
        "memory_gib": 4,
        "timeout_seconds": timeout,
        "max_containers": 1,
        "concurrency": 1,
        "single_use": True,
        "maximum_compute_cost_usd": cost,
    }
    expected_types: dict[str, type | tuple[type, ...]] = {
        "result": str,
        "gpu": type(None) if gpu is None else str,
        "physical_cores": int,
        "memory_gib": int,
        "timeout_seconds": int,
        "max_containers": int,
        "concurrency": int,
        "single_use": bool,
        "maximum_compute_cost_usd": str,
    }
    for field, expected_type in expected_types.items():
        if type(stage[field]) is not expected_type:
            raise ValidationError(f"{path}.{field} has the wrong JSON type")
    wall_seconds = stage["wall_seconds"]
    _value_type(wall_seconds, float, f"{path}.wall_seconds")
    if (
        not Decimal(str(wall_seconds)).is_finite()
        or wall_seconds < 0
        or wall_seconds > timeout
    ):
        raise ValidationError(f"{path}.wall_seconds is outside the timeout")
    locked_stage = {
        key: value for key, value in stage.items() if key != "wall_seconds"
    }
    if locked_stage != expected:
        raise ValidationError(f"{path} does not match locked resources")


def validate_manifest(
    manifest: object,
    lock: object,
    *,
    expected_source_commit: str,
    expected_source_bundle_sha256: str,
    expected_dependency_lock_sha256: str,
) -> None:
    """Reject any incomplete, inconsistent, or falsely passing manifest."""
    locked = _object(lock, "lock", {"schema_version", "registry_image", "toolchain", "modal"})
    _value_type(locked["schema_version"], int, "lock.schema_version")
    if locked["schema_version"] != 1:
        raise ValidationError("lock.schema_version must be 1")
    registry_lock = _object(
        locked["registry_image"],
        "lock.registry_image",
        {
            "reference",
            "index_digest",
            "linux_amd64_digest",
            "operating_system",
            "platform",
        },
    )
    toolchain_lock = _object(
        locked["toolchain"],
        "lock.toolchain",
        {
            "cuda",
            "cmake",
            "ninja_distribution",
            "ninja_binary",
            "cpp_standard",
            "cuda_standard",
            "cuda_architectures",
            "compatibility_policy",
        },
    )
    modal_lock = _object(
        locked["modal"], "lock.modal", {"sdk", "gpu", "max_containers"}
    )
    if (
        not _exact_json_equal(registry_lock, EXPECTED_REGISTRY_LOCK)
        or not _exact_json_equal(toolchain_lock, EXPECTED_TOOLCHAIN_LOCK)
        or not _exact_json_equal(modal_lock, EXPECTED_MODAL_LOCK)
    ):
        raise ValidationError("toolchain lock does not match the frozen PR-6 lock")

    root = _object(
        manifest,
        "manifest",
        {
            "schema_version",
            "result",
            "source",
            "image",
            "toolchain",
            "modal",
            "gpu",
            "stages",
            "probe",
            "compute_sanitizer",
            "profilers",
            "budget",
        },
    )
    _value_type(root["schema_version"], int, "manifest.schema_version")
    _value_type(root["result"], str, "manifest.result")
    if root["schema_version"] != 1 or root["result"] != "pass":
        raise ValidationError("manifest must be a passing schema-version-1 record")

    source = _object(
        root["source"], "manifest.source", {"commit", "bundle_sha256", "dirty"}
    )
    if source["commit"] != expected_source_commit:
        raise ValidationError("source commit mismatch")
    commit = _nonempty_string(source["commit"], "manifest.source.commit")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValidationError("source commit must be a full lowercase Git id")
    if source["dirty"] is not False:
        raise ValidationError("source must be clean")
    _sha256(source["bundle_sha256"], "manifest.source.bundle_sha256")
    if source["bundle_sha256"] != expected_source_bundle_sha256:
        raise ValidationError("source bundle hash mismatch")

    image = _object(
        root["image"],
        "manifest.image",
        {
            "reference",
            "locked_index_digest",
            "observed_index_digest",
            "locked_linux_amd64_digest",
            "observed_linux_amd64_digest",
            "operating_system",
            "platform",
        },
    )
    for key in (
        "locked_index_digest",
        "observed_index_digest",
        "locked_linux_amd64_digest",
        "observed_linux_amd64_digest",
    ):
        _digest(image[key], f"manifest.image.{key}")
    expected_image = {
        "reference": registry_lock["reference"],
        "locked_index_digest": registry_lock["index_digest"],
        "observed_index_digest": registry_lock["index_digest"],
        "locked_linux_amd64_digest": registry_lock["linux_amd64_digest"],
        "observed_linux_amd64_digest": registry_lock["linux_amd64_digest"],
        "operating_system": registry_lock["operating_system"],
        "platform": registry_lock["platform"],
    }
    if image != expected_image:
        raise ValidationError("image evidence disagrees with the lock")

    toolchain = _object(
        root["toolchain"],
        "manifest.toolchain",
        {
            "locked",
            "observed",
            "cuda_architectures",
            "compile_flags",
            "link_flags",
        },
    )
    if not _exact_json_equal(toolchain["locked"], toolchain_lock):
        raise ValidationError("embedded toolchain lock mismatch")
    observed = _object(
        toolchain["observed"],
        "manifest.toolchain.observed",
        {
            "operating_system",
            "host_compiler",
            "cmake",
            "ninja_distribution",
            "ninja_binary",
            "cuda_toolkit",
            "nvcc",
            "cuda_runtime",
            "driver",
            "driver_api",
            "cublas",
        },
    )
    if observed["operating_system"] != registry_lock["operating_system"]:
        raise ValidationError("observed operating system mismatch")
    host_compiler = _nonempty_string(
        observed["host_compiler"],
        "manifest.toolchain.observed.host_compiler",
    )
    exact_versions = {
        "cmake": toolchain_lock["cmake"],
        "ninja_distribution": toolchain_lock["ninja_distribution"],
        "ninja_binary": toolchain_lock["ninja_binary"],
        "cuda_toolkit": toolchain_lock["cuda"],
    }
    for field, expected in exact_versions.items():
        _nonempty_string(
            observed[field], f"manifest.toolchain.observed.{field}"
        )
        if observed[field] != expected:
            raise ValidationError(f"observed {field} mismatch")
    version_pattern = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}$")
    for field in ("nvcc", "cuda_runtime", "driver", "driver_api", "cublas"):
        value = _nonempty_string(
            observed[field], f"manifest.toolchain.observed.{field}"
        )
        if version_pattern.fullmatch(value) is None:
            raise ValidationError(f"observed {field} is not numeric version text")
    policy = toolchain_lock["compatibility_policy"]
    host_match = re.fullmatch(r"([a-z0-9_-]+) ([0-9]+(?:\.[0-9]+){1,2})", host_compiler)
    if (
        host_match is None
        or host_match.group(1) != policy["host_compiler"]["identity"]
        or int(host_match.group(2).split(".", 1)[0])
        != policy["host_compiler"]["major"]
    ):
        raise ValidationError("observed host compiler violates compatibility policy")
    for field, policy_name in (("nvcc", "nvcc"), ("cuda_runtime", "cuda_runtime")):
        parts = observed[field].split(".")
        if (
            int(parts[0]) != policy[policy_name]["cuda_major"]
            or int(parts[1]) != policy[policy_name]["cuda_minor"]
        ):
            raise ValidationError(f"observed {field} violates compatibility policy")
    driver_api = [int(part) for part in observed["driver_api"].split(".")[:2]]
    minimum_driver_api = [
        policy["driver"]["minimum_cuda_api_major"],
        policy["driver"]["minimum_cuda_api_minor"],
    ]
    if driver_api < minimum_driver_api:
        raise ValidationError("observed driver API violates compatibility policy")
    if int(observed["cublas"].split(".", 1)[0]) != policy["cublas"]["major"]:
        raise ValidationError("observed cuBLAS violates compatibility policy")
    architectures = toolchain["cuda_architectures"]
    if (
        type(architectures) is not list
        or len(architectures) != 1
        or type(architectures[0]) is not int
        or architectures != [89]
    ):
        raise ValidationError("CUDA architecture set must be exactly {89}")
    compile_flags = _string_list(
        toolchain["compile_flags"], "manifest.toolchain.compile_flags"
    )
    link_flags = _string_list(
        toolchain["link_flags"], "manifest.toolchain.link_flags"
    )
    joined_flags = " ".join(compile_flags + link_flags)
    architecture_flags = {
        int(value)
        for value in re.findall(r"(?:compute_|sm_|arch=)([0-9]+)", joined_flags)
    }
    if architecture_flags != {89} or "c++20" not in joined_flags:
        raise ValidationError("compile/link flags omit locked architecture or standard")
    lowered_flags = joined_flags.lower()
    forbidden_flags = (
        "fast-math",
        "use_fast_math",
        "--ftz=true",
        "-ftz=true",
        "--prec-div=false",
        "-prec-div=false",
        "--prec-sqrt=false",
        "-prec-sqrt=false",
        "-ofast",
        "finite-math-only",
        "unsafe-math-optimizations",
        "associative-math",
        "reciprocal-math",
        "/fp:fast",
        "fmad=true",
    )
    if any(flag in lowered_flags for flag in forbidden_flags):
        raise ValidationError("fast math is forbidden")
    if "cublas" in joined_flags.lower():
        raise ValidationError("PR-6 must not link cuBLAS")
    if "cudart" not in joined_flags.lower():
        raise ValidationError("CUDA runtime link evidence is missing")

    modal = _object(
        root["modal"],
        "manifest.modal",
        {
            "sdk_version",
            "dependency_lock_sha256",
            "application_id",
            "compile_call_id",
            "gpu_call_id",
        },
    )
    if modal["sdk_version"] != modal_lock["sdk"]:
        raise ValidationError("Modal SDK mismatch")
    _sha256(
        modal["dependency_lock_sha256"],
        "manifest.modal.dependency_lock_sha256",
    )
    if modal["dependency_lock_sha256"] != expected_dependency_lock_sha256:
        raise ValidationError("Modal dependency lock hash mismatch")
    for field in ("application_id", "compile_call_id", "gpu_call_id"):
        identifier = _nonempty_string(
            modal[field], f"manifest.modal.{field}"
        )
        prefix = "ap-" if field == "application_id" else "fc-"
        if not identifier.startswith(prefix) or len(identifier) <= len(prefix):
            raise ValidationError(f"manifest.modal.{field} is not a Modal id")
    if modal["compile_call_id"] == modal["gpu_call_id"]:
        raise ValidationError("compile and GPU call identifiers must be distinct")

    gpu = _object(
        root["gpu"],
        "manifest.gpu",
        {"model", "compute_capability", "total_memory_mib"},
    )
    _value_type(gpu["model"], str, "manifest.gpu.model")
    _value_type(
        gpu["compute_capability"], str, "manifest.gpu.compute_capability"
    )
    if gpu["model"] != modal_lock["gpu"] or gpu["compute_capability"] != "8.9":
        raise ValidationError("GPU identity or compute capability mismatch")
    _value_type(gpu["total_memory_mib"], int, "manifest.gpu.total_memory_mib")
    if gpu["total_memory_mib"] <= 0:
        raise ValidationError("GPU memory must be positive")

    stages = _object(
        root["stages"], "manifest.stages", {"cuda_compile", "gpu_smoke"}
    )
    _validate_stage(
        stages["cuda_compile"],
        "manifest.stages.cuda_compile",
        gpu=None,
        timeout=600,
        cost="0.021048",
    )
    _validate_stage(
        stages["gpu_smoke"],
        "manifest.stages.gpu_smoke",
        gpu="L4",
        timeout=900,
        cost="0.231372",
    )

    probe = _object(
        root["probe"],
        "manifest.probe",
        {"result", "known_answer_lengths", "sentinels", "lifecycle_repetitions"},
    )
    if (
        probe["result"] != "pass"
        or probe["known_answer_lengths"] != REQUIRED_LENGTHS
        or probe["sentinels"] != "pass"
    ):
        raise ValidationError("known-answer evidence is incomplete")
    if type(probe["known_answer_lengths"]) is not list or any(
        type(length) is not int for length in probe["known_answer_lengths"]
    ):
        raise ValidationError("known-answer lengths have invalid JSON types")
    _value_type(
        probe["lifecycle_repetitions"],
        int,
        "manifest.probe.lifecycle_repetitions",
    )
    if probe["lifecycle_repetitions"] < 100:
        raise ValidationError("lifecycle repetitions must be at least 100")

    sanitizer = _object(
        root["compute_sanitizer"],
        "manifest.compute_sanitizer",
        {"command", "result", "exit_status"},
    )
    command = _string_list(
        sanitizer["command"], "manifest.compute_sanitizer.command"
    )
    _value_type(
        sanitizer["exit_status"],
        int,
        "manifest.compute_sanitizer.exit_status",
    )
    if (
        sanitizer["result"] != "pass"
        or sanitizer["exit_status"] != 0
        or command[0] != "compute-sanitizer"
        or command[1:5] != ["--tool", "memcheck", "--leak-check", "full"]
        or "--error-exitcode=97" not in command
        or "--repetitions" not in command
        or command[-1] != "100"
        or not any(item.endswith("marketforge_cuda_probe") for item in command)
    ):
        raise ValidationError("Compute Sanitizer hard evidence is incomplete")

    profilers = _object(
        root["profilers"], "manifest.profilers", {"nsys", "ncu"}
    )
    _validate_profiler(profilers["nsys"], "manifest.profilers.nsys")
    _validate_profiler(profilers["ncu"], "manifest.profilers.ncu")

    budget = _object(
        root["budget"],
        "manifest.budget",
        {
            "month_to_date_usd",
            "maximum_planned_cost_usd",
            "estimated_actual_compute_cost_usd",
            "gpu_minutes",
            "monthly_budget_usd",
            "project_soft_cap_usd",
            "reserve_usd",
            "billing_report_caveat",
        },
    )
    month_to_date = _decimal(
        budget["month_to_date_usd"], "manifest.budget.month_to_date_usd"
    )
    maximum = _decimal(
        budget["maximum_planned_cost_usd"],
        "manifest.budget.maximum_planned_cost_usd",
    )
    actual = _decimal(
        budget["estimated_actual_compute_cost_usd"],
        "manifest.budget.estimated_actual_compute_cost_usd",
    )
    gpu_minutes = _decimal(
        budget["gpu_minutes"], "manifest.budget.gpu_minutes"
    )
    if maximum != CHAIN_COST_CEILING_USD:
        raise ValidationError("maximum planned cost mismatch")
    if month_to_date + maximum > Decimal("24"):
        raise ValidationError("combined stages cross the project soft cap")
    if (
        actual > TRIAL_COMPUTE_CEILING_USD
        or gpu_minutes > TRIAL_GPU_MINUTE_CEILING
    ):
        raise ValidationError("trial compute ceiling exceeded")
    compile_seconds = Decimal(
        str(stages["cuda_compile"]["wall_seconds"])
    )
    gpu_seconds = Decimal(str(stages["gpu_smoke"]["wall_seconds"]))
    expected_actual = compile_seconds * (
        Decimal("2") * CPU_CORE_SECOND_USD
        + Decimal("4") * MEMORY_GIB_SECOND_USD
    ) + gpu_seconds * (
        Decimal("2") * CPU_CORE_SECOND_USD
        + Decimal("4") * MEMORY_GIB_SECOND_USD
        + GPU_SECOND_USD["L4"]
    )
    if actual != expected_actual:
        raise ValidationError("estimated actual compute cost is inconsistent")
    if gpu_minutes != gpu_seconds / Decimal("60"):
        raise ValidationError("GPU minutes are inconsistent with stage duration")
    if (
        budget["monthly_budget_usd"] != "30"
        or budget["project_soft_cap_usd"] != "24"
        or budget["reserve_usd"] != "6"
    ):
        raise ValidationError("budget constants mismatch")
    caveat = _nonempty_string(
        budget["billing_report_caveat"],
        "manifest.budget.billing_report_caveat",
    )
    lowered_caveat = caveat.lower()
    if (
        len(caveat) < 80
        or "modal billing" not in lowered_caveat
        or "authoritative" not in lowered_caveat
        or "image" not in lowered_caveat
        or "storage" not in lowered_caveat
    ):
        raise ValidationError("billing caveat is not substantive")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).with_name("cuda-toolchain-lock.json"),
    )
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument("--expected-dependency-lock-sha256", required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    manifest = strict_json_loads(args.manifest.read_text(encoding="utf-8"))
    lock = strict_json_loads(args.lock.read_text(encoding="utf-8"))
    try:
        validate_manifest(
            manifest,
            lock,
            expected_source_commit=args.expected_source_commit,
            expected_source_bundle_sha256=args.expected_source_bundle_sha256,
            expected_dependency_lock_sha256=args.expected_dependency_lock_sha256,
        )
    except ValidationError as error:
        raise SystemExit(f"invalid CUDA evidence: {error}") from error
    print(json.dumps({"schema_version": 1, "result": "pass"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
