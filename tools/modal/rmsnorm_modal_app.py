"""Small, bounded L4 gate for the fast-track CUDA RMSNorm vertical slice."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time

import modal

from tools.modal.cuda_ci import SourceBundle, create_source_bundle, parse_cost
from tools.modal.modal_budget import (
    CPU_CORE_SECOND_USD,
    GPU_SECOND_USD,
    MEMORY_GIB_SECOND_USD,
    require_project_headroom,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCK = json.loads(
    (PROJECT_ROOT / "tools/modal/cuda-toolchain-lock.json").read_text(
        encoding="utf-8"
    )
)
REMOTE_SOURCE = Path("/tmp/marketforge-rmsnorm-source")
REMOTE_BUILD = Path("/tmp/marketforge-rmsnorm-build")
TIMEOUT_SECONDS = 600
PHYSICAL_CORES = Decimal("2")
MEMORY_GIB = Decimal("4")
MAXIMUM_COST_USD = Decimal(TIMEOUT_SECONDS) * (
    PHYSICAL_CORES * CPU_CORE_SECOND_USD
    + MEMORY_GIB * MEMORY_GIB_SECOND_USD
    + GPU_SECOND_USD["L4"]
)

if MAXIMUM_COST_USD != Decimal("0.1542480"):
    raise RuntimeError("RMSNorm L4 cost ceiling changed")


cuda_image = (
    modal.Image.from_registry(
        LOCK["registry_image"]["reference"],
        add_python="3.12",
    )
    .pip_install("cmake==3.30.5", "ninja==1.11.1.1")
)

app = modal.App(
    "marketforge-fast-cuda-rmsnorm",
    tags={
        "project": "marketforge",
        "purpose": "cuda-rmsnorm-fast-track",
    },
)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            **(environment or {}),
        },
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result = {
        "command": command,
        "exit_status": completed.returncode,
        "output": completed.stdout,
        "wall_seconds": round(time.monotonic() - started, 6),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: "
            f"{' '.join(command)}\n{completed.stdout}"
        )
    return result


def _extract_bundle(content: bytes, expected_sha256: str) -> Path:
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise RuntimeError("source bundle SHA-256 mismatch")
    shutil.rmtree(REMOTE_SOURCE, ignore_errors=True)
    REMOTE_SOURCE.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (member.isdir() or member.isfile())
            ):
                raise RuntimeError(
                    f"unsafe source archive member: {member.name}"
                )
        archive.extractall(REMOTE_SOURCE, filter="data")
    embedded_lock = json.loads(
        (
            REMOTE_SOURCE / "tools/modal/cuda-toolchain-lock.json"
        ).read_text(encoding="utf-8")
    )
    if embedded_lock != LOCK:
        raise RuntimeError("source-bundle CUDA lock mismatch")
    return REMOTE_SOURCE


def _strict_json_line(output: str) -> dict[str, object]:
    lines = [line for line in output.splitlines() if line.startswith("{")]
    if len(lines) != 1:
        raise RuntimeError("benchmark emitted no unique JSON line")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key: {key}")
            value[key] = item
        return value

    value = json.loads(
        lines[0],
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda name: (_ for _ in ()).throw(
            ValueError(f"non-finite constant: {name}")
        ),
    )
    if type(value) is not dict:
        raise RuntimeError("benchmark JSON is not an object")
    return value


@app.function(
    image=cuda_image,
    gpu="L4",
    cpu=2.0,
    memory=4096,
    timeout=TIMEOUT_SECONDS,
    max_containers=1,
    single_use_containers=True,
)
@modal.concurrent(max_inputs=1)
def rmsnorm_l4_gate(
    source_content: bytes,
    source_sha256: str,
    candidate_commit: str,
) -> dict[str, object]:
    started = time.monotonic()
    if re.fullmatch(r"[0-9a-f]{40}", candidate_commit) is None:
        raise RuntimeError("candidate commit identity is malformed")
    source = _extract_bundle(source_content, source_sha256)
    shutil.rmtree(REMOTE_BUILD, ignore_errors=True)
    environment = {
        "CUDACXX": "/usr/local/cuda/bin/nvcc",
        "CTEST_OUTPUT_ON_FAILURE": "1",
    }
    commands = [
        _run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(REMOTE_BUILD),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DMARKETFORGE_ENABLE_CUDA=ON",
                "-DMARKETFORGE_WARNINGS_AS_ERRORS=ON",
                "-DCMAKE_CUDA_ARCHITECTURES=89",
            ],
            environment=environment,
        ),
        _run(
            ["cmake", "--build", str(REMOTE_BUILD), "--parallel", "2"],
            environment=environment,
        ),
        _run(
            [
                "ctest",
                "--test-dir",
                str(REMOTE_BUILD),
                "--output-on-failure",
            ],
            environment=environment,
        ),
    ]
    benchmark_run = _run(
        [str(REMOTE_BUILD / "marketforge_cuda_rmsnorm_bench")],
        environment=environment,
    )
    benchmark = _strict_json_line(str(benchmark_run["output"]))
    measurements = benchmark.get("measurements")
    if (
        set(benchmark)
        != {
            "schema_version",
            "result",
            "operator",
            "model_shape",
            "gpu",
            "measurements",
        }
        or type(benchmark["schema_version"]) is not int
        or benchmark["schema_version"] != 1
        or benchmark["result"] != "pass"
        or benchmark["operator"] != "rms_norm_f32"
        or benchmark["model_shape"]
        != "SmolLM2-135M hidden_size=576"
        or type(benchmark["gpu"]) is not dict
        or "L4" not in str(benchmark["gpu"].get("name"))
        or benchmark["gpu"].get("compute_capability") != "8.9"
        or type(measurements) is not list
        or len(measurements) != 4
    ):
        raise RuntimeError("RMSNorm benchmark evidence mismatch")
    expected_cases = {
        (1, 2_000),
        (16, 1_000),
        (256, 500),
        (1_024, 200),
    }
    observed_cases: set[tuple[int, int]] = set()
    for measurement in measurements:
        if (
            type(measurement) is not dict
            or set(measurement)
            != {
                "rows",
                "hidden_size",
                "iterations",
                "average_microseconds",
                "logical_gib_per_second",
            }
            or type(measurement["rows"]) is not int
            or type(measurement["hidden_size"]) is not int
            or measurement["hidden_size"] != 576
            or type(measurement["iterations"]) is not int
            or type(measurement["average_microseconds"]) is not float
            or not math.isfinite(measurement["average_microseconds"])
            or measurement["average_microseconds"] <= 0.0
            or type(measurement["logical_gib_per_second"]) is not float
            or not math.isfinite(
                measurement["logical_gib_per_second"]
            )
            or measurement["logical_gib_per_second"] <= 0.0
        ):
            raise RuntimeError("RMSNorm measurement evidence mismatch")
        observed_cases.add(
            (measurement["rows"], measurement["iterations"])
        )
    if observed_cases != expected_cases:
        raise RuntimeError("RMSNorm benchmark case inventory mismatch")
    return {
        "schema_version": 1,
        "result": "pass",
        "candidate_commit": candidate_commit,
        "source_bundle_sha256": source_sha256,
        "application_id": app.app_id,
        "function_call_id": modal.current_function_call_id(),
        "gpu": "L4",
        "cuda_architecture": 89,
        "commands": commands,
        "benchmark": benchmark,
        "benchmark_output_sha256": hashlib.sha256(
            str(benchmark_run["output"]).encode("utf-8")
        ).hexdigest(),
        "wall_seconds": round(time.monotonic() - started, 6),
    }


def _validate_result(
    result: object,
    bundle: SourceBundle,
) -> dict[str, object]:
    if type(result) is not dict or set(result) != {
        "schema_version",
        "result",
        "candidate_commit",
        "source_bundle_sha256",
        "application_id",
        "function_call_id",
        "gpu",
        "cuda_architecture",
        "commands",
        "benchmark",
        "benchmark_output_sha256",
        "wall_seconds",
    }:
        raise RuntimeError("remote RMSNorm evidence schema mismatch")
    if (
        type(result["schema_version"]) is not int
        or result["schema_version"] != 1
        or result["result"] != "pass"
        or result["candidate_commit"] != bundle.commit
        or result["source_bundle_sha256"] != bundle.sha256
        or result["gpu"] != "L4"
        or type(result["cuda_architecture"]) is not int
        or result["cuda_architecture"] != 89
        or type(result["application_id"]) is not str
        or not result["application_id"].startswith("ap-")
        or type(result["function_call_id"]) is not str
        or not result["function_call_id"].startswith("fc-")
        or type(result["wall_seconds"]) is not float
        or not 0.0 <= result["wall_seconds"] <= TIMEOUT_SECONDS
        or type(result["benchmark_output_sha256"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}", result["benchmark_output_sha256"]
        )
        is None
    ):
        raise RuntimeError("remote RMSNorm evidence provenance mismatch")
    return result


@app.local_entrypoint()
def main(month_to_date_usd: str) -> None:
    month_to_date = parse_cost(month_to_date_usd)
    require_project_headroom(
        month_to_date_usd=month_to_date,
        planned_cost_usd=MAXIMUM_COST_USD,
    )
    bundle = create_source_bundle(PROJECT_ROOT)
    result = rmsnorm_l4_gate.remote(
        bundle.content,
        bundle.sha256,
        bundle.commit,
    )
    accepted = _validate_result(result, bundle)
    accepted["budget"] = {
        "month_to_date_usd": str(month_to_date),
        "maximum_compute_cost_usd": f"{MAXIMUM_COST_USD:.6f}",
        "project_soft_cap_usd": "24",
    }
    print(json.dumps(accepted, indent=2, sort_keys=True, allow_nan=False))
