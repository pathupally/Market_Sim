"""Run portable C++ acceptance checks on a bounded Modal CPU container."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import time

import modal

from tools.modal.modal_budget import (
    PROJECT_SOFT_CAP_USD,
    RESERVE_USD,
    estimate_compute_cost,
    require_project_headroom,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTE_SOURCE = Path("/workspace/marketforge")
FUNCTION_TIMEOUT_SECONDS = 600
PHYSICAL_CORES = 2.0
MEMORY_MIB = 2048

source_ignore = modal.FilePatternMatcher.from_file(PROJECT_ROOT / ".modalignore")
linux_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("clang", "cmake", "g++", "ninja-build")
    .add_local_dir(
        PROJECT_ROOT,
        remote_path=str(REMOTE_SOURCE),
        ignore=source_ignore,
    )
)

app = modal.App(
    "marketforge-cpu-ci",
    tags={"project": "marketforge", "purpose": "ci"},
)


def _run(command: list[str]) -> dict[str, object]:
    print("+", " ".join(command), flush=True)
    started = time.monotonic()
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "CTEST_OUTPUT_ON_FAILURE": "1"},
    )
    elapsed = time.monotonic() - started
    print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}"
        )
    return {"command": command, "seconds": round(elapsed, 3)}


def _first_line(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.splitlines()[0]


def _build(
    *,
    label: str,
    compiler: str,
    sanitizers: bool,
) -> list[dict[str, object]]:
    build_dir = Path("/tmp") / f"marketforge-{label}"
    configure = [
        "cmake",
        "-S",
        str(REMOTE_SOURCE),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Debug",
        f"-DCMAKE_CXX_COMPILER={compiler}",
        "-DMARKETFORGE_WARNINGS_AS_ERRORS=ON",
        (
            "-DMARKETFORGE_ENABLE_SANITIZERS=ON"
            if sanitizers
            else "-DMARKETFORGE_ENABLE_SANITIZERS=OFF"
        ),
    ]
    return [
        _run(configure),
        _run(["cmake", "--build", str(build_dir), "--parallel", "2"]),
        _run(
            [
                "ctest",
                "--test-dir",
                str(build_dir),
                "--output-on-failure",
            ]
        ),
    ]


@app.function(
    image=linux_image,
    cpu=PHYSICAL_CORES,
    memory=MEMORY_MIB,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    max_containers=1,
    single_use_containers=True,
)
def run_linux_ci() -> dict[str, object]:
    started = time.monotonic()
    environment = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cmake": _first_line(["cmake", "--version"]),
        "ninja": _first_line(["ninja", "--version"]),
        "gcc": _first_line(["g++", "--version"]),
        "clang": _first_line(["clang++", "--version"]),
    }
    checks = {
        "gcc-debug": _build(
            label="gcc-debug",
            compiler="g++",
            sanitizers=False,
        ),
        "clang-sanitize": _build(
            label="clang-sanitize",
            compiler="clang++",
            sanitizers=True,
        ),
    }
    return {
        "status": "pass",
        "environment": environment,
        "checks": checks,
        "wall_seconds": round(time.monotonic() - started, 3),
    }


@app.local_entrypoint()
def main(month_to_date_usd: str = "0") -> None:
    from decimal import Decimal

    maximum_compute_cost = estimate_compute_cost(
        seconds=FUNCTION_TIMEOUT_SECONDS,
        physical_cores=Decimal(str(PHYSICAL_CORES)),
        memory_gib=Decimal(MEMORY_MIB) / Decimal(1024),
    )
    require_project_headroom(
        month_to_date_usd=Decimal(month_to_date_usd),
        planned_cost_usd=maximum_compute_cost,
    )
    print(
        "Budget preflight: "
        f"maximum function compute ${maximum_compute_cost:.4f}; "
        f"soft cap ${PROJECT_SOFT_CAP_USD}; reserve ${RESERVE_USD}",
        flush=True,
    )
    result = run_linux_ci.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
