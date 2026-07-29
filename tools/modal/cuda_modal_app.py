"""Pinned, bounded Modal application for the PR-6 CUDA acceptance gate.

The local entrypoint is the only dispatch authority. It performs the combined
two-stage budget preflight, verifies local gates, creates one immutable Git
archive, and dispatches the compile and L4 stages in order. Importing this
module or using ``--dry-run`` never contacts Modal.
"""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
import hashlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import tarfile
import time
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import modal

from tools.modal.cuda_ci import (
    CHAIN_COST_CEILING_USD,
    COMPILE_TIMEOUT_SECONDS,
    AuthorizationTicketStore,
    EvidenceCache,
    GATE_ID,
    GPU_TIMEOUT_SECONDS,
    STAGES,
    TrialLedger,
    create_source_bundle,
    dry_run_manifest,
    parse_cost,
    source_bundle_member_sha256,
    strict_json_loads,
)
from tools.modal.cuda_evidence import validate_manifest
from tools.modal.modal_budget import (
    CPU_CORE_SECOND_USD,
    GPU_SECOND_USD,
    MEMORY_GIB_SECOND_USD,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = PROJECT_ROOT / "tools/modal/cuda-toolchain-lock.json"
LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
REGISTRY_REFERENCE = LOCK["registry_image"]["reference"]
REMOTE_SOURCE = Path("/tmp/marketforge-source")
COMPUTE_SANITIZER_ERROR_EXIT = 97

cuda_image = (
    modal.Image.from_registry(REGISTRY_REFERENCE, add_python="3.12")
    .pip_install("cmake==3.30.5", "ninja==1.11.1.1")
)

app = modal.App(
    "marketforge-pr6-cuda-lifecycle",
    tags={"project": "marketforge", "purpose": "pr6-cuda-acceptance"},
)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    check: bool = True,
) -> dict[str, object]:
    started = time.monotonic()
    effective_environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        **(environment or {}),
    }
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=effective_environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result: dict[str, object] = {
        "command": command,
        "exit_status": completed.returncode,
        "output": completed.stdout,
        "wall_seconds": round(time.monotonic() - started, 6),
    }
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n{completed.stdout}"
        )
    return result


def _safe_extract_bundle(content: bytes, expected_sha256: str) -> Path:
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise RuntimeError("source bundle SHA-256 mismatch before extraction")
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
                raise RuntimeError(f"unsafe source archive member: {member.name}")
        archive.extractall(REMOTE_SOURCE, filter="data")
    embedded_lock = json.loads(
        (REMOTE_SOURCE / "tools/modal/cuda-toolchain-lock.json").read_text(
            encoding="utf-8"
        )
    )
    if embedded_lock != LOCK:
        raise RuntimeError("source bundle toolchain lock differs from dispatcher")
    return REMOTE_SOURCE


def _first_line(command: list[str]) -> str:
    result = _run(command)
    output = str(result["output"]).strip()
    if not output:
        raise RuntimeError(f"version command returned no output: {command}")
    return output.splitlines()[0]


def _installed_distribution_version(distribution: str) -> str:
    try:
        observed = distribution_version(distribution)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"installed Python distribution is missing: {distribution}"
        ) from error
    if type(observed) is not str or not observed:
        raise RuntimeError(
            f"installed Python distribution version is missing: {distribution}"
        )
    return observed


def _version(pattern: str, text: str, name: str) -> str:
    match = re.search(pattern, text)
    if match is None:
        raise RuntimeError(f"could not parse observed {name}: {text}")
    return match.group(1)


def _observed_platform() -> dict[str, str]:
    os_release: dict[str, str] = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip('"')
    operating_system = (
        f"ubuntu{os_release.get('VERSION_ID')}"
        if os_release.get("ID") == "ubuntu"
        else f"{os_release.get('ID', 'unknown')}{os_release.get('VERSION_ID', '')}"
    )
    machine = platform.machine()
    canonical_machine = "amd64" if machine in {"x86_64", "amd64"} else machine
    return {
        "operating_system": operating_system,
        "platform": f"linux/{canonical_machine}",
    }


def _observe_registry_image() -> dict[str, str]:
    """Resolve the digest-qualified Docker Hub index and AMD64 manifest."""

    repository = "nvidia/cuda"
    index_digest = REGISTRY_REFERENCE.rsplit("@", 1)[1]
    token_query = urlencode(
        {
            "service": "registry.docker.io",
            "scope": f"repository:{repository}:pull",
        }
    )
    with urlopen(
        f"https://auth.docker.io/token?{token_query}", timeout=20
    ) as response:
        token_document = json.loads(response.read())
    token = token_document.get("token")
    if type(token) is not str or not token:
        raise RuntimeError("Docker Hub returned no registry bearer token")
    request = Request(
        (
            "https://registry-1.docker.io/v2/"
            f"{repository}/manifests/{index_digest}"
        ),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": (
                "application/vnd.oci.image.index.v1+json,"
                "application/vnd.docker.distribution.manifest.list.v2+json"
            ),
        },
    )
    with urlopen(request, timeout=20) as response:
        observed_index = response.headers.get("Docker-Content-Digest")
        index = json.loads(response.read())
    if type(observed_index) is not str:
        raise RuntimeError("registry omitted Docker-Content-Digest")
    manifests = index.get("manifests")
    if type(manifests) is not list:
        raise RuntimeError("registry response is not an image index")
    matches = [
        manifest
        for manifest in manifests
        if type(manifest) is dict
        and type(manifest.get("platform")) is dict
        and manifest["platform"].get("architecture") == "amd64"
        and manifest["platform"].get("os") == "linux"
    ]
    if len(matches) != 1 or type(matches[0].get("digest")) is not str:
        raise RuntimeError("registry index has no unique Linux/AMD64 manifest")
    return {
        "index_digest": observed_index,
        "linux_amd64_digest": matches[0]["digest"],
    }


def _read_cuda_toolkit_version(
    nvcc_version: str,
    *,
    environment: Any = None,
    version_file: Path = Path("/usr/local/cuda/version.json"),
) -> str:
    """Read the image-declared toolkit version and verify independent evidence."""

    effective_environment = os.environ if environment is None else environment
    declared_version = effective_environment.get("CUDA_VERSION")
    exact_pattern = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    nvcc_pattern = (
        r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
        r"(?:\.(?:0|[1-9][0-9]*))?"
    )
    if (
        type(declared_version) is not str
        or re.fullmatch(exact_pattern, declared_version) is None
    ):
        raise RuntimeError(
            "pinned image CUDA_VERSION is missing or malformed"
        )
    if declared_version != LOCK["toolchain"]["cuda"]:
        raise RuntimeError(
            "pinned image CUDA_VERSION disagrees with the toolkit lock"
        )
    if (
        type(nvcc_version) is not str
        or re.fullmatch(nvcc_pattern, nvcc_version) is None
    ):
        raise RuntimeError("parsed nvcc version is malformed")
    if declared_version.split(".")[:2] != nvcc_version.split(".")[:2]:
        raise RuntimeError("CUDA_VERSION and nvcc CUDA versions disagree")

    if version_file.exists():
        try:
            value = strict_json_loads(
                version_file.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise RuntimeError("CUDA version.json is malformed") from error
        if type(value) is not dict:
            raise RuntimeError("CUDA version.json has an unexpected schema")
        cuda = value.get("cuda")
        if type(cuda) is not dict:
            raise RuntimeError("CUDA version.json has an unexpected schema")
        json_version = cuda.get("version")
        if (
            type(json_version) is not str
            or re.fullmatch(exact_pattern, json_version) is None
        ):
            raise RuntimeError("CUDA version.json has an unexpected schema")
        if json_version != declared_version:
            raise RuntimeError(
                "CUDA version.json disagrees with CUDA_VERSION"
            )
    return declared_version


def _compile_flags(build: Path) -> tuple[list[str], list[str]]:
    database = json.loads(
        (build / "compile_commands.json").read_text(encoding="utf-8")
    )
    compile_commands: list[str] = []
    for entry in database:
        file_name = str(entry.get("file", ""))
        if file_name.endswith((".cu", "cuda_probe.cpp")):
            command = entry.get("command")
            if type(command) is not str:
                arguments = entry.get("arguments")
                if type(arguments) is not list:
                    raise RuntimeError("compile command has no command or arguments")
                command = " ".join(str(argument) for argument in arguments)
            compile_commands.append(command)
    if not compile_commands:
        raise RuntimeError("CUDA compile commands are absent")
    links = _run(
        ["ninja", "-C", str(build), "-t", "commands", "marketforge_cuda_probe"]
    )
    link_commands = [
        line
        for line in str(links["output"]).splitlines()
        if line.strip() and (" -o marketforge_cuda_probe" in line or line.endswith("marketforge_cuda_probe"))
    ]
    if not link_commands:
        raise RuntimeError("CUDA probe link command is absent")
    return compile_commands, link_commands


def _configure_cuda(
    source: Path, build: Path, *, run_gpu_tests: bool
) -> dict[str, object]:
    environment = {
        **os.environ,
        "CUDACXX": "/usr/local/cuda/bin/nvcc",
        "CTEST_OUTPUT_ON_FAILURE": "1",
    }
    commands = [
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DMARKETFORGE_ENABLE_CUDA=ON",
            "-DMARKETFORGE_WARNINGS_AS_ERRORS=ON",
            "-DCMAKE_CUDA_ARCHITECTURES=89",
        ],
        ["cmake", "--build", str(build), "--parallel", "2"],
    ]
    if run_gpu_tests:
        commands.append(
            ["ctest", "--test-dir", str(build), "--output-on-failure"]
        )
    else:
        commands.append(
            [
                "ctest",
                "--test-dir",
                str(build),
                "--output-on-failure",
                "--tests-regex",
                "^marketforge.cuda_public_headers$",
            ]
        )
    results = [
        _run(command, environment=environment)
        for command in commands
    ]
    compile_flags, link_flags = _compile_flags(build)
    return {
        "commands": results,
        "compile_flags": compile_flags,
        "link_flags": link_flags,
    }


def _configure_cuda_off(source: Path, build: Path) -> list[dict[str, object]]:
    environment = {
        **os.environ,
        "CUDACXX": "/definitely/not/a/cuda/compiler",
        "CTEST_OUTPUT_ON_FAILURE": "1",
    }
    commands = [
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DMARKETFORGE_ENABLE_CUDA=OFF",
            "-DMARKETFORGE_WARNINGS_AS_ERRORS=ON",
        ],
        ["cmake", "--build", str(build), "--parallel", "2"],
        ["ctest", "--test-dir", str(build), "--output-on-failure"],
    ]
    return [_run(command, environment=environment) for command in commands]


def _device_probe(build: Path) -> dict[str, object]:
    source = build / "pr6_device_evidence.cu"
    executable = build / "pr6_device_evidence"
    source.write_text(
        """
#include <cuda_runtime_api.h>
#include <cstdio>
int main() {
  cudaDeviceProp p{};
  int count = 0;
  int runtime = 0;
  int driver = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess ||
      cudaRuntimeGetVersion(&runtime) != cudaSuccess ||
      cudaDriverGetVersion(&driver) != cudaSuccess ||
      cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
    return 1;
  }
  std::printf("{\\\"name\\\":\\\"%s\\\",\\\"major\\\":%d,\\\"minor\\\":%d,"
              "\\\"memory_bytes\\\":%llu,\\\"runtime_integer\\\":%d,"
              "\\\"driver_api_integer\\\":%d,\\\"visible_device_count\\\":%d}\\n",
              p.name, p.major, p.minor,
              static_cast<unsigned long long>(p.totalGlobalMem),
              runtime, driver, count);
  return 0;
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _run(
        [
            "/usr/local/cuda/bin/nvcc",
            "-std=c++20",
            "-arch=sm_89",
            str(source),
            "-o",
            str(executable),
        ]
    )
    result = _run([str(executable)])
    try:
        value = json.loads(str(result["output"]))
    except json.JSONDecodeError as error:
        raise RuntimeError("device evidence probe did not emit JSON") from error
    if type(value) is not dict:
        raise RuntimeError("device evidence probe JSON must be an object")
    return value


def _integer_cuda_version(value: object) -> str:
    if type(value) is not int or value <= 0:
        raise RuntimeError("CUDA integer version must be positive")
    return f"{value // 1000}.{(value % 1000) // 10}"


def _cublas_header_version() -> str:
    text = Path("/usr/local/cuda/include/cublas_api.h").read_text(
        encoding="utf-8", errors="replace"
    )
    values: list[str] = []
    for component in ("MAJOR", "MINOR", "PATCH", "BUILD"):
        values.append(
            _version(
                rf"#define\s+CUBLAS_VER_{component}\s+(\d+)",
                text,
                f"cuBLAS {component}",
            )
        )
    return ".".join(values)


def _parse_probe(output: str, repetitions: int) -> dict[str, object]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("CUDA lifecycle probe returned no output")
    expected = {
        "schema_version": 1,
        "result": "pass",
        "known_answer_lengths": [0, 1, 255, 256, 257, 1025],
        "sentinels": "pass",
        "lifecycle_repetitions": repetitions,
    }
    value: object | None = None
    for line in lines:
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate == expected:
            value = candidate
            break
    if value != expected:
        raise RuntimeError("CUDA lifecycle probe emitted no matching JSON evidence")
    return {
        "result": "pass",
        "known_answer_lengths": value["known_answer_lengths"],
        "sentinels": "pass",
        "lifecycle_repetitions": repetitions,
    }


def _tool_version(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    result = _run(command, check=False)
    output = str(result["output"]).strip()
    return output.splitlines()[0] if output else None


def _profiler_capability(
    *,
    name: str,
    command: list[str],
    capture_candidates: list[Path],
    version_command: list[str],
    verification_command: list[str],
    verify_output: Callable[[str], str | None],
) -> dict[str, object]:
    for path in capture_candidates:
        path.unlink(missing_ok=True)
    version = _tool_version(version_command)
    if shutil.which(name) is None:
        return {
            "availability": "unavailable",
            "command": command,
            "version": version,
            "exit_status": 127,
            "reason": f"{name} executable is not installed in the pinned image",
            "capture_sha256": None,
            "verification_command": verification_command,
            "verification_exit_status": 127,
            "evidence": None,
        }
    result = _run(command, check=False)
    capture = next(
        (
            path
            for path in capture_candidates
            if path.is_file() and path.stat().st_size > 0
        ),
        None,
    )
    verification = _run(verification_command, check=False)
    evidence = verify_output(str(verification["output"]))
    if (
        result["exit_status"] == 0
        and capture is not None
        and verification["exit_status"] == 0
        and evidence is not None
    ):
        digest = hashlib.sha256(capture.read_bytes()).hexdigest()
        for path in capture_candidates:
            path.unlink(missing_ok=True)
        return {
            "availability": "available",
            "command": command,
            "version": version,
            "exit_status": 0,
            "reason": None,
            "capture_sha256": digest,
            "verification_command": verification_command,
            "verification_exit_status": 0,
            "evidence": evidence,
        }
    for path in capture_candidates:
        path.unlink(missing_ok=True)
    reason = "\n".join(
        part
        for part in (
            str(result["output"]).strip(),
            str(verification["output"]).strip(),
        )
        if part
    )
    if not reason:
        reason = (
            f"{name} returned {result['exit_status']} without a nonempty capture"
        )
    return {
        "availability": "unavailable",
        "command": command,
        "version": version,
        "exit_status": int(result["exit_status"]),
        "reason": reason[-4000:],
        "capture_sha256": None,
        "verification_command": verification_command,
        "verification_exit_status": int(verification["exit_status"]),
        "evidence": None,
    }


def _verify_nsys_output(output: str) -> str | None:
    if (
        "lifecycle_kernel" not in output
        or re.search(r"\b[1-9][0-9]*(?:\.[0-9]+)?\b", output) is None
    ):
        return None
    return "CUDA lifecycle_kernel timeline row with numeric duration verified"


def _verify_ncu_output(output: str) -> str | None:
    metric = "sm__cycles_elapsed.avg"
    if metric not in output:
        return None
    match = re.search(
        rf"{re.escape(metric)}[^\r\n]*?([1-9][0-9]*(?:\.[0-9]+)?)",
        output,
    )
    if match is None:
        return None
    return f"{metric}={match.group(1)}"


@app.function(
    image=cuda_image,
    cpu=2.0,
    memory=4096,
    timeout=COMPILE_TIMEOUT_SECONDS,
    max_containers=1,
    single_use_containers=True,
)
@modal.concurrent(max_inputs=1)
def cuda_compile(
    source_bundle: bytes, source_sha256: str, candidate_commit: str
) -> dict[str, object]:
    started = time.monotonic()
    source = _safe_extract_bundle(source_bundle, source_sha256)
    off_build = Path("/tmp/marketforge-build-off")
    cuda_build = Path("/tmp/marketforge-build-cuda")
    off_checks = _configure_cuda_off(source, off_build)
    cuda_checks = _configure_cuda(
        source, cuda_build, run_gpu_tests=False
    )
    platform_evidence = _observed_platform()
    cmake_line = _first_line(["cmake", "--version"])
    ninja_distribution = _installed_distribution_version("ninja")
    ninja_binary = _first_line(["ninja", "--version"])
    nvcc_text = str(_run(["/usr/local/cuda/bin/nvcc", "--version"])["output"])
    nvcc_version = _version(r"V([0-9.]+)", nvcc_text, "nvcc")
    result = {
        "result": "pass",
        "candidate_commit": candidate_commit,
        "source_bundle_sha256": source_sha256,
        "remote_call_id": modal.current_function_call_id(),
        "image": _observe_registry_image(),
        "observed": {
            **platform_evidence,
            "host_compiler": (
                "gcc "
                + _first_line(
                    ["c++", "-dumpfullversion", "-dumpversion"]
                )
            ),
            "cmake": _version(r"cmake version ([0-9.]+)", cmake_line, "CMake"),
            "ninja_distribution": ninja_distribution,
            "ninja_binary": ninja_binary,
            "cuda_toolkit": _read_cuda_toolkit_version(nvcc_version),
            "nvcc": nvcc_version,
        },
        "compile_flags": cuda_checks["compile_flags"],
        "link_flags": cuda_checks["link_flags"],
        "cuda_off_checks": off_checks,
        "cuda_checks": cuda_checks["commands"],
        "wall_seconds": round(time.monotonic() - started, 6),
    }
    return result


@app.function(
    image=cuda_image,
    gpu="L4",
    cpu=2.0,
    memory=4096,
    timeout=GPU_TIMEOUT_SECONDS,
    max_containers=1,
    single_use_containers=True,
)
@modal.concurrent(max_inputs=1)
def gpu_smoke(
    source_bundle: bytes, source_sha256: str, candidate_commit: str
) -> dict[str, object]:
    started = time.monotonic()
    source = _safe_extract_bundle(source_bundle, source_sha256)
    build = Path("/tmp/marketforge-build-gpu")
    cuda_checks = _configure_cuda(source, build, run_gpu_tests=True)
    probe_executable = build / "marketforge_cuda_probe"

    probe_run = _run([str(probe_executable), "--repetitions", "1"])
    probe = _parse_probe(str(probe_run["output"]), 1)

    sanitizer_command = [
        "compute-sanitizer",
        "--tool",
        "memcheck",
        "--leak-check",
        "full",
        f"--error-exitcode={COMPUTE_SANITIZER_ERROR_EXIT}",
        str(probe_executable),
        "--repetitions",
        "100",
    ]
    sanitizer_run = _run(sanitizer_command)
    sanitizer_probe = _parse_probe(str(sanitizer_run["output"]), 100)
    if "ERROR SUMMARY: 0 errors" not in str(sanitizer_run["output"]):
        raise RuntimeError("Compute Sanitizer did not report a zero-error summary")

    device = _device_probe(build)
    nvidia_smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    driver_lines = str(nvidia_smi["output"]).strip().splitlines()
    if len(driver_lines) != 1:
        raise RuntimeError("GPU stage did not expose exactly one driver row")
    driver_version = driver_lines[0]

    nsys_base = Path("/tmp/marketforge-pr6-nsys")
    ncu_base = Path("/tmp/marketforge-pr6-ncu")
    nsys_report = nsys_base.with_suffix(".nsys-rep")
    ncu_report = ncu_base.with_suffix(".ncu-rep")
    profilers = {
        "nsys": _profiler_capability(
            name="nsys",
            command=[
                "nsys",
                "profile",
                "--force-overwrite",
                "true",
                "--output",
                str(nsys_base),
                str(probe_executable),
                "--repetitions",
                "1",
            ],
            capture_candidates=[
                nsys_report,
                nsys_base,
                nsys_base.with_suffix(".sqlite"),
            ],
            version_command=["nsys", "--version"],
            verification_command=[
                "nsys",
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--format",
                "csv",
                str(nsys_report),
            ],
            verify_output=_verify_nsys_output,
        ),
        "ncu": _profiler_capability(
            name="ncu",
            command=[
                "ncu",
                "--target-processes",
                "all",
                "--metrics",
                "sm__cycles_elapsed.avg",
                "--force-overwrite",
                "--export",
                str(ncu_base),
                str(probe_executable),
                "--repetitions",
                "1",
            ],
            capture_candidates=[ncu_report, ncu_base],
            version_command=["ncu", "--version"],
            verification_command=[
                "ncu",
                "--import",
                str(ncu_report),
                "--page",
                "raw",
                "--csv",
            ],
            verify_output=_verify_ncu_output,
        ),
    }
    if (
        "L4" not in str(device["name"])
        or device["major"] != 8
        or device["minor"] != 9
        or device["visible_device_count"] != 1
    ):
        raise RuntimeError(f"Modal did not allocate the locked L4/SM89 GPU: {device}")

    return {
        "result": "pass",
        "candidate_commit": candidate_commit,
        "source_bundle_sha256": source_sha256,
        "remote_call_id": modal.current_function_call_id(),
        "probe": {
            **sanitizer_probe,
            "known_answer_lengths": probe["known_answer_lengths"],
        },
        "compute_sanitizer": {
            "command": sanitizer_command,
            "result": "pass",
            "exit_status": 0,
        },
        "profilers": profilers,
        "gpu": {
            "model": "L4",
            "compute_capability": "8.9",
            "total_memory_mib": int(device["memory_bytes"]) // (1024 * 1024),
        },
        "runtime": {
            "cuda_runtime": _integer_cuda_version(device["runtime_integer"]),
            "driver": driver_version,
            "driver_api": _integer_cuda_version(device["driver_api_integer"]),
            "cublas": _cublas_header_version(),
        },
        "cuda_checks": cuda_checks["commands"],
        "wall_seconds": round(time.monotonic() - started, 6),
    }


class ModalDispatcher:
    def __init__(self, source_bundle: Any) -> None:
        self.source_bundle = source_bundle
        self.compile_call_id: str | None = None
        self.gpu_call_id: str | None = None

    def dispatch_compile(self) -> dict[str, object]:
        call = cuda_compile.spawn(
            self.source_bundle.content,
            self.source_bundle.sha256,
            self.source_bundle.commit,
        )
        self.compile_call_id = call.object_id
        result = call.get()
        if result.get("remote_call_id") != self.compile_call_id:
            raise RuntimeError("compile-stage call identifier mismatch")
        return result

    def dispatch_gpu(self) -> dict[str, object]:
        call = gpu_smoke.spawn(
            self.source_bundle.content,
            self.source_bundle.sha256,
            self.source_bundle.commit,
        )
        self.gpu_call_id = call.object_id
        result = call.get()
        if result.get("remote_call_id") != self.gpu_call_id:
            raise RuntimeError("GPU-stage call identifier mismatch")
        return result


def _validate_compile_result(
    result: object,
    *,
    source_bundle: Any,
    expected_call_id: str,
) -> dict[str, object]:
    if type(result) is not dict:
        raise RuntimeError("cuda_compile returned a non-object result")
    required = {
        "result",
        "candidate_commit",
        "source_bundle_sha256",
        "remote_call_id",
        "image",
        "observed",
        "compile_flags",
        "link_flags",
        "cuda_off_checks",
        "cuda_checks",
        "wall_seconds",
    }
    if set(result) != required:
        raise RuntimeError("cuda_compile returned malformed evidence")
    if (
        result["result"] != "pass"
        or result["candidate_commit"] != source_bundle.commit
        or result["source_bundle_sha256"] != source_bundle.sha256
        or result["remote_call_id"] != expected_call_id
    ):
        raise RuntimeError("cuda_compile provenance or result mismatch")
    if (
        type(result["result"]) is not str
        or type(result["candidate_commit"]) is not str
        or re.fullmatch(r"[0-9a-f]{40}", result["candidate_commit"]) is None
        or type(result["source_bundle_sha256"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}", result["source_bundle_sha256"]
        )
        is None
        or type(result["remote_call_id"]) is not str
        or not result["remote_call_id"].startswith("fc-")
    ):
        raise RuntimeError("cuda_compile provenance JSON types are invalid")
    image = result["image"]
    if type(image) is not dict or image != {
        "index_digest": LOCK["registry_image"]["index_digest"],
        "linux_amd64_digest": LOCK["registry_image"]["linux_amd64_digest"],
    }:
        raise RuntimeError("cuda_compile observed registry image disagrees with lock")
    observed = result["observed"]
    if type(observed) is not dict or set(observed) != {
        "operating_system",
        "platform",
        "host_compiler",
        "cmake",
        "ninja_distribution",
        "ninja_binary",
        "cuda_toolkit",
        "nvcc",
    }:
        raise RuntimeError("cuda_compile toolchain evidence is malformed")
    exact_observed = {
        "operating_system": LOCK["registry_image"]["operating_system"],
        "platform": LOCK["registry_image"]["platform"],
        "cmake": LOCK["toolchain"]["cmake"],
        "ninja_distribution": LOCK["toolchain"]["ninja_distribution"],
        "ninja_binary": LOCK["toolchain"]["ninja_binary"],
        "cuda_toolkit": LOCK["toolchain"]["cuda"],
    }
    for field, expected in exact_observed.items():
        if type(observed[field]) is not str or not observed[field]:
            raise RuntimeError(
                f"cuda_compile observed {field} is missing"
            )
        if observed[field] != expected:
            raise RuntimeError(
                f"cuda_compile observed {field} disagrees with lock"
            )
    for field in ("host_compiler", "nvcc"):
        if type(observed[field]) is not str or not observed[field]:
            raise RuntimeError(f"cuda_compile observed {field} is missing")
    if not re.fullmatch(r"gcc 13(?:\.[0-9]+){1,2}", observed["host_compiler"]):
        raise RuntimeError("cuda_compile host compiler violates lock policy")
    if not re.fullmatch(r"12\.6(?:\.[0-9]+)?", observed["nvcc"]):
        raise RuntimeError("cuda_compile nvcc violates lock policy")
    for field in ("compile_flags", "link_flags"):
        if (
            type(result[field]) is not list
            or not result[field]
            or any(type(item) is not str or not item for item in result[field])
        ):
            raise RuntimeError(f"cuda_compile {field} evidence is missing")
    joined_flags = " ".join(result["compile_flags"] + result["link_flags"])
    architectures = {
        int(value)
        for value in re.findall(
            r"(?:compute_|sm_|arch=)([0-9]+)", joined_flags
        )
    }
    forbidden = (
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
    if (
        architectures != {89}
        or "c++20" not in joined_flags
        or "cudart" not in joined_flags.lower()
        or "cublas" in joined_flags.lower()
        or any(flag in joined_flags.lower() for flag in forbidden)
    ):
        raise RuntimeError("cuda_compile flags violate the PR-6 lock")
    for field in ("cuda_off_checks", "cuda_checks"):
        if type(result[field]) is not list or not result[field]:
            raise RuntimeError(f"cuda_compile {field} evidence is missing")
    if (
        type(result["wall_seconds"]) is not float
        or result["wall_seconds"] < 0
        or result["wall_seconds"] > COMPILE_TIMEOUT_SECONDS
    ):
        raise RuntimeError("cuda_compile duration is outside its timeout")
    return result


def _validate_gpu_result(
    result: object,
    *,
    source_bundle: Any,
    expected_call_id: str,
    compile_call_id: str,
) -> dict[str, object]:
    if type(result) is not dict:
        raise RuntimeError("gpu_smoke returned a non-object result")
    required = {
        "result",
        "candidate_commit",
        "source_bundle_sha256",
        "remote_call_id",
        "probe",
        "compute_sanitizer",
        "profilers",
        "gpu",
        "runtime",
        "cuda_checks",
        "wall_seconds",
    }
    if set(result) != required:
        raise RuntimeError("gpu_smoke returned malformed evidence")
    if (
        result["result"] != "pass"
        or result["candidate_commit"] != source_bundle.commit
        or result["source_bundle_sha256"] != source_bundle.sha256
        or result["remote_call_id"] != expected_call_id
        or expected_call_id == compile_call_id
    ):
        raise RuntimeError("gpu_smoke provenance, result, or call id mismatch")
    if (
        type(result["result"]) is not str
        or type(result["candidate_commit"]) is not str
        or re.fullmatch(r"[0-9a-f]{40}", result["candidate_commit"]) is None
        or type(result["source_bundle_sha256"]) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}", result["source_bundle_sha256"]
        )
        is None
        or type(result["remote_call_id"]) is not str
        or not result["remote_call_id"].startswith("fc-")
    ):
        raise RuntimeError("gpu_smoke provenance JSON types are invalid")
    if (
        type(result["wall_seconds"]) is not float
        or result["wall_seconds"] < 0
        or result["wall_seconds"] > GPU_TIMEOUT_SECONDS
    ):
        raise RuntimeError("gpu_smoke duration is outside its timeout")
    if type(result["cuda_checks"]) is not list or not result["cuda_checks"]:
        raise RuntimeError("gpu_smoke CUDA test evidence is missing")
    probe = result["probe"]
    if type(probe) is not dict or probe != {
        "result": "pass",
        "known_answer_lengths": [0, 1, 255, 256, 257, 1025],
        "sentinels": "pass",
        "lifecycle_repetitions": 100,
    }:
        raise RuntimeError("gpu_smoke probe evidence is malformed")
    sanitizer = result["compute_sanitizer"]
    if (
        type(sanitizer) is not dict
        or set(sanitizer) != {"command", "result", "exit_status"}
        or sanitizer["result"] != "pass"
        or type(sanitizer["exit_status"]) is not int
        or sanitizer["exit_status"] != 0
        or type(sanitizer["command"]) is not list
    ):
        raise RuntimeError("gpu_smoke sanitizer evidence is malformed")
    sanitizer_command = sanitizer["command"]
    if (
        sanitizer_command[:5]
        != [
            "compute-sanitizer",
            "--tool",
            "memcheck",
            "--leak-check",
            "full",
        ]
        or "--error-exitcode=97" not in sanitizer_command
        or sanitizer_command[-2:] != ["--repetitions", "100"]
    ):
        raise RuntimeError("gpu_smoke sanitizer command is incomplete")
    gpu = result["gpu"]
    if (
        type(gpu) is not dict
        or set(gpu) != {"model", "compute_capability", "total_memory_mib"}
        or gpu["model"] != "L4"
        or gpu["compute_capability"] != "8.9"
        or type(gpu["total_memory_mib"]) is not int
        or gpu["total_memory_mib"] <= 0
    ):
        raise RuntimeError("gpu_smoke device evidence is malformed")
    runtime = result["runtime"]
    if type(runtime) is not dict or set(runtime) != {
        "cuda_runtime",
        "driver",
        "driver_api",
        "cublas",
    }:
        raise RuntimeError("gpu_smoke runtime evidence is malformed")
    for field in runtime:
        if type(runtime[field]) is not str or not runtime[field]:
            raise RuntimeError(f"gpu_smoke runtime {field} is missing")
    if type(result["profilers"]) is not dict or set(result["profilers"]) != {
        "nsys",
        "ncu",
    }:
        raise RuntimeError("gpu_smoke profiler evidence is malformed")
    profiler_keys = {
        "availability",
        "command",
        "version",
        "exit_status",
        "reason",
        "capture_sha256",
        "verification_command",
        "verification_exit_status",
        "evidence",
    }
    for name, profiler in result["profilers"].items():
        if type(profiler) is not dict or set(profiler) != profiler_keys:
            raise RuntimeError(f"gpu_smoke {name} profiler schema is malformed")
        if profiler["availability"] not in {"available", "unavailable"}:
            raise RuntimeError(f"gpu_smoke {name} availability is invalid")
        for field in ("command", "verification_command"):
            if (
                type(profiler[field]) is not list
                or not profiler[field]
                or any(type(item) is not str for item in profiler[field])
            ):
                raise RuntimeError(
                    f"gpu_smoke {name} {field} is malformed"
                )
        for field in ("exit_status", "verification_exit_status"):
            if type(profiler[field]) is not int:
                raise RuntimeError(
                    f"gpu_smoke {name} {field} has invalid JSON type"
                )
        if profiler["availability"] == "available":
            if (
                profiler["exit_status"] != 0
                or profiler["verification_exit_status"] != 0
                or profiler["reason"] is not None
                or type(profiler["capture_sha256"]) is not str
                or re.fullmatch(
                    r"[0-9a-f]{64}", profiler["capture_sha256"]
                )
                is None
                or type(profiler["evidence"]) is not str
                or not profiler["evidence"]
            ):
                raise RuntimeError(
                    f"gpu_smoke {name} available claim is unverified"
                )
        elif (
            type(profiler["reason"]) is not str
            or not profiler["reason"]
            or profiler["capture_sha256"] is not None
            or profiler["evidence"] is not None
        ):
            raise RuntimeError(
                f"gpu_smoke {name} unavailable claim is malformed"
            )
    return result


def _stage_manifest(
    stage_index: int, result: dict[str, object]
) -> dict[str, object]:
    stage = asdict(STAGES[stage_index])
    del stage["name"]
    stage["result"] = "pass"
    stage["wall_seconds"] = result["wall_seconds"]
    return stage


def _actual_cost(
    compile_seconds: object, gpu_seconds: object
) -> tuple[Decimal, Decimal]:
    compile_duration = Decimal(str(compile_seconds))
    gpu_duration = Decimal(str(gpu_seconds))
    if (
        not compile_duration.is_finite()
        or not gpu_duration.is_finite()
        or compile_duration < 0
        or gpu_duration < 0
        or compile_duration > COMPILE_TIMEOUT_SECONDS
        or gpu_duration > GPU_TIMEOUT_SECONDS
    ):
        raise RuntimeError("remote stage duration is outside its locked timeout")
    compile_cost = compile_duration * (
        Decimal("2") * CPU_CORE_SECOND_USD
        + Decimal("4") * MEMORY_GIB_SECOND_USD
    )
    gpu_cost = gpu_duration * (
        Decimal("2") * CPU_CORE_SECOND_USD
        + Decimal("4") * MEMORY_GIB_SECOND_USD
        + GPU_SECOND_USD["L4"]
    )
    return compile_cost + gpu_cost, gpu_duration / Decimal("60")


def _manifest(
    *,
    month_to_date: Decimal,
    source_bundle: Any,
    dispatcher: ModalDispatcher,
    compile_result: dict[str, object],
    gpu_result: dict[str, object],
    dependency_lock_sha256: str,
    application_id: str,
    modal_sdk_version: str,
) -> dict[str, object]:
    if dispatcher.compile_call_id is None or dispatcher.gpu_call_id is None:
        raise RuntimeError("both remote call identifiers are required")
    if type(application_id) is not str or not application_id:
        raise RuntimeError("Modal application identifier is unavailable")
    actual_cost, gpu_minutes = _actual_cost(
        compile_result["wall_seconds"], gpu_result["wall_seconds"]
    )
    observed = {
        **compile_result["observed"],
        **gpu_result["runtime"],
    }
    return {
        "schema_version": 1,
        "result": "pass",
        "source": {
            "commit": source_bundle.commit,
            "bundle_sha256": source_bundle.sha256,
            "dirty": False,
        },
        "image": {
            "reference": LOCK["registry_image"]["reference"],
            "locked_index_digest": LOCK["registry_image"]["index_digest"],
            "observed_index_digest": compile_result["image"]["index_digest"],
            "locked_linux_amd64_digest": LOCK["registry_image"][
                "linux_amd64_digest"
            ],
            "observed_linux_amd64_digest": compile_result["image"][
                "linux_amd64_digest"
            ],
            "operating_system": observed["operating_system"],
            "platform": observed["platform"],
        },
        "toolchain": {
            "locked": LOCK["toolchain"],
            "observed": {
                "operating_system": observed["operating_system"],
                "host_compiler": observed["host_compiler"],
                "cmake": observed["cmake"],
                "ninja_distribution": observed["ninja_distribution"],
                "ninja_binary": observed["ninja_binary"],
                "cuda_toolkit": observed["cuda_toolkit"],
                "nvcc": observed["nvcc"],
                "cuda_runtime": observed["cuda_runtime"],
                "driver": observed["driver"],
                "driver_api": observed["driver_api"],
                "cublas": observed["cublas"],
            },
            "cuda_architectures": [89],
            "compile_flags": compile_result["compile_flags"],
            "link_flags": compile_result["link_flags"],
        },
        "modal": {
            "sdk_version": modal_sdk_version,
            "dependency_lock_sha256": dependency_lock_sha256,
            "application_id": application_id,
            "compile_call_id": dispatcher.compile_call_id,
            "gpu_call_id": dispatcher.gpu_call_id,
        },
        "gpu": gpu_result["gpu"],
        "stages": {
            "cuda_compile": _stage_manifest(0, compile_result),
            "gpu_smoke": _stage_manifest(1, gpu_result),
        },
        "probe": gpu_result["probe"],
        "compute_sanitizer": gpu_result["compute_sanitizer"],
        "profilers": gpu_result["profilers"],
        "budget": {
            "month_to_date_usd": str(month_to_date),
            "maximum_planned_cost_usd": f"{CHAIN_COST_CEILING_USD:.6f}",
            "estimated_actual_compute_cost_usd": format(actual_cost, "f"),
            "gpu_minutes": format(gpu_minutes, "f"),
            "monthly_budget_usd": "30",
            "project_soft_cap_usd": "24",
            "reserve_usd": "6",
            "billing_report_caveat": (
                "Modal billing is authoritative; image construction and "
                "storage charges may be absent from this function estimate."
            ),
        },
    }


@app.local_entrypoint()
def main(authorization_ticket: str) -> None:
    ticket = AuthorizationTicketStore().consume(Path(authorization_ticket))
    month_to_date = parse_cost(ticket["month_to_date_usd"])
    dry_run_manifest(month_to_date)
    source_bundle = create_source_bundle(PROJECT_ROOT)
    dependency_lock_sha256 = source_bundle_member_sha256(
        source_bundle, "tools/modal/requirements.txt"
    )
    if (
        source_bundle.commit != ticket["candidate_commit"]
        or source_bundle.sha256 != ticket["source_bundle_sha256"]
        or dependency_lock_sha256 != ticket["dependency_lock_sha256"]
        or ticket["gate_id"] != GATE_ID
    ):
        raise RuntimeError("authorization ticket is not bound to this source")
    TrialLedger().require_reserved(
        ticket["reservation_id"],
        candidate_commit=source_bundle.commit,
        gate_id=GATE_ID,
    )
    cache = EvidenceCache()
    cached = cache.load(
        candidate_commit=source_bundle.commit,
        gate_id=GATE_ID,
    )
    if cached is not None:
        validate_manifest(
            cached,
            LOCK,
            expected_source_commit=source_bundle.commit,
            expected_source_bundle_sha256=source_bundle.sha256,
            expected_dependency_lock_sha256=dependency_lock_sha256,
        )
        print(json.dumps(cached, indent=2, sort_keys=True, allow_nan=False))
        return

    dispatcher = ModalDispatcher(source_bundle)
    compile_result = dispatcher.dispatch_compile()
    if dispatcher.compile_call_id is None:
        raise RuntimeError("cuda_compile call identifier is unavailable")
    compile_result = _validate_compile_result(
        compile_result,
        source_bundle=source_bundle,
        expected_call_id=dispatcher.compile_call_id,
    )
    gpu_result = dispatcher.dispatch_gpu()
    if dispatcher.gpu_call_id is None:
        raise RuntimeError("gpu_smoke call identifier is unavailable")
    gpu_result = _validate_gpu_result(
        gpu_result,
        source_bundle=source_bundle,
        expected_call_id=dispatcher.gpu_call_id,
        compile_call_id=dispatcher.compile_call_id,
    )
    manifest = _manifest(
        month_to_date=month_to_date,
        source_bundle=source_bundle,
        dispatcher=dispatcher,
        compile_result=compile_result,
        gpu_result=gpu_result,
        dependency_lock_sha256=dependency_lock_sha256,
        application_id=app.app_id,
        modal_sdk_version=modal.__version__,
    )
    def validate(value: dict[str, object]) -> None:
        validate_manifest(
            value,
            LOCK,
            expected_source_commit=source_bundle.commit,
            expected_source_bundle_sha256=source_bundle.sha256,
            expected_dependency_lock_sha256=dependency_lock_sha256,
        )

    validate(manifest)
    cache.store(
        candidate_commit=source_bundle.commit,
        gate_id=GATE_ID,
        manifest=manifest,
        lock=LOCK,
        expected_source_bundle_sha256=source_bundle.sha256,
        expected_dependency_lock_sha256=dependency_lock_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
