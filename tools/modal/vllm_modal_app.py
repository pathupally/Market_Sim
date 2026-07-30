"""Bounded vLLM/SmolLM2 L4 conformance run for PR 7."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import io
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time

import modal

from tools.inference.contract import (
    HardwareIdentity,
    InferenceRequest,
    ModelIdentity,
    SourceIdentity,
    validate_run_payload,
)
from tools.inference.vllm_backend import (
    VLLM_VERSION,
    VllmConfig,
    create_engine,
    run_greedy_batch,
)
from tools.modal.cuda_ci import create_source_bundle, parse_cost
from tools.modal.modal_budget import (
    CPU_CORE_SECOND_USD,
    GPU_SECOND_USD,
    MEMORY_GIB_SECOND_USD,
    require_project_headroom,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCK = json.loads(
    (PROJECT_ROOT / "tools/modal/vllm-lock.json").read_text(encoding="utf-8")
)
EXECUTION = LOCK["execution"]
MODEL = LOCK["model"]
REMOTE_SOURCE = Path("/tmp/marketforge-pr7-source")
REMOTE_BUILD = Path("/tmp/marketforge-pr7-build")
TIMEOUT_SECONDS = int(EXECUTION["timeout_seconds"])
PHYSICAL_CORES = Decimal(str(EXECUTION["physical_cores"]))
MEMORY_GIB = Decimal(str(EXECUTION["memory_mib"])) / Decimal(1024)
MAXIMUM_COST_USD = Decimal(TIMEOUT_SECONDS) * (
    PHYSICAL_CORES * CPU_CORE_SECOND_USD
    + MEMORY_GIB * MEMORY_GIB_SECOND_USD
    + GPU_SECOND_USD["L4"]
)

if MAXIMUM_COST_USD != Decimal("0.23936400"):
    raise RuntimeError("vLLM L4 cost ceiling changed")


def verify_checkpoint(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    if (
        path.stat().st_size != int(MODEL["checkpoint_bytes"])
        or digest.hexdigest() != MODEL["checkpoint_sha256"]
    ):
        raise RuntimeError("cached SmolLM2 checkpoint violates its lock")


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
    result: dict[str, object] = {
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
        (REMOTE_SOURCE / "tools/modal/vllm-lock.json").read_text(
            encoding="utf-8"
        )
    )
    if embedded_lock != LOCK:
        raise RuntimeError("source-bundle vLLM lock mismatch")
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


def _validate_benchmark(
    benchmark: object,
    *,
    operator: str,
    measurement_count: int,
) -> None:
    if type(benchmark) is not dict or set(benchmark) != {
        "schema_version",
        "result",
        "operator",
        "model_shape",
        "gpu",
        "measurements",
    }:
        raise RuntimeError("native benchmark schema mismatch")
    gpu = benchmark["gpu"]
    measurements = benchmark["measurements"]
    if (
        type(benchmark["schema_version"]) is not int
        or benchmark["schema_version"] != 1
        or benchmark["result"] != "pass"
        or benchmark["operator"] != operator
        or type(benchmark["model_shape"]) is not str
        or type(gpu) is not dict
        or "L4" not in str(gpu.get("name"))
        or gpu.get("compute_capability") != "8.9"
        or type(measurements) is not list
        or len(measurements) != measurement_count
    ):
        raise RuntimeError("native benchmark evidence mismatch")
    if operator == "rms_norm_f32":
        measurement_fields = {
            "rows",
            "hidden_size",
            "iterations",
            "average_microseconds",
            "logical_gib_per_second",
        }
        rate_field = "logical_gib_per_second"
    else:
        measurement_fields = {
            "rows",
            "input_features",
            "output_features",
            "iterations",
            "average_microseconds",
            "tera_flops",
        }
        rate_field = "tera_flops"
    for measurement in measurements:
        if type(measurement) is not dict or set(measurement) != measurement_fields:
            raise RuntimeError("native benchmark measurement is malformed")
        for field in ("average_microseconds", rate_field):
            value = measurement.get(field)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise RuntimeError("native benchmark timing is malformed")


vllm_image = (
    modal.Image.from_registry(
        LOCK["base_image"]["reference"],
        add_python=LOCK["python"],
    )
    .entrypoint([])
    .uv_pip_install(
        f"cmake=={LOCK['packages']['cmake']}",
        f"ninja=={LOCK['packages']['ninja']}",
        f"nvidia-ml-py=={LOCK['packages']['nvidia-ml-py']}",
        f"vllm=={LOCK['packages']['vllm']}",
    )
    .env(
        {
            "HF_HOME": "/models/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    .add_local_python_source("tools")
)

model_cache = modal.Volume.from_name(
    "marketforge-smollm2-v1",
    create_if_missing=True,
)
app = modal.App(
    "marketforge-pr7-native-vllm",
    tags={"project": "marketforge", "purpose": "pr7-native-vllm-gate"},
)


@app.function(
    image=vllm_image,
    gpu="L4",
    cpu=float(PHYSICAL_CORES),
    memory=int(EXECUTION["memory_mib"]),
    timeout=TIMEOUT_SECONDS,
    max_containers=int(EXECUTION["max_containers"]),
    single_use_containers=True,
    volumes={"/models": model_cache},
)
@modal.concurrent(max_inputs=1)
def pr7_l4_gate(
    source_content: bytes,
    source_bundle_sha256: str,
    candidate_commit: str,
) -> dict[str, object]:
    import threading

    import torch
    from huggingface_hub import snapshot_download
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
        nvmlShutdown,
    )

    if VLLM_VERSION != LOCK["packages"]["vllm"]:
        raise RuntimeError("vLLM adapter and image locks disagree")
    if (
        re.fullmatch(r"[0-9a-f]{40}", candidate_commit) is None
        or re.fullmatch(r"[0-9a-f]{64}", source_bundle_sha256) is None
    ):
        raise RuntimeError("source identity is malformed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    properties = torch.cuda.get_device_properties(0)
    if properties.major != 8 or properties.minor != 9:
        raise RuntimeError("PR 7 vLLM conformance requires compute 8.9")
    torch.cuda.reset_peak_memory_stats()

    source = _extract_bundle(source_content, source_bundle_sha256)
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
    rmsnorm_run = _run(
        [str(REMOTE_BUILD / "marketforge_cuda_rmsnorm_bench")],
        environment=environment,
    )
    linear_run = _run(
        [str(REMOTE_BUILD / "marketforge_cuda_linear_bench")],
        environment=environment,
    )
    rmsnorm_benchmark = _strict_json_line(str(rmsnorm_run["output"]))
    linear_benchmark = _strict_json_line(str(linear_run["output"]))
    _validate_benchmark(
        rmsnorm_benchmark,
        operator="rms_norm_f32",
        measurement_count=4,
    )
    _validate_benchmark(
        linear_benchmark,
        operator="linear_f16_fp32_accumulate",
        measurement_count=3,
    )

    model = ModelIdentity(
        model_id=MODEL["id"],
        repository=MODEL["repository"],
        revision=MODEL["revision"],
        checkpoint_sha256=MODEL["checkpoint_sha256"],
        vocabulary_size=int(MODEL["vocabulary_size"]),
    )
    snapshot = Path(
        snapshot_download(
            repo_id=model.repository,
            revision=model.revision,
            cache_dir="/models/huggingface/hub",
            allow_patterns=["config.json", "model.safetensors"],
        )
    )
    checkpoint = snapshot / "model.safetensors"
    verify_checkpoint(checkpoint)

    config = VllmConfig(
        max_output_tokens=3,
        max_model_len=int(EXECUTION["max_model_len"]),
        max_num_seqs=int(EXECUTION["max_num_seqs"]),
        gpu_memory_utilization=float(EXECUTION["gpu_memory_utilization"]),
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    nvmlInit()
    nvml_handle = nvmlDeviceGetHandleByIndex(0)
    memory_stop = threading.Event()
    peak_gpu_memory_bytes = [int(nvmlDeviceGetMemoryInfo(nvml_handle).used)]
    memory_errors: list[str] = []

    def sample_device_memory() -> None:
        try:
            while not memory_stop.is_set():
                used = int(nvmlDeviceGetMemoryInfo(nvml_handle).used)
                peak_gpu_memory_bytes[0] = max(
                    peak_gpu_memory_bytes[0], used
                )
                memory_stop.wait(0.01)
        except Exception as error:
            memory_errors.append(str(error))
            memory_stop.set()

    memory_thread = threading.Thread(
        target=sample_device_memory,
        name="pr7-gpu-memory-sampler",
        daemon=True,
    )
    memory_thread.start()
    try:
        engine, sampling_params_factory, load_seconds = create_engine(
            model,
            config,
            model_path=str(snapshot),
        )
        run = run_greedy_batch(
            engine=engine,
            sampling_params_factory=sampling_params_factory,
            model=model,
            source=SourceIdentity(
                commit=candidate_commit,
                bundle_sha256=source_bundle_sha256,
            ),
            hardware=HardwareIdentity(
                device_name=torch.cuda.get_device_name(0),
                compute_capability=f"{properties.major}.{properties.minor}",
                cuda_version=str(torch.version.cuda),
            ),
            requests=(InferenceRequest("pr4-greedy", (0, 1, 2, 3)),),
            config=config,
            model_load_seconds=load_seconds,
            peak_gpu_memory=lambda: peak_gpu_memory_bytes[0],
        )
    finally:
        memory_stop.set()
        memory_thread.join(timeout=1.0)
        nvmlShutdown()
    if memory_errors or run.metrics.peak_gpu_memory_bytes <= 0:
        raise RuntimeError("device-wide NVML peak sampling failed")
    payload = run.to_payload()
    if payload["requests"][0]["generated_token_ids"] != [198, 198, 504]:
        raise RuntimeError("vLLM tokens do not match the PR 4 oracle")
    validate_run_payload(payload)
    model_cache.commit()
    return {
        "schema_version": 1,
        "result": "pass",
        "source": {
            "commit": candidate_commit,
            "bundle_sha256": source_bundle_sha256,
        },
        "native_cuda": {
            "commands": commands,
            "rmsnorm_benchmark": rmsnorm_benchmark,
            "linear_benchmark": linear_benchmark,
        },
        "inference": payload,
    }


@app.local_entrypoint()
def main(month_to_date_usd: str) -> None:
    month_to_date = parse_cost(month_to_date_usd)
    require_project_headroom(
        month_to_date_usd=month_to_date,
        planned_cost_usd=MAXIMUM_COST_USD,
    )
    bundle = create_source_bundle(PROJECT_ROOT)
    result = pr7_l4_gate.remote(
        bundle.content,
        bundle.sha256,
        bundle.commit,
    )
    if type(result) is not dict or set(result) != {
        "schema_version",
        "result",
        "source",
        "native_cuda",
        "inference",
    }:
        raise RuntimeError("remote PR 7 gate schema mismatch")
    validate_run_payload(result["inference"])
    if (
        result["source"]["commit"] != bundle.commit
        or result["source"]["bundle_sha256"] != bundle.sha256
        or result["inference"]["source"] != result["source"]
    ):
        raise RuntimeError("remote PR 7 evidence is not source-bound")
    result["budget"] = {
        "month_to_date_usd": str(month_to_date),
        "maximum_compute_cost_usd": f"{MAXIMUM_COST_USD:.6f}",
        "project_soft_cap_usd": "24",
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
