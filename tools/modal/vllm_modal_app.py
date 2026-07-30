"""Bounded native-CUDA/vLLM serving ablation gate for PR 8."""

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
import selectors
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import modal

from tools.inference.contract import (
    ModelIdentity,
    validate_run_payload,
)
from tools.inference.vllm_ablation_worker import RESULT_PREFIX
from tools.inference.vllm_backend import VLLM_VERSION
from tools.modal.cuda_ci import create_source_bundle, parse_cost
from tools.modal.modal_budget import (
    CPU_CORE_SECOND_USD,
    GPU_SECOND_USD,
    MEMORY_GIB_SECOND_USD,
    require_project_headroom,
)


REMOTE_LOCK_PATH = Path("/root/marketforge-vllm-lock.json")
if modal.is_local():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    LOCK_PATH = PROJECT_ROOT / "tools/modal/vllm-lock.json"
else:
    PROJECT_ROOT = Path("/root")
    LOCK_PATH = REMOTE_LOCK_PATH
LOCK = json.loads(
    LOCK_PATH.read_text(encoding="utf-8")
)
EXECUTION = LOCK["execution"]
MODEL = LOCK["model"]
REMOTE_SOURCE = Path("/tmp/marketforge-pr8-source")
REMOTE_BUILD = Path("/tmp/marketforge-pr8-build")
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


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("vLLM worker process group did not stop") from error


def _run_vllm_worker(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float = 330.0,
) -> dict[str, object]:
    if timeout_seconds <= 0:
        raise ValueError("vLLM worker timeout must be positive")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            **environment,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    if process.stdout is None:
        _terminate_process_group(process)
        raise RuntimeError("vLLM worker stdout pipe is unavailable")

    started = time.monotonic()
    output: list[str] = []
    payload: dict[str, object] | None = None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while time.monotonic() - started < timeout_seconds:
            events = selector.select(timeout=1.0)
            if not events:
                if process.poll() is not None:
                    break
                continue
            line = process.stdout.readline()
            if line == "":
                if process.poll() is not None:
                    break
                continue
            output.append(line)
            if line.startswith(RESULT_PREFIX):
                payload = _prefixed_json_line(line, RESULT_PREFIX)
                break
    finally:
        selector.close()
        try:
            _terminate_process_group(process)
        finally:
            process.stdout.close()

    if payload is None:
        tail = "".join(output[-200:])
        raise RuntimeError(
            "vLLM worker produced no validated artifact within "
            f"{timeout_seconds:.0f} seconds\n{tail}"
        )
    return payload


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


def _prefixed_json_line(output: str, prefix: str) -> dict[str, object]:
    lines = [
        line[len(prefix) :]
        for line in output.splitlines()
        if line.startswith(prefix)
    ]
    if len(lines) != 1:
        raise RuntimeError("worker emitted no unique prefixed JSON line")

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
        raise RuntimeError("worker JSON is not an object")
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
    elif operator == "linear_f16_fp32_accumulate":
        measurement_fields = {
            "rows",
            "input_features",
            "output_features",
            "iterations",
            "average_microseconds",
            "tera_flops",
        }
        rate_field = "tera_flops"
    elif operator == "transformer_elementwise_f16":
        measurement_fields = {
            "operation",
            "rows",
            "elements",
            "iterations",
            "average_microseconds",
            "logical_gib_per_second",
        }
        rate_field = "logical_gib_per_second"
    else:
        raise RuntimeError("unknown native benchmark operator")
    for measurement in measurements:
        if type(measurement) is not dict or set(measurement) != measurement_fields:
            raise RuntimeError("native benchmark measurement is malformed")
        if operator == "transformer_elementwise_f16" and measurement[
            "operation"
        ] not in {"rope_f16", "swiglu_f16"}:
            raise RuntimeError("native benchmark operation is malformed")
        for field in ("average_microseconds", rate_field):
            value = measurement.get(field)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise RuntimeError("native benchmark timing is malformed")


def _validate_restricted_benchmark(benchmark: object) -> None:
    if type(benchmark) is not dict or set(benchmark) != {
        "schema_version",
        "result",
        "operator",
        "grammar",
        "vocabulary_size",
        "grammar_states",
        "grammar_arcs",
        "maximum_allowed_tokens",
        "gpu",
        "measurements",
    }:
        raise RuntimeError("restricted benchmark schema mismatch")
    gpu = benchmark["gpu"]
    measurements = benchmark["measurements"]
    if (
        benchmark["schema_version"] != 1
        or benchmark["result"] != "pass"
        or benchmark["operator"] != "restricted_greedy_f16"
        or benchmark["grammar"] != "smollm2_market_action_v1"
        or benchmark["vocabulary_size"] != 49_152
        or type(benchmark["grammar_states"]) is not int
        or benchmark["grammar_states"] <= 0
        or type(benchmark["grammar_arcs"]) is not int
        or benchmark["grammar_arcs"] <= 0
        or type(benchmark["maximum_allowed_tokens"]) is not int
        or benchmark["maximum_allowed_tokens"] <= 0
        or type(gpu) is not dict
        or set(gpu) != {"name", "compute_capability"}
        or "L4" not in str(gpu["name"])
        or gpu["compute_capability"] != "8.9"
        or type(measurements) is not list
        or len(measurements) != 3
    ):
        raise RuntimeError("restricted benchmark evidence mismatch")
    fields = {
        "rows",
        "total_allowed_candidates",
        "iterations",
        "average_microseconds",
        "sequences_per_second",
        "candidate_gib_per_second",
    }
    for measurement in measurements:
        if type(measurement) is not dict or set(measurement) != fields:
            raise RuntimeError(
                "restricted benchmark measurement is malformed"
            )
        for field in ("rows", "total_allowed_candidates", "iterations"):
            value = measurement[field]
            if type(value) is not int or value <= 0:
                raise RuntimeError(
                    "restricted benchmark count is malformed"
                )
        for field in (
            "average_microseconds",
            "sequences_per_second",
            "candidate_gib_per_second",
        ):
            value = measurement[field]
            if (
                type(value) is not float
                or not math.isfinite(value)
                or value <= 0
            ):
                raise RuntimeError(
                    "restricted benchmark timing is malformed"
                )


def _inference_seconds(payload: dict[str, object]) -> float:
    metrics = payload["metrics"]
    if type(metrics) is not dict:
        raise RuntimeError("vLLM metrics are malformed")
    return float(metrics["inference_seconds"])


def _requests_per_second(payload: dict[str, object]) -> float:
    metrics = payload["metrics"]
    if type(metrics) is not dict:
        raise RuntimeError("vLLM metrics are malformed")
    return float(metrics["requests_per_second"])


def _validate_mode_ablations(
    value: object,
    *,
    mode: str,
    source: dict[str, object],
) -> None:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "result",
        "mode",
        "settings",
        "runs",
    }:
        raise RuntimeError("vLLM mode ablation schema mismatch")
    settings = value["settings"]
    runs = value["runs"]
    expected_run_keys = (
        {"single", "batch"}
        if mode == "eager"
        else {"single", "batch", "prefix_cold", "prefix_warm"}
    )
    if (
        value["schema_version"] != 1
        or value["result"] != "pass"
        or value["mode"] != mode
        or type(settings) is not dict
        or settings
        != {
            "batch_size": 16,
            "prefix_batch_size": 8,
            "prefix_tokens": 128,
            "prefix_caching": True,
        }
        or type(runs) is not dict
        or set(runs) != expected_run_keys
    ):
        raise RuntimeError("vLLM mode ablation evidence mismatch")
    expected_counts = {
        "single": 1,
        "batch": 16,
        "prefix_cold": 8,
        "prefix_warm": 8,
    }
    for name, payload in runs.items():
        validate_run_payload(payload)
        if (
            payload["source"] != source
            or payload["backend"]["mode"] != mode
            or payload["features"]["prefix_caching"] is not True
            or payload["features"]["cuda_graphs"]
            is not (mode == "cuda_graph")
            or len(payload["requests"]) != expected_counts[name]
        ):
            raise RuntimeError("vLLM ablation run identity mismatch")
    expected_tokens = [198, 198, 504]
    if any(
        request["generated_token_ids"] != expected_tokens
        for name in ("single", "batch")
        for request in runs[name]["requests"]
    ):
        raise RuntimeError(
            "vLLM ablation tokens do not match the oracle"
        )
    if mode == "cuda_graph" and (
        runs["prefix_cold"]["requests"]
        != runs["prefix_warm"]["requests"]
    ):
        raise RuntimeError(
            "vLLM prefix-cache replay changed exact outputs"
        )


def _build_vllm_ablations(
    eager: dict[str, object],
    cuda_graph: dict[str, object],
    *,
    source: dict[str, object],
) -> dict[str, object]:
    _validate_mode_ablations(eager, mode="eager", source=source)
    _validate_mode_ablations(
        cuda_graph,
        mode="cuda_graph",
        source=source,
    )
    eager_runs = eager["runs"]
    graph_runs = cuda_graph["runs"]
    if type(eager_runs) is not dict or type(graph_runs) is not dict:
        raise RuntimeError("vLLM ablation runs are malformed")
    comparisons = {
        "graph_single_speedup": (
            _inference_seconds(eager_runs["single"])
            / _inference_seconds(graph_runs["single"])
        ),
        "graph_batch_speedup": (
            _inference_seconds(eager_runs["batch"])
            / _inference_seconds(graph_runs["batch"])
        ),
        "eager_batch_request_throughput_gain": (
            _requests_per_second(eager_runs["batch"])
            / _requests_per_second(eager_runs["single"])
        ),
        "graph_batch_request_throughput_gain": (
            _requests_per_second(graph_runs["batch"])
            / _requests_per_second(graph_runs["single"])
        ),
        "warm_prefix_speedup": (
            _inference_seconds(graph_runs["prefix_cold"])
            / _inference_seconds(graph_runs["prefix_warm"])
        ),
    }
    result = {
        "schema_version": 1,
        "result": "pass",
        "source": source,
        "eager": eager,
        "cuda_graph": cuda_graph,
        "comparisons": comparisons,
    }
    _validate_vllm_ablations(result)
    return result


def _validate_vllm_ablations(value: object) -> None:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "result",
        "source",
        "eager",
        "cuda_graph",
        "comparisons",
    }:
        raise RuntimeError("vLLM ablation schema mismatch")
    source = value["source"]
    comparisons = value["comparisons"]
    if (
        value["schema_version"] != 1
        or value["result"] != "pass"
        or type(source) is not dict
        or set(source) != {"commit", "bundle_sha256"}
        or re.fullmatch(r"[0-9a-f]{40}", str(source["commit"])) is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(source["bundle_sha256"])
        )
        is None
        or type(comparisons) is not dict
        or set(comparisons)
        != {
            "graph_single_speedup",
            "graph_batch_speedup",
            "eager_batch_request_throughput_gain",
            "graph_batch_request_throughput_gain",
            "warm_prefix_speedup",
        }
    ):
        raise RuntimeError("vLLM ablation identity mismatch")
    _validate_mode_ablations(
        value["eager"],
        mode="eager",
        source=source,
    )
    _validate_mode_ablations(
        value["cuda_graph"],
        mode="cuda_graph",
        source=source,
    )
    for measurement in comparisons.values():
        if (
            type(measurement) is not float
            or not math.isfinite(measurement)
            or measurement <= 0
        ):
            raise RuntimeError(
                "vLLM ablation comparison is malformed"
            )


def _validate_native_inference(value: object) -> None:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "result",
        "backend",
        "model",
        "gpu",
        "prompt_token_ids",
        "generated_token_ids",
        "model_load_seconds",
        "inference_seconds",
        "context_length",
        "memory",
    }:
        raise RuntimeError("native inference schema mismatch")
    gpu = value["gpu"]
    memory = value["memory"]
    if (
        value["schema_version"] != 1
        or value["result"] != "pass"
        or value["backend"] != "native_cuda_f16"
        or value["model"] != "SmolLM2-135M"
        or value["prompt_token_ids"] != [0, 1, 2, 3]
        or value["generated_token_ids"] != [198, 198, 504]
        or value["context_length"] != 6
        or type(gpu) is not dict
        or gpu.get("compute_capability") != "8.9"
        or "L4" not in str(gpu.get("name"))
        or type(memory) is not dict
        or set(memory) != {
            "weight_bytes",
            "kv_bytes",
            "execution_bytes",
            "total_device_bytes",
        }
        or memory["total_device_bytes"]
        != memory["weight_bytes"]
        + memory["kv_bytes"]
        + memory["execution_bytes"]
    ):
        raise RuntimeError("native inference evidence mismatch")
    for field in ("model_load_seconds", "inference_seconds"):
        measurement = value[field]
        if (
            type(measurement) is not float
            or not math.isfinite(measurement)
            or measurement <= 0.0
        ):
            raise RuntimeError("native inference timing is malformed")
    for measurement in memory.values():
        if type(measurement) is not int or measurement <= 0:
            raise RuntimeError("native inference memory is malformed")


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
if modal.is_local():
    vllm_image = vllm_image.add_local_file(
        LOCK_PATH,
        str(REMOTE_LOCK_PATH),
    )

model_cache = modal.Volume.from_name(
    "marketforge-smollm2-v1",
    create_if_missing=True,
)
app = modal.App(
    "marketforge-pr8-serving-ablation",
    tags={
        "project": "marketforge",
        "purpose": "pr8-serving-ablation-gate",
    },
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
def pr8_l4_gate(
    source_content: bytes,
    source_bundle_sha256: str,
    candidate_commit: str,
) -> dict[str, object]:
    import torch
    from huggingface_hub import snapshot_download

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
        raise RuntimeError("PR 8 serving gate requires compute 8.9")
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
    transformer_ops_run = _run(
        [str(REMOTE_BUILD / "marketforge_cuda_transformer_ops_bench")],
        environment=environment,
    )
    restricted_greedy_run = _run(
        [str(REMOTE_BUILD / "marketforge_cuda_restricted_greedy_bench")],
        environment=environment,
    )
    rmsnorm_benchmark = _strict_json_line(str(rmsnorm_run["output"]))
    linear_benchmark = _strict_json_line(str(linear_run["output"]))
    transformer_ops_benchmark = _strict_json_line(
        str(transformer_ops_run["output"])
    )
    restricted_greedy_benchmark = _strict_json_line(
        str(restricted_greedy_run["output"])
    )
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
    _validate_benchmark(
        transformer_ops_benchmark,
        operator="transformer_elementwise_f16",
        measurement_count=6,
    )
    _validate_restricted_benchmark(restricted_greedy_benchmark)

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
    native_inference_run = _run(
        [
            str(REMOTE_BUILD / "marketforge_cuda_smollm2_conformance"),
            str(checkpoint),
        ],
        environment=environment,
    )
    native_inference = _strict_json_line(
        str(native_inference_run["output"])
    )
    _validate_native_inference(native_inference)

    worker_environment = {
        **environment,
        "PYTHONPATH": str(source),
    }
    worker_command = [
        sys.executable,
        "-m",
        "tools.inference.vllm_ablation_worker",
        "--model-path",
        str(snapshot),
        "--source-commit",
        candidate_commit,
        "--source-bundle-sha256",
        source_bundle_sha256,
    ]
    eager_ablation = _run_vllm_worker(
        [*worker_command, "--mode", "eager"],
        cwd=source,
        environment=worker_environment,
    )
    graph_ablation = _run_vllm_worker(
        [*worker_command, "--mode", "cuda_graph"],
        cwd=source,
        environment=worker_environment,
    )
    source_identity = {
        "commit": candidate_commit,
        "bundle_sha256": source_bundle_sha256,
    }
    ablations = _build_vllm_ablations(
        eager_ablation,
        graph_ablation,
        source=source_identity,
    )
    model_cache.commit()
    return {
        "schema_version": 1,
        "result": "pass",
        "source": source_identity,
        "native_cuda": {
            "commands": commands,
            "rmsnorm_benchmark": rmsnorm_benchmark,
            "linear_benchmark": linear_benchmark,
            "transformer_ops_benchmark": transformer_ops_benchmark,
            "restricted_greedy_benchmark": restricted_greedy_benchmark,
            "smollm2_inference": native_inference,
        },
        "vllm_ablations": ablations,
    }


@app.local_entrypoint()
def main(month_to_date_usd: str) -> None:
    month_to_date = parse_cost(month_to_date_usd)
    require_project_headroom(
        month_to_date_usd=month_to_date,
        planned_cost_usd=MAXIMUM_COST_USD,
    )
    bundle = create_source_bundle(PROJECT_ROOT)
    result = pr8_l4_gate.remote(
        bundle.content,
        bundle.sha256,
        bundle.commit,
    )
    if type(result) is not dict or set(result) != {
        "schema_version",
        "result",
        "source",
        "native_cuda",
        "vllm_ablations",
    }:
        raise RuntimeError("remote PR 8 gate schema mismatch")
    _validate_vllm_ablations(result["vllm_ablations"])
    if (
        result["source"]["commit"] != bundle.commit
        or result["source"]["bundle_sha256"] != bundle.sha256
        or result["vllm_ablations"]["source"] != result["source"]
    ):
        raise RuntimeError("remote PR 8 evidence is not source-bound")
    result["budget"] = {
        "month_to_date_usd": str(month_to_date),
        "maximum_compute_cost_usd": f"{MAXIMUM_COST_USD:.6f}",
        "project_soft_cap_usd": "24",
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
