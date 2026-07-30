"""One-process vLLM serving-mode worker used by the PR 8 Modal gate."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import threading
from typing import Sequence

from tools.inference.contract import (
    HardwareIdentity,
    InferenceRequest,
    ModelIdentity,
    SourceIdentity,
    validate_run_payload,
)
from tools.inference.vllm_backend import (
    VllmConfig,
    create_engine,
    run_greedy_batch,
)


RESULT_PREFIX = "MARKETFORGE_VLLM_ABLATION="
BATCH_SIZE = 16
PREFIX_BATCH_SIZE = 8
PREFIX_TOKENS = 128
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_PROMPT = (0, 1, 2, 3)
BASE_EXPECTED = (198, 198, 504)


def standard_requests(
    count: int, *, label: str
) -> tuple[InferenceRequest, ...]:
    if count < 1 or not label:
        raise ValueError("standard request configuration is invalid")
    return tuple(
        InferenceRequest(f"{label}-{index:02d}", BASE_PROMPT)
        for index in range(count)
    )


def shared_prefix_requests(
    count: int,
    *,
    prefix_tokens: int,
    vocabulary_size: int,
) -> tuple[InferenceRequest, ...]:
    if count < 1 or prefix_tokens < 1 or vocabulary_size <= count + 1:
        raise ValueError("shared-prefix request configuration is invalid")
    prefix = tuple(
        ((index * 37 + 11) % (vocabulary_size - 1)) + 1
        for index in range(prefix_tokens)
    )
    return tuple(
        InferenceRequest(
            f"prefix-{index:02d}",
            prefix + (vocabulary_size - count + index,),
        )
        for index in range(count)
    )


def _generated(payload: dict[str, object]) -> list[list[int]]:
    requests = payload["requests"]
    if type(requests) is not list:
        raise RuntimeError("vLLM payload request list is malformed")
    return [
        list(request["generated_token_ids"])
        for request in requests
        if type(request) is dict
    ]


def _shutdown_engine(engine: object, *, timeout: float = 30.0) -> None:
    llm_engine = getattr(engine, "llm_engine", None)
    candidates = (
        engine,
        llm_engine,
        getattr(llm_engine, "engine_core", None),
        getattr(llm_engine, "engine_core_client", None),
    )
    for candidate in candidates:
        shutdown = getattr(candidate, "shutdown", None)
        if not callable(shutdown):
            continue
        try:
            shutdown(timeout=timeout)
        except TypeError:
            shutdown()
        return
    raise RuntimeError("vLLM engine exposes no shutdown path")


def _run_mode(
    *,
    mode: str,
    model_path: str,
    source: SourceIdentity,
) -> dict[str, object]:
    import torch
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlInit,
        nvmlShutdown,
    )

    if mode not in {"eager", "cuda_graph"}:
        raise ValueError("unknown vLLM execution mode")
    lock = json.loads(
        (PROJECT_ROOT / "tools/modal/vllm-lock.json").read_text(
            encoding="utf-8"
        )
    )
    execution = lock["execution"]
    model_lock = lock["model"]
    model = ModelIdentity(
        model_id=model_lock["id"],
        repository=model_lock["repository"],
        revision=model_lock["revision"],
        checkpoint_sha256=model_lock["checkpoint_sha256"],
        vocabulary_size=int(model_lock["vocabulary_size"]),
    )
    config = VllmConfig(
        max_output_tokens=3,
        max_model_len=int(execution["max_model_len"]),
        max_num_seqs=int(execution["max_num_seqs"]),
        gpu_memory_utilization=float(execution["gpu_memory_utilization"]),
        enforce_eager=mode == "eager",
        enable_prefix_caching=True,
    )
    properties = torch.cuda.get_device_properties(0)
    hardware = HardwareIdentity(
        device_name=torch.cuda.get_device_name(0),
        compute_capability=f"{properties.major}.{properties.minor}",
        cuda_version=str(torch.version.cuda),
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
        name=f"pr8-{mode}-gpu-memory-sampler",
        daemon=True,
    )
    memory_thread.start()
    engine: object | None = None
    try:
        engine, sampling_params_factory, load_seconds = create_engine(
            model,
            config,
            model_path=model_path,
        )
        run_arguments = {
            "engine": engine,
            "sampling_params_factory": sampling_params_factory,
            "model": model,
            "source": source,
            "hardware": hardware,
            "config": config,
            "model_load_seconds": load_seconds,
            "peak_gpu_memory": lambda: peak_gpu_memory_bytes[0],
        }
        single = run_greedy_batch(
            requests=standard_requests(1, label="single"),
            **run_arguments,
        ).to_payload()
        batch = run_greedy_batch(
            requests=standard_requests(BATCH_SIZE, label="batch"),
            **run_arguments,
        ).to_payload()
        runs: dict[str, object] = {
            "single": single,
            "batch": batch,
        }
        if mode == "cuda_graph":
            prefix_requests = shared_prefix_requests(
                PREFIX_BATCH_SIZE,
                prefix_tokens=PREFIX_TOKENS,
                vocabulary_size=model.vocabulary_size,
            )
            runs["prefix_cold"] = run_greedy_batch(
                requests=prefix_requests,
                **run_arguments,
            ).to_payload()
            runs["prefix_warm"] = run_greedy_batch(
                requests=prefix_requests,
                **run_arguments,
            ).to_payload()
    finally:
        try:
            if engine is not None:
                _shutdown_engine(engine)
                engine = None
                gc.collect()
        finally:
            memory_stop.set()
            memory_thread.join(timeout=1.0)
            nvmlShutdown()

    if memory_errors or peak_gpu_memory_bytes[0] <= 0:
        raise RuntimeError("device-wide NVML peak sampling failed")
    if _generated(single) != [list(BASE_EXPECTED)]:
        raise RuntimeError("single-request tokens do not match the oracle")
    if _generated(batch) != [list(BASE_EXPECTED)] * BATCH_SIZE:
        raise RuntimeError("batched tokens do not match the oracle")
    if mode == "cuda_graph" and _generated(
        runs["prefix_cold"]
    ) != _generated(runs["prefix_warm"]):
        raise RuntimeError("warm prefix-cache tokens changed")
    for payload in runs.values():
        validate_run_payload(payload)

    return {
        "schema_version": 1,
        "result": "pass",
        "mode": mode,
        "settings": {
            "batch_size": BATCH_SIZE,
            "prefix_batch_size": PREFIX_BATCH_SIZE,
            "prefix_tokens": PREFIX_TOKENS,
            "prefix_caching": True,
        },
        "runs": runs,
    }


def _parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "cuda_graph"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    values = _parse_arguments(arguments)
    if (
        re.fullmatch(r"[0-9a-f]{40}", values.source_commit) is None
        or re.fullmatch(
            r"[0-9a-f]{64}", values.source_bundle_sha256
        )
        is None
    ):
        raise ValueError("source identity is malformed")
    result = _run_mode(
        mode=values.mode,
        model_path=values.model_path,
        source=SourceIdentity(
            commit=values.source_commit,
            bundle_sha256=values.source_bundle_sha256,
        ),
    )
    print(
        RESULT_PREFIX
        + json.dumps(
            result,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
