"""Strict JSON contract shared by CPU, native CUDA, and vLLM runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Sequence


SCHEMA_VERSION = 1
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")
_COMPUTE_CAPABILITY = re.compile(r"(?:cpu|[0-9]+\.[0-9]+)")
_BACKEND_MODES = {"cpu_reference", "eager", "cuda_graph", "ordinary"}
_DTYPES = {"float32", "float16", "bfloat16"}
_FINISH_REASONS = {"length", "eos", "stop"}


class ContractError(ValueError):
    """Raised when an inference artifact violates the frozen schema."""


@dataclass(frozen=True)
class SourceIdentity:
    commit: str
    bundle_sha256: str


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    repository: str
    revision: str
    checkpoint_sha256: str
    vocabulary_size: int


@dataclass(frozen=True)
class BackendIdentity:
    name: str
    version: str
    mode: str


@dataclass(frozen=True)
class HardwareIdentity:
    device_name: str
    compute_capability: str
    cuda_version: str


@dataclass(frozen=True)
class GenerationSettings:
    dtype: str
    temperature: float
    seed: int
    max_output_tokens: int


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class InferenceOutput:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    finish_reason: str


@dataclass(frozen=True)
class RunMetrics:
    model_load_seconds: float
    inference_seconds: float
    input_tokens: int
    output_tokens: int
    requests_per_second: float
    output_tokens_per_second: float
    peak_gpu_memory_bytes: int


@dataclass(frozen=True)
class InferenceRun:
    source: SourceIdentity
    backend: BackendIdentity
    model: ModelIdentity
    hardware: HardwareIdentity
    generation: GenerationSettings
    prefix_caching: bool
    cuda_graphs: bool
    structured_output: bool
    outputs: tuple[InferenceOutput, ...]
    metrics: RunMetrics

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "result": "pass",
            "source": {
                "commit": self.source.commit,
                "bundle_sha256": self.source.bundle_sha256,
            },
            "backend": {
                "name": self.backend.name,
                "version": self.backend.version,
                "mode": self.backend.mode,
            },
            "model": {
                "id": self.model.model_id,
                "repository": self.model.repository,
                "revision": self.model.revision,
                "checkpoint_sha256": self.model.checkpoint_sha256,
                "vocabulary_size": self.model.vocabulary_size,
            },
            "hardware": {
                "device_name": self.hardware.device_name,
                "compute_capability": self.hardware.compute_capability,
                "cuda_version": self.hardware.cuda_version,
            },
            "generation": {
                "dtype": self.generation.dtype,
                "temperature": self.generation.temperature,
                "seed": self.generation.seed,
                "max_output_tokens": self.generation.max_output_tokens,
            },
            "features": {
                "prefix_caching": self.prefix_caching,
                "cuda_graphs": self.cuda_graphs,
                "structured_output": self.structured_output,
            },
            "requests": [
                {
                    "request_id": output.request_id,
                    "prompt_token_ids": list(output.prompt_token_ids),
                    "generated_token_ids": list(
                        output.generated_token_ids
                    ),
                    "finish_reason": output.finish_reason,
                }
                for output in self.outputs
            ],
            "metrics": {
                "model_load_seconds": self.metrics.model_load_seconds,
                "inference_seconds": self.metrics.inference_seconds,
                "input_tokens": self.metrics.input_tokens,
                "output_tokens": self.metrics.output_tokens,
                "requests_per_second": self.metrics.requests_per_second,
                "output_tokens_per_second": (
                    self.metrics.output_tokens_per_second
                ),
                "peak_gpu_memory_bytes": (
                    self.metrics.peak_gpu_memory_bytes
                ),
            },
        }
        validate_run_payload(payload)
        return payload

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _exact_keys(
    value: object, expected: set[str], location: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise ContractError(f"{location} must be an object")
    if set(value) != expected:
        raise ContractError(f"{location} fields do not match schema")
    return value


def _string(value: object, location: str) -> str:
    if type(value) is not str or not value:
        raise ContractError(f"{location} must be a non-empty string")
    return value


def _integer(
    value: object, location: str, *, minimum: int = 0
) -> int:
    if type(value) is not int or value < minimum:
        raise ContractError(f"{location} must be an integer >= {minimum}")
    return value


def _finite_float(
    value: object, location: str, *, strictly_positive: bool = False
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ContractError(f"{location} must be a finite float")
    if strictly_positive and value <= 0.0:
        raise ContractError(f"{location} must be positive")
    if not strictly_positive and value < 0.0:
        raise ContractError(f"{location} must be non-negative")
    return value


def _boolean(value: object, location: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{location} must be a boolean")
    return value


def _tokens(
    value: object, location: str, vocabulary_size: int, *, empty: bool
) -> list[int]:
    if type(value) is not list or (not empty and not value):
        raise ContractError(f"{location} must be a token list")
    result: list[int] = []
    for index, token in enumerate(value):
        token_id = _integer(token, f"{location}[{index}]")
        if token_id >= vocabulary_size:
            raise ContractError(f"{location}[{index}] exceeds vocabulary")
        result.append(token_id)
    return result


def validate_run_payload(payload: object) -> None:
    root = _exact_keys(
        payload,
        {
            "schema_version",
            "result",
            "source",
            "backend",
            "model",
            "hardware",
            "generation",
            "features",
            "requests",
            "metrics",
        },
        "run",
    )
    if (
        type(root["schema_version"]) is not int
        or root["schema_version"] != SCHEMA_VERSION
        or root["result"] != "pass"
    ):
        raise ContractError("run version or result is invalid")

    source = _exact_keys(
        root["source"], {"commit", "bundle_sha256"}, "source"
    )
    commit = _string(source["commit"], "source.commit")
    bundle_sha256 = _string(
        source["bundle_sha256"], "source.bundle_sha256"
    )
    if _SHA40.fullmatch(commit) is None:
        raise ContractError("source.commit must be a lowercase Git object ID")
    if _SHA64.fullmatch(bundle_sha256) is None:
        raise ContractError("source.bundle_sha256 must be lowercase SHA-256")

    backend = _exact_keys(
        root["backend"], {"name", "version", "mode"}, "backend"
    )
    _string(backend["name"], "backend.name")
    _string(backend["version"], "backend.version")
    if backend["mode"] not in _BACKEND_MODES:
        raise ContractError("backend.mode is unsupported")

    model = _exact_keys(
        root["model"],
        {
            "id",
            "repository",
            "revision",
            "checkpoint_sha256",
            "vocabulary_size",
        },
        "model",
    )
    _string(model["id"], "model.id")
    _string(model["repository"], "model.repository")
    revision = _string(model["revision"], "model.revision")
    checkpoint = _string(
        model["checkpoint_sha256"], "model.checkpoint_sha256"
    )
    if _SHA40.fullmatch(revision) is None:
        raise ContractError("model.revision must be a lowercase commit")
    if _SHA64.fullmatch(checkpoint) is None:
        raise ContractError("model.checkpoint_sha256 must be lowercase SHA-256")
    vocabulary_size = _integer(
        model["vocabulary_size"], "model.vocabulary_size", minimum=1
    )

    hardware = _exact_keys(
        root["hardware"],
        {"device_name", "compute_capability", "cuda_version"},
        "hardware",
    )
    _string(hardware["device_name"], "hardware.device_name")
    capability = _string(
        hardware["compute_capability"], "hardware.compute_capability"
    )
    if _COMPUTE_CAPABILITY.fullmatch(capability) is None:
        raise ContractError("hardware.compute_capability is malformed")
    _string(hardware["cuda_version"], "hardware.cuda_version")

    generation = _exact_keys(
        root["generation"],
        {"dtype", "temperature", "seed", "max_output_tokens"},
        "generation",
    )
    if generation["dtype"] not in _DTYPES:
        raise ContractError("generation.dtype is unsupported")
    temperature = _finite_float(
        generation["temperature"], "generation.temperature"
    )
    if temperature != 0.0:
        raise ContractError("release-1 inference must be greedy")
    _integer(generation["seed"], "generation.seed")
    max_output_tokens = _integer(
        generation["max_output_tokens"],
        "generation.max_output_tokens",
        minimum=1,
    )

    features = _exact_keys(
        root["features"],
        {"prefix_caching", "cuda_graphs", "structured_output"},
        "features",
    )
    for feature in features:
        _boolean(features[feature], f"features.{feature}")

    requests = root["requests"]
    if type(requests) is not list or not requests:
        raise ContractError("requests must be a non-empty list")
    request_ids: set[str] = set()
    input_tokens = 0
    output_tokens = 0
    for index, raw_request in enumerate(requests):
        request = _exact_keys(
            raw_request,
            {
                "request_id",
                "prompt_token_ids",
                "generated_token_ids",
                "finish_reason",
            },
            f"requests[{index}]",
        )
        request_id = _string(
            request["request_id"], f"requests[{index}].request_id"
        )
        if request_id in request_ids:
            raise ContractError("request IDs must be unique")
        request_ids.add(request_id)
        prompt = _tokens(
            request["prompt_token_ids"],
            f"requests[{index}].prompt_token_ids",
            vocabulary_size,
            empty=False,
        )
        generated = _tokens(
            request["generated_token_ids"],
            f"requests[{index}].generated_token_ids",
            vocabulary_size,
            empty=True,
        )
        if len(generated) > max_output_tokens:
            raise ContractError("generated token count exceeds configured cap")
        if request["finish_reason"] not in _FINISH_REASONS:
            raise ContractError("request finish reason is unsupported")
        input_tokens += len(prompt)
        output_tokens += len(generated)

    metrics = _exact_keys(
        root["metrics"],
        {
            "model_load_seconds",
            "inference_seconds",
            "input_tokens",
            "output_tokens",
            "requests_per_second",
            "output_tokens_per_second",
            "peak_gpu_memory_bytes",
        },
        "metrics",
    )
    _finite_float(metrics["model_load_seconds"], "metrics.model_load_seconds")
    elapsed = _finite_float(
        metrics["inference_seconds"],
        "metrics.inference_seconds",
        strictly_positive=True,
    )
    if _integer(metrics["input_tokens"], "metrics.input_tokens") != input_tokens:
        raise ContractError("metrics.input_tokens does not match requests")
    if (
        _integer(metrics["output_tokens"], "metrics.output_tokens")
        != output_tokens
    ):
        raise ContractError("metrics.output_tokens does not match requests")
    request_rate = _finite_float(
        metrics["requests_per_second"],
        "metrics.requests_per_second",
        strictly_positive=True,
    )
    output_rate = _finite_float(
        metrics["output_tokens_per_second"],
        "metrics.output_tokens_per_second",
    )
    _integer(
        metrics["peak_gpu_memory_bytes"],
        "metrics.peak_gpu_memory_bytes",
    )
    if not math.isclose(
        request_rate, len(requests) / elapsed, rel_tol=1.0e-9
    ):
        raise ContractError("request rate is inconsistent")
    if not math.isclose(
        output_rate, output_tokens / elapsed, rel_tol=1.0e-9
    ):
        raise ContractError("output-token rate is inconsistent")


def load_run_json(source: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ContractError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(name: str) -> object:
        raise ContractError(f"non-finite JSON constant: {name}")

    value = json.loads(
        source,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    validate_run_payload(value)
    return value
