"""Pinned vLLM adapter for token-ID-only MarketForge inference."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Protocol, Sequence

from tools.inference.contract import (
    BackendIdentity,
    GenerationSettings,
    HardwareIdentity,
    InferenceOutput,
    InferenceRequest,
    InferenceRun,
    ModelIdentity,
    RunMetrics,
    SourceIdentity,
)


VLLM_VERSION = "0.25.1"


class VllmAdapterError(RuntimeError):
    """Raised when vLLM violates the adapter's narrow generation contract."""


class Completion(Protocol):
    token_ids: Sequence[int]
    finish_reason: str | None


class RequestResult(Protocol):
    outputs: Sequence[Completion]


class Engine(Protocol):
    def generate(
        self,
        prompts: Sequence[list[int]],
        sampling_params: object,
        *,
        use_tqdm: bool,
    ) -> Sequence[RequestResult]: ...


SamplingParamsFactory = Callable[..., object]


@dataclass(frozen=True)
class VllmConfig:
    dtype: str = "float16"
    seed: int = 0
    max_output_tokens: int = 3
    max_model_len: int = 512
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.35
    enforce_eager: bool = True
    enable_prefix_caching: bool = False


def create_engine(
    model: ModelIdentity,
    config: VllmConfig,
    *,
    model_path: str | None = None,
) -> tuple[Engine, SamplingParamsFactory, float]:
    started = time.perf_counter()
    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise VllmAdapterError(
            "vLLM is unavailable; run this backend in the pinned Modal image"
        ) from error
    if vllm.__version__ != VLLM_VERSION:
        raise VllmAdapterError(
            f"expected vLLM {VLLM_VERSION}, found {vllm.__version__}"
        )
    engine = LLM(
        model=model_path or model.repository,
        revision=None if model_path is not None else model.revision,
        tokenizer_revision=None if model_path is not None else model.revision,
        trust_remote_code=False,
        dtype=config.dtype,
        seed=config.seed,
        tensor_parallel_size=1,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        gpu_memory_utilization=config.gpu_memory_utilization,
        enforce_eager=config.enforce_eager,
        enable_prefix_caching=config.enable_prefix_caching,
        skip_tokenizer_init=True,
        generation_config="vllm",
        disable_log_stats=True,
    )
    return engine, SamplingParams, time.perf_counter() - started


def run_greedy_batch(
    *,
    engine: Engine,
    sampling_params_factory: SamplingParamsFactory,
    model: ModelIdentity,
    source: SourceIdentity,
    hardware: HardwareIdentity,
    requests: Sequence[InferenceRequest],
    config: VllmConfig,
    model_load_seconds: float,
    peak_gpu_memory: Callable[[], int],
    clock: Callable[[], float] = time.perf_counter,
) -> InferenceRun:
    if not requests:
        raise VllmAdapterError("at least one request is required")
    sampling = sampling_params_factory(
        temperature=0.0,
        seed=config.seed,
        max_tokens=config.max_output_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    started = clock()
    results = engine.generate(
        [list(request.prompt_token_ids) for request in requests],
        sampling,
        use_tqdm=False,
    )
    elapsed = clock() - started
    if elapsed <= 0.0:
        raise VllmAdapterError("inference duration must be positive")
    if len(results) != len(requests):
        raise VllmAdapterError("vLLM changed the request cardinality")

    outputs: list[InferenceOutput] = []
    for request, result in zip(requests, results, strict=True):
        if len(result.outputs) != 1:
            raise VllmAdapterError("vLLM returned a non-single completion")
        completion = result.outputs[0]
        if any(type(token) is not int for token in completion.token_ids):
            raise VllmAdapterError("vLLM returned a non-integer token ID")
        generated = tuple(completion.token_ids)
        if len(generated) > config.max_output_tokens:
            raise VllmAdapterError("vLLM exceeded the output-token cap")
        finish_reason = completion.finish_reason
        if finish_reason is None:
            finish_reason = "length"
        if finish_reason not in {"length", "eos", "stop"}:
            raise VllmAdapterError(
                f"unsupported vLLM finish reason: {finish_reason}"
            )
        outputs.append(
            InferenceOutput(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                generated_token_ids=generated,
                finish_reason=finish_reason,
            )
        )

    input_tokens = sum(len(item.prompt_token_ids) for item in outputs)
    output_tokens = sum(len(item.generated_token_ids) for item in outputs)
    return InferenceRun(
        source=source,
        backend=BackendIdentity(
            name="vllm",
            version=VLLM_VERSION,
            mode="eager" if config.enforce_eager else "cuda_graph",
        ),
        model=model,
        hardware=hardware,
        generation=GenerationSettings(
            dtype=config.dtype,
            temperature=0.0,
            seed=config.seed,
            max_output_tokens=config.max_output_tokens,
        ),
        prefix_caching=config.enable_prefix_caching,
        cuda_graphs=not config.enforce_eager,
        structured_output=False,
        outputs=tuple(outputs),
        metrics=RunMetrics(
            model_load_seconds=float(model_load_seconds),
            inference_seconds=float(elapsed),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            requests_per_second=len(outputs) / elapsed,
            output_tokens_per_second=output_tokens / elapsed,
            peak_gpu_memory_bytes=peak_gpu_memory(),
        ),
    )
