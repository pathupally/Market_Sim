"""Bounded vLLM/SmolLM2 L4 conformance run for PR 7."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

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


vllm_image = (
    modal.Image.from_registry(
        LOCK["base_image"]["reference"],
        add_python=LOCK["python"],
    )
    .entrypoint([])
    .uv_pip_install(f"vllm=={LOCK['packages']['vllm']}")
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
    "marketforge-pr7-vllm-smoke",
    tags={"project": "marketforge", "purpose": "pr7-vllm-conformance"},
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
def vllm_smoke(
    candidate_commit: str,
    source_bundle_sha256: str,
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
        raise RuntimeError("PR 7 vLLM conformance requires compute 8.9")
    torch.cuda.reset_peak_memory_stats()

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
        peak_gpu_memory=lambda: int(torch.cuda.max_memory_allocated()),
    )
    payload = run.to_payload()
    if payload["requests"][0]["generated_token_ids"] != [198, 198, 504]:
        raise RuntimeError("vLLM tokens do not match the PR 4 oracle")
    validate_run_payload(payload)
    model_cache.commit()
    return payload


@app.local_entrypoint()
def main(month_to_date_usd: str) -> None:
    month_to_date = parse_cost(month_to_date_usd)
    require_project_headroom(
        month_to_date_usd=month_to_date,
        planned_cost_usd=MAXIMUM_COST_USD,
    )
    bundle = create_source_bundle(PROJECT_ROOT)
    result = vllm_smoke.remote(bundle.commit, bundle.sha256)
    validate_run_payload(result)
    if (
        result["source"]["commit"] != bundle.commit
        or result["source"]["bundle_sha256"] != bundle.sha256
    ):
        raise RuntimeError("remote vLLM evidence is not source-bound")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
