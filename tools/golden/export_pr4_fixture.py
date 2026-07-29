#!/usr/bin/env python3
"""Export full SmolLM2 FP32 logits from explicit pretokenized IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import torch
import transformers
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "golden"
    / "smollm2-pr4-greedy-f32.safetensors"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_suffix(".json")
PROMPT_TOKEN_IDS = [0, 1, 2, 3]
GENERATED_STEPS = 3
EXPECTED_TORCH = "2.3.1"
EXPECTED_TRANSFORMERS = "4.40.1"
EXPECTED_SAFETENSORS = "0.4.3"


def canonicalize_safetensors(path: Path) -> None:
    contents = path.read_bytes()
    if len(contents) < 8:
        raise ValueError("safetensors output is missing its header length")
    original_size = struct.unpack("<Q", contents[:8])[0]
    data_start = 8 + original_size
    if data_start > len(contents):
        raise ValueError("safetensors output has a truncated header")
    header = json.loads(contents[8:data_start])
    encoded = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    padding = (-len(encoded)) % 8
    canonical_header = encoded + b" " * padding
    path.write_bytes(
        struct.pack("<Q", len(canonical_header))
        + canonical_header
        + contents[data_start:]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locked_smollm() -> dict[str, object]:
    lock = json.loads(
        (PROJECT_ROOT / "models" / "model-lock.json").read_text(
            encoding="utf-8"
        )
    )
    return next(
        model for model in lock["models"] if model["id"] == "smollm2-135m"
    )


def require_versions() -> None:
    import safetensors

    actual = {
        "torch": torch.__version__.split("+", maxsplit=1)[0],
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
    }
    expected = {
        "torch": EXPECTED_TORCH,
        "transformers": EXPECTED_TRANSFORMERS,
        "safetensors": EXPECTED_SAFETENSORS,
    }
    if actual != expected:
        raise RuntimeError(
            f"oracle package versions do not match: {actual} != {expected}"
        )


def verify_model_directory(model_dir: Path) -> dict[str, object]:
    model = locked_smollm()
    locked_files = {
        str(entry["path"]): entry for entry in model["files"]
    }
    for name in ("config.json", "model.safetensors"):
        path = model_dir / name
        entry = locked_files[name]
        if not path.is_file():
            raise FileNotFoundError(f"missing locked model file: {path}")
        if path.stat().st_size != int(entry["size"]):
            raise RuntimeError(f"locked size mismatch: {path}")
        if sha256(path) != str(entry["sha256"]):
            raise RuntimeError(f"locked SHA-256 mismatch: {path}")
    return model


def build_fixture(
    model_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    require_versions()
    model_lock = verify_model_directory(model_dir)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    config = LlamaConfig.from_json_file(model_dir / "config.json")
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).float().eval()
    state = load_file(model_dir / "model.safetensors", device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["lm_head.weight"] or unexpected:
        raise RuntimeError(
            f"unexpected checkpoint keys: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.tie_weights()
    if (
        model.lm_head.weight.data_ptr()
        != model.model.embed_tokens.weight.data_ptr()
    ):
        raise RuntimeError("tied output head does not alias the embedding")
    del state

    logits_by_step: list[torch.Tensor] = []
    generated: list[int] = []
    margins: list[float] = []
    current = torch.tensor([PROMPT_TOKEN_IDS], dtype=torch.int64)
    past_key_values = None
    with torch.inference_mode():
        for _ in range(GENERATED_STEPS):
            output = model(
                input_ids=current,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            logits = output.logits[0, -1].to(torch.float32).contiguous()
            top = torch.topk(logits, k=2)
            token = int(top.indices[0])
            generated.append(token)
            margins.append(float(top.values[0] - top.values[1]))
            logits_by_step.append(logits)
            current = torch.tensor([[token]], dtype=torch.int64)

    if min(margins) < 0.5:
        raise RuntimeError(
            f"fixture is not safely margin-qualified: margins={margins}"
        )

    tensors = {
        "input.token_ids": torch.tensor(
            PROMPT_TOKEN_IDS, dtype=torch.int32
        ),
        "expected.tokens": torch.tensor(generated, dtype=torch.int32),
        "expected.logits": torch.stack(logits_by_step).contiguous(),
        "expected.margins": torch.tensor(margins, dtype=torch.float32),
        "tolerance.logits_absolute": torch.tensor(
            [1.0e-4], dtype=torch.float32
        ),
        "tolerance.logits_relative": torch.tensor(
            [1.0e-5], dtype=torch.float32
        ),
    }
    manifest = {
        "schema_version": 1,
        "generator": "tools/golden/export_pr4_fixture.py",
        "fixture": "marketforge-pr4-smollm2-greedy-f32",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": EXPECTED_SAFETENSORS,
        "model_repository": model_lock["repository"],
        "model_revision": model_lock["revision"],
        "checkpoint_size": model_lock["files"][1]["size"],
        "checkpoint_sha256": model_lock["files"][1]["sha256"],
        "computation": "BF16 checkpoint materialized to FP32; eager CPU forward",
        "tokenization": "none; input is an explicit pretokenized integer fixture",
        "prompt_token_ids": PROMPT_TOKEN_IDS,
        "expected_tokens": generated,
        "logit_margins": margins,
        "tolerances": {
            "logits_absolute": 1.0e-4,
            "logits_relative": 1.0e-5,
            "greedy_tokens": "exact",
            "minimum_fixture_margin": 0.5,
        },
    }
    return tensors, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()

    model_dir = arguments.model_dir.expanduser().resolve()
    output = arguments.output.resolve()
    manifest_path = arguments.manifest.resolve()
    tensors, manifest = build_fixture(model_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(output),
        metadata={
            "fixture": str(manifest["fixture"]),
            "model_repository": str(manifest["model_repository"]),
            "model_revision": str(manifest["model_revision"]),
            "torch_version": str(manifest["torch_version"]),
            "transformers_version": str(manifest["transformers_version"]),
        },
    )
    canonicalize_safetensors(output)
    manifest["fixture_size"] = output.stat().st_size
    manifest["fixture_sha256"] = sha256(output)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {output} ({output.stat().st_size} bytes, "
        f"sha256={manifest['fixture_sha256']})"
    )
    print(
        f"tokens={manifest['expected_tokens']} "
        f"margins={manifest['logit_margins']}"
    )


if __name__ == "__main__":
    main()
