#!/usr/bin/env python3
"""Export a deterministic tiny Llama decoder layer from pinned PyTorch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import torch
import transformers
from safetensors.torch import save_file
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    apply_rotary_pos_emb,
    repeat_kv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "golden"
    / "smollm2-tiny-layer-f32.safetensors"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_suffix(".json")


def smollm_provenance() -> tuple[str, str]:
    lock_path = PROJECT_ROOT / "models" / "model-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for model in lock["models"]:
        if model["id"] == "smollm2-135m":
            return str(model["repository"]), str(model["revision"])
    raise ValueError("model lock does not contain smollm2-135m")


def canonicalize_safetensors(path: Path) -> None:
    """Sort the JSON header so safetensors metadata order is reproducible."""
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


def deterministic_parameter(name: str, parameter: torch.Tensor) -> torch.Tensor:
    index = torch.arange(parameter.numel(), dtype=torch.float32).reshape(
        parameter.shape
    )
    if name.endswith("layernorm.weight"):
        return 1.0 + 0.025 * torch.sin(index * 0.7 + len(name))
    phase = (sum(name.encode("utf-8")) % 29) / 11.0
    return 0.16 * torch.sin(index * 0.37 + phase)


def build_fixture() -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        rms_norm_eps=1.0e-5,
        rope_theta=10_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        pretraining_tp=1,
    )
    config._attn_implementation = "eager"
    layer = LlamaDecoderLayer(config, layer_idx=0).float().eval()
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            parameter.copy_(deterministic_parameter(name, parameter))

    batch = 2
    tokens = 3
    hidden = config.hidden_size
    input_index = torch.arange(
        batch * tokens * hidden, dtype=torch.float32
    ).reshape(batch, tokens, hidden)
    input_hidden = 0.35 * torch.sin(input_index * 0.23) + 0.08 * torch.cos(
        input_index * 0.11
    )
    positions = torch.arange(tokens, dtype=torch.int64).repeat(batch, 1)
    contexts = torch.full((batch,), tokens, dtype=torch.int32)

    causal_mask = torch.full(
        (batch, 1, tokens, tokens),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)

    with torch.no_grad():
        input_norm = layer.input_layernorm(input_hidden)
        query = layer.self_attn.q_proj(input_norm)
        key = layer.self_attn.k_proj(input_norm)
        value = layer.self_attn.v_proj(input_norm)

        head_dim = config.hidden_size // config.num_attention_heads
        query = query.view(
            batch, tokens, config.num_attention_heads, head_dim
        ).transpose(1, 2)
        key = key.view(
            batch, tokens, config.num_key_value_heads, head_dim
        ).transpose(1, 2)
        value = value.view(
            batch, tokens, config.num_key_value_heads, head_dim
        ).transpose(1, 2)

        cosine, sine = layer.self_attn.rotary_emb(value, positions)
        query_rope, key_rope = apply_rotary_pos_emb(
            query, key, cosine, sine
        )
        repeated_key = repeat_kv(
            key_rope,
            config.num_attention_heads // config.num_key_value_heads,
        )
        repeated_value = repeat_kv(
            value,
            config.num_attention_heads // config.num_key_value_heads,
        )
        scores = torch.matmul(
            query_rope, repeated_key.transpose(2, 3)
        ) / (head_dim**0.5)
        probabilities = torch.softmax(
            scores + causal_mask, dim=-1, dtype=torch.float32
        )
        attention_heads = torch.matmul(probabilities, repeated_value)
        attention_merged = (
            attention_heads.transpose(1, 2)
            .contiguous()
            .reshape(batch, tokens, hidden)
        )
        attention_projected = layer.self_attn.o_proj(attention_merged)
        after_attention = input_hidden + attention_projected
        post_attention_norm = layer.post_attention_layernorm(
            after_attention
        )
        gate = layer.mlp.gate_proj(post_attention_norm)
        up = layer.mlp.up_proj(post_attention_norm)
        swiglu = torch.nn.functional.silu(gate) * up
        mlp_down = layer.mlp.down_proj(swiglu)
        after_mlp = after_attention + mlp_down

        module_output = layer(
            input_hidden,
            attention_mask=causal_mask,
            position_ids=positions,
            output_attentions=False,
            use_cache=False,
        )[0]
        torch.testing.assert_close(
            after_mlp, module_output, rtol=0.0, atol=2.0e-7
        )

    tensors = {
        "input.hidden": input_hidden.contiguous(),
        "position_ids": positions.to(torch.int32).contiguous(),
        "context_lengths": contexts.contiguous(),
        "weights.input_norm": layer.input_layernorm.weight.detach().contiguous(),
        "weights.q_proj": layer.self_attn.q_proj.weight.detach().contiguous(),
        "weights.k_proj": layer.self_attn.k_proj.weight.detach().contiguous(),
        "weights.v_proj": layer.self_attn.v_proj.weight.detach().contiguous(),
        "weights.o_proj": layer.self_attn.o_proj.weight.detach().contiguous(),
        "weights.post_attention_norm": (
            layer.post_attention_layernorm.weight.detach().contiguous()
        ),
        "weights.gate_proj": layer.mlp.gate_proj.weight.detach().contiguous(),
        "weights.up_proj": layer.mlp.up_proj.weight.detach().contiguous(),
        "weights.down_proj": layer.mlp.down_proj.weight.detach().contiguous(),
        "expected.input_norm": input_norm.contiguous(),
        "expected.query_rope": query_rope.transpose(1, 2).contiguous(),
        "expected.key_rope": key_rope.transpose(1, 2).contiguous(),
        "expected.value": value.transpose(1, 2).contiguous(),
        "expected.attention_logits": scores.contiguous(),
        "expected.attention_probabilities": probabilities.contiguous(),
        "expected.attention_heads": (
            attention_heads.transpose(1, 2).contiguous()
        ),
        "expected.attention_projected": attention_projected.contiguous(),
        "expected.after_attention": after_attention.contiguous(),
        "expected.post_attention_norm": post_attention_norm.contiguous(),
        "expected.gate": gate.contiguous(),
        "expected.up": up.contiguous(),
        "expected.swiglu": swiglu.contiguous(),
        "expected.mlp_down": mlp_down.contiguous(),
        "expected.after_mlp": after_mlp.contiguous(),
        "expected.key_cache": key_rope.transpose(1, 2).contiguous().clone(),
        "expected.value_cache": value.transpose(1, 2).contiguous().clone(),
    }
    model_repository, model_revision = smollm_provenance()
    metadata = {
        "fixture": "marketforge-pr3-tiny-smollm2-layer",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_repository": model_repository,
        "model_revision": model_revision,
        "layout": "hidden=[batch,tokens,hidden], qkv=[batch,tokens,heads,head_dim]",
        "rope": "Llama half rotation",
    }
    manifest = {
        "schema_version": 1,
        "generator": "tools/golden/export_pr3_fixture.py",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_repository": model_repository,
        "model_revision": model_revision,
        "config": {
            "batch": batch,
            "tokens": tokens,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "query_heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
            "head_dim": head_dim,
            "rms_norm_epsilon": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "max_positions": config.max_position_embeddings,
        },
        "tolerances": {
            "rms_norm_absolute": 2.0e-6,
            "rope_absolute": 2.0e-6,
            "attention_absolute": 5.0e-6,
            "layer_absolute": 1.0e-5,
        },
        "metadata": metadata,
    }
    return tensors, manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    tensors, manifest = build_fixture()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, arguments.output, metadata=manifest["metadata"])
    canonicalize_safetensors(arguments.output)
    manifest["sha256"] = hashlib.sha256(
        arguments.output.read_bytes()
    ).hexdigest()
    manifest["size"] = arguments.output.stat().st_size
    arguments.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {arguments.output} ({manifest['size']} bytes, "
        f"sha256 {manifest['sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
