# ADR 0001: release-1 runtime scope

Status: accepted
Date: 2026-07-28

## Decision

The portable runtime accepts contiguous token and tensor data. It is not a
general tensor framework and does not tokenize arbitrary text.

Release 1 supports:

- a fixed-rank contiguous tensor view;
- checked byte/shape arithmetic;
- SmolLM2/Llama as the first execution architecture;
- Qwen2 as a later explicit architecture;
- FP32 CPU reference and FP16 CUDA execution;
- pretokenized inputs and a compiled finite action grammar.

Release 1 excludes:

- arbitrary strides or broadcasting;
- autograd and training;
- arbitrary Hugging Face model plugins;
- native tokenizer implementation;
- CUDA discovery in the portable build;
- model downloads during default tests.

## Consequences

The core remains small enough to audit. Python tooling may create token fixtures
and action DFAs, but the inference hot path is native C++/CUDA. Supporting a new
architecture requires an explicit weight-binding and operator contract rather
than accepting unchecked configuration JSON.
