# PR 7 contract: vLLM inference and native CUDA model core

Status: implementation started
Date: 2026-07-29
Base commit: `bf190b5`

## Outcome

PR 7 combines the old PR 7–9 sequence into one vertical release train. It must
produce two independently useful inference paths:

1. a pinned vLLM baseline that accepts pretokenized SmolLM2 requests; and
2. a native C++/CUDA path built from cuBLAS plus focused custom kernels.

The vLLM path is not the numerical oracle and the native path does not copy
vLLM internals. Both emit the strict artifact schema in
`tools/inference/contract.py` and compare against the existing PyTorch/CPU
fixtures.

## Reviewable milestone commits

1. Backend-neutral inference record and bounded vLLM L4 smoke.
2. FP16/BF16 device weight materialization and explicit-stream cuBLAS adapter.
3. RoPE, SwiGLU, KV-write, and greedy-selection CUDA kernels.
4. Contiguous-KV native SmolLM2 prefill/decode with three-backend comparison.

Each milestone must keep local CPU tests green. GPU milestones require a
source-bound Modal result before their performance claims are accepted.

## Frozen vLLM lane

- vLLM: `0.25.1`;
- model: `HuggingFaceTB/SmolLM2-135M`;
- model revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- weight file: 269,060,552 bytes;
- input: token IDs only;
- output: greedy token IDs only;
- dtype: FP16 baseline;
- prompt: `[0, 1, 2, 3]`;
- expected output: `[198, 198, 504]`;
- GPU: one Modal L4 container;
- smoke timeout: 900 seconds;
- maximum compute cost: $0.239364.

The first smoke enforces eager execution. CUDA-graph and prefix-cache modes are
PR 8 ablations and must use the same artifact schema.

## Milestone-1 acceptance

- the contract rejects duplicate JSON keys, non-finite values, unknown fields,
  moving revisions, out-of-vocabulary tokens, inconsistent counts, and
  inconsistent rates;
- every result identifies the clean Git commit and immutable source-bundle hash;
- the vLLM adapter never tokenizes or detokenizes;
- one request produces one completion and cannot exceed the output cap;
- the Modal image, vLLM package, model revision, checkpoint hash, GPU type,
  concurrency, timeout, and cost ceiling are pinned;
- the cached checkpoint is size- and SHA-256-verified before vLLM loads it;
- default local tests import no vLLM, torch, or CUDA runtime;
- no network or GPU job runs during default tests.
