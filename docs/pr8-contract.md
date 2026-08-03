# PR 8 contract: serving optimization core

Status: complete
Date: 2026-07-30
Base commit: `fbe16f4`
Accepted implementation commit:
`8756527e532c21cda1e9513421bf73cae163465a`

## Outcome

PR 8 collapses the remaining v1 serving work into one source-bound vertical
release. It preserves the native CUDA and vLLM work while replacing the longer
original PR train with four reviewable milestones:

1. deterministic continuous-batching scheduler;
2. paged KV allocation with immutable shared-prefix ownership;
3. CUDA restricted-token selection for the market-action grammar;
4. vLLM eager/graph and cold/warm-prefix batching ablations.

The release is complete only when all CPU state-machine properties pass under
sanitizers and one bounded Modal L4 gate builds/tests the exact clean commit,
runs native CUDA differential and performance tests, and records the vLLM
ablations using the locked SmolLM2 checkpoint.

## Scheduler invariants

- request IDs are unique and lifecycle transitions are explicit;
- a sequence can have at most one in-flight work item;
- prefill and decode batches share one deterministic round-robin admission
  policy;
- bounded batches eventually visit every runnable sequence;
- pause, resume, cancellation, maximum-output, and terminal completion are
  deterministic;
- invalid completion cannot partially mutate sequence accounting;
- CPU tests perform no model download, CUDA initialization, or network access.

## Paged KV and prefix invariants

- physical pages have one owner or one immutable-prefix reference set;
- reservations either commit atomically or return every page;
- page selection is deterministic and lowest-index first;
- shared-prefix pages are immutable while referenced;
- detach and sequence teardown release exactly one reference;
- metrics distinguish payload tokens, reserved capacity, fragmentation, and
  shared physical pages.

## CUDA restricted-token invariants

- selection accepts an explicit allowed-token set per sequence;
- FP16 logits use FP32 comparison with lowest-token-ID tie behavior;
- NaN never wins and an all-NaN allowed set has a defined fallback;
- the GPU result is differential-tested against the action-DFA CPU oracle;
- model-shaped batches report CUDA-event latency and logical bandwidth.

## vLLM ablation matrix

The same immutable model revision and token-ID prompt family must compare:

- eager single-request decode;
- graph-enabled single-request decode;
- eager batched decode;
- graph-enabled batched decode;
- cold common-prefix batch;
- warm prefix-cache batch.

Each record includes exact input/output token IDs, batch size, prefix length,
cache mode, execution mode, latency, throughput, source commit, source-bundle
SHA-256, package versions, and GPU identity. Performance claims are descriptive
for the one L4 run and do not replace exact token parity.

## Budget

Every combined Modal attempt is limited to one L4, 15 minutes, and $0.239364.
The project stops submitting work before the user account would fall below the
approved $6 reserve.

## Acceptance

The combined gate passed on one NVIDIA L4 with CUDA 12.9.41. It built all 76
steps, passed 6/6 CTests, reproduced `[198, 198, 504]` through both native CUDA
and vLLM, measured the complete grammar-restricted CUDA kernel, and accepted
the eager/graph and cold/warm-prefix ablation matrix. See
`docs/pr8-modal-result.json`.
