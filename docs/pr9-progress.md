# PR 9 progress log

Last updated: 2026-07-31
Branch: `codex/pr-009-constrained-output-head`
Base: `1a9d65e`

## Milestone 1 — generic finite-choice grammar

Status: locally complete

Delivered:

- application-neutral token sequences mapped to opaque choice IDs;
- deterministic sorted transitions and terminal decoding;
- prefix-free, duplicate, vocabulary, and resource-limit validation;
- focused unit tests under Debug and ASan/UBSan.

## Milestone 2 — fused restricted CUDA output head

Status: locally complete; GPU validation pending

Delivered:

- FP16 hidden/embedding input with FP32 dot-product accumulation;
- one fused projection/argmax launch without full-vocabulary logits;
- deterministic tie, NaN, fallback, invalid-row, size, alias, stream, and
  arithmetic behavior;
- native SmolLM2 `prefill_restricted` and `decode_restricted` entrypoints;
- real-checkpoint conformance that repeats the full decode through restricted
  legal sets and requires exact `[198, 198, 504]` parity;
- a 12-cell L4 benchmark over three batch sizes and four legal-set sizes.

Local evidence:

- repository-root Python discovery: 114 passed, 1 expected skip;
- Apple Clang Debug: 4/4 CTests passed;
- Apple Clang ASan/UBSan: 4/4 CTests passed;
- public CUDA headers compile on macOS without CUDA headers;
- `git diff --check`: passed.

Remote evidence: pending the source-bound combined L4 gate.

Remote process:

- attempt `ap-UnIzpR7EA4H53h03uHEQBI` compiled the new CUDA kernel and native
  runtime, then failed while compiling a test-only `std::array` whose extent
  used a local `std::uint64_t` constant accepted by Apple Clang but rejected as
  an NVCC non-type template argument;
- no GPU test, benchmark, checkpoint load, or inference ran;
- its complete $0.239364 ceiling remains counted conservatively, bringing the
  project tracker to $4.613196;
- the test now uses the equivalent literal extent and preserves the runtime
  `rows` value passed to the API under test.
