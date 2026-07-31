# PR 9 progress log

Last updated: 2026-07-31
Branch: `codex/pr-009-constrained-output-head`
Base: `1a9d65e`

## Milestone 1 — generic finite-choice grammar

Status: accepted

Delivered:

- application-neutral token sequences mapped to opaque choice IDs;
- deterministic sorted transitions and terminal decoding;
- prefix-free, duplicate, vocabulary, and resource-limit validation;
- focused unit tests under Debug and ASan/UBSan.

## Milestone 2 — fused restricted CUDA output head

Status: accepted

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

Remote evidence:

- accepted app: `ap-cudAIzy6z8tpmSyazMUWTy`;
- source: commit `236c0346b724332c821cce46f79f8853d3e3a447`, bundle
  `1f6ec1e5cd42ab84a8f907ff2eed77ed60dc5f81feffd56a5b3eb0b3295f8da2`;
- NVIDIA L4 (compute capability 8.9), Release CUDA build and 6/6 CTests;
- exact native SmolLM2 output `[198, 198, 504]` and exact vLLM eager/graph
  parity;
- 12/12 restricted-head benchmark cells passed exact token parity;
- observed restricted-head speedup ranged from 1.272x to 20.717x while
  avoiding 98,304 to 25,165,824 bytes of materialized logits per operation;
- CUDA graphs improved single-request inference by 3.378x and batch inference
  by 2.089x; warm prefix replay improved by 2.170x;
- compact machine-readable evidence: `docs/pr9-modal-result.json`.

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
- attempt `ap-5fuQAkND8AWFjp1N4wrv7e` confirmed that NVCC rejected the local
  `std::array` declaration itself even with a literal extent, while again
  compiling the production kernel and native runtime successfully;
- no GPU test, benchmark, checkpoint load, or inference ran;
- its complete $0.239364 ceiling remains counted conservatively, bringing the
  project tracker to $4.852560;
- the three-row invalid-device test now uses fixed C arrays, removing the
  problematic template without weakening the tested device-buffer contract.
- attempt `ap-Lcc6goooJKEXxXdP7EN5Yr` passed the previous NVCC failure point
  and reached the cached-checkpoint stage, then Modal canceled the input when
  the initiating client disconnected; retained logs contain no product-test
  failure;
- its complete $0.239364 ceiling remains counted conservatively, bringing the
  project tracker to $5.091924;
- subsequent launches use detached mode and emit a strict source-bound result
  line inside the remote function, so client lifetime cannot cancel compute or
  discard accepted evidence.
- detached attempt `ap-cudAIzy6z8tpmSyazMUWTy` passed the complete gate; its
  full $0.239364 ceiling is counted conservatively, bringing the project
  tracker to $5.331288.
