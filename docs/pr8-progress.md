# PR 8 progress log

Last updated: 2026-07-30
Branch: `codex/pr-008-serving-optimization-core`
Base: `fbe16f4`

## Milestone 1 — deterministic sequence scheduler

Status: locally complete

Delivered:

- a backend-neutral sequence lifecycle with explicit queued, decoding, paused,
  finished, and cancelled states;
- deterministic round-robin admission shared by prefill and decode work;
- one-in-flight-item enforcement and monotonic batch generations;
- deferred cancellation for already-dispatched GPU work;
- atomic rejection of invalid completions;
- exact lifecycle, snapshot, fairness, and bounded-batch tests.

Evidence:

- repository-root Python discovery: 103 passed, 1 expected skip;
- Apple Clang Debug: build passed, 4/4 CTests passed;
- Apple Clang ASan/UBSan: build passed, 4/4 CTests passed;
- `git diff --check`: passed.

## Milestone 2 — paged KV and immutable shared prefixes

Status: locally complete

Delivered:

- fixed-size physical KV pages with lowest-index deterministic allocation;
- one transactional reservation per sequence with explicit commit/rollback;
- committed page tables that never expose uncommitted capacity;
- immutable shared-prefix publication and exact attachment reference counts;
- copy-on-append behavior for partially filled shared prefix pages;
- page-owner integrity validation and deterministic page reuse;
- physical, logical, pending, shared, and fragmentation accounting;
- repeated churn and page-conservation tests.

Evidence:

- repository-root Python discovery: 103 passed, 1 expected skip;
- Apple Clang Debug: build passed, 4/4 CTests passed;
- Apple Clang ASan/UBSan: build passed, 4/4 CTests passed;
- `git diff --check`: passed.

## Milestone 3 — CUDA restricted-token selection

Status: locally complete; GPU validation pending combined gate

Delivered:

- explicit-stream FP16 restricted greedy selection over per-row candidate
  tables;
- variable allowed-token counts with fixed-width, batch-friendly storage;
- FP32 comparisons, lowest-token-ID ties, NaN-loses behavior, and a defined
  all-NaN fallback;
- deterministic invalid-row sentinel for empty, oversized, or
  out-of-vocabulary device candidate sets;
- exact buffer-size, arithmetic, alias, stream, and launch validation;
- differential coverage against nonterminal states from the generated
  SmolLM2 market-action DFA;
- an L4 CUDA-event benchmark driven by the complete DFA state catalog.

Local evidence:

- repository-root Python discovery: 103 passed, 1 expected skip;
- Apple Clang Debug: build passed, 4/4 CTests passed;
- Apple Clang ASan/UBSan: build passed, 4/4 CTests passed;
- public CUDA header compilation passed without CUDA headers on macOS;
- `git diff --check`: passed.

Remote evidence: pending the one combined PR 8 L4 gate.

## Milestone 4 — vLLM serving ablations and combined L4 gate

Status: pending
