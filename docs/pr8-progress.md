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

Status: pending

## Milestone 3 — CUDA restricted-token selection

Status: pending

## Milestone 4 — vLLM serving ablations and combined L4 gate

Status: pending
