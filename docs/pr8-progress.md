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

Remote evidence:

- accepted source-bound L4 commit:
  `8756527e532c21cda1e9513421bf73cae163465a`;
- source bundle SHA-256:
  `7af82a300b9d12506d28b2a1adaa1c8499e4f1792ac5aab7fb6904c1fe916d1a`;
- CUDA 12.9.41/GNU 11.4.0 build: 76/76 steps passed;
- CTest: 6/6 passed, including the action-DFA differential test;
- the complete 3,230-state/3,229-arc grammar ran at 3.703 microseconds for
  256 rows and 501 total allowed candidates on one NVIDIA L4.

## Milestone 4 — vLLM serving ablations and combined L4 gate

Status: remotely validated; PR 8 implementation complete

Delivered:

- pinned vLLM 0.25.1 eager and CUDA-graph/hybrid workers;
- token-ID-only single-request and 16-request batch comparisons;
- automatic prefix caching enabled in both execution modes so graph
  comparisons hold cache configuration constant;
- a graph-mode 8-request cold/warm replay with one exact 128-token common
  prefix;
- separate child processes for eager and graph engines so GPU state is
  deterministically released between configurations;
- strict source identity, exact-output, feature-mode, cardinality, timing,
  throughput, and prefix-replay validation;
- one combined Modal function that builds/tests native CUDA, benchmarks the
  action-DFA restricted kernel, runs native SmolLM2 inference, and executes
  both vLLM workers under the existing 15-minute L4 ceiling.

Local evidence:

- focused vLLM worker/gate tests: 12 passed;
- repository-root Python discovery: 109 passed, 1 expected skip;
- no local test imports vLLM, initializes CUDA, downloads a model, or performs
  network access.

Remote evidence:

- accepted application: `ap-mpGjoZOhPdVnWOXQnKXpfW`;
- exact native and vLLM standard-request output:
  `[198, 198, 504]`;
- eager single/batch throughput: 6.153/142.646 requests per second;
- graph single/batch throughput: 23.341/344.428 requests per second;
- graph speedup: 3.793x single and 2.415x batch;
- eight-request, 128-token common-prefix replay preserved every output and
  measured a 2.080x warm-cache speedup;
- the complete accepted summary is recorded in `docs/pr8-modal-result.json`.

Remote process:

- attempt `ap-op8zaNZ2egZlAJij9VEWH5` reached an L4 container but failed during
  function import because the worker eagerly read `vllm-lock.json` from
  Modal's Python-only package mount;
- no CUDA build, benchmark, model load, or inference ran;
- the app was stopped, and the complete $0.239364 ceiling remains counted
  conservatively in the project budget tracker;
- lock loading now occurs only inside the extracted source-bound worker.
- attempt `ap-9FIbqhhVX9XaVsengswyl7` then exposed the gate module's older
  import-time assumption that its mounted path retained the local directory
  depth; Modal mounts the entry module at `/root/vllm_modal_app.py`;
- no CUDA build, benchmark, model load, or inference ran in that attempt;
- its complete $0.239364 ceiling also remains counted conservatively;
- the gate now uses `modal.is_local()` and an explicit `Image.add_local_file`
  lock mount, so remote imports make no assumption about local path depth.
- attempt `ap-Wgvy7OiQaCfOkeMhdOlhVu` passed container import, the CUDA
  build/tests/benchmarks, checkpoint verification, and native inference, then
  reached the 900-second ceiling while waiting for the first isolated vLLM
  subprocess to exit;
- its complete $0.239364 ceiling remains counted conservatively;
- the worker now calls the documented engine/engine-core `shutdown()` path
  with a timeout before returning its artifact, then forces collection so
  vLLM's background processes and GPU state cannot keep the child alive.
- attempt `ap-WWDgPsp06R8eTzEga0UESA` confirmed that vLLM's process teardown
  can still outlive that API-level shutdown and again reached the 900-second
  gate ceiling after the native phases;
- its complete $0.239364 ceiling remains counted conservatively;
- workers now flush the fully validated artifact before teardown; the parent
  reads it from a dedicated process-group pipe, then sends TERM/KILL to the
  entire group before starting the next execution mode;
- each worker also has a 330-second artifact deadline inside the 900-second
  combined gate, preventing an opaque child from consuming the whole run.
- attempt `ap-mpGjoZOhPdVnWOXQnKXpfW` accepted the artifact-first worker
  lifecycle and completed the full combined gate;
- accepted commit:
  `8756527e532c21cda1e9513421bf73cae163465a`;
- accepted source bundle SHA-256:
  `7af82a300b9d12506d28b2a1adaa1c8499e4f1792ac5aab7fb6904c1fe916d1a`;
- CUDA 12.9.41/GNU 11.4.0 build: 76/76 steps passed;
- CTest: 6/6 passed;
- native FP16 SmolLM2 generated `[198, 198, 504]` in 288.467
  milliseconds and owned 269,372,588 device bytes;
- vLLM eager batch-16 generated the same result for all requests at 142.646
  requests per second;
- vLLM graph batch-16 generated the same result for all requests at 344.428
  requests per second;
- graph execution measured 3.793x single-request and 2.415x batch speedups
  over eager execution;
- automatic prefix caching preserved exact cold/warm outputs and measured a
  2.080x warm replay speedup for eight 129-token prompts sharing 128 tokens;
- every attempt was bounded to $0.239364; conservative project accounting is
  now $4.373832, below the $24 software cap and the required $6 reserve.

## Accepted gate

The accepted source-bound Modal invocation:

1. configured and compiled the immutable clean Git archive for compute 8.9;
2. passed all native CPU/CUDA tests;
3. recorded cuBLAS, RMSNorm, RoPE, SwiGLU, and DFA-restricted CUDA timings;
4. verified the locked 269 MB checkpoint;
5. ran exact native FP16 SmolLM2 prefill/decode;
6. ran isolated vLLM eager and CUDA-graph single/batch ablations;
7. replayed an exact common-prefix batch through automatic prefix caching;
8. validated source identity, schemas, timings, cardinalities, and token
   equality before accepting the result.

Result: **passed**.
Maximum function compute cost: **$0.239364**.
Maximum duration: **15 L4 minutes**.
Maximum containers: **1**.
