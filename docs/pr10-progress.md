# PR 10 progress log

Last updated: 2026-08-01

Branch: `main`

Base: `c3a4aaf`

## Milestone 1 — deterministic radar workload

Status: locally complete

- added the C++20 radar workload library now exposed as
  `MarketForge::workloads`;
- added deterministic vehicle/target dynamics and noisy range/bearing radar;
- connected five commands to the generic token DFA;
- exercised continuous batching and transactional shared-prefix KV pages;
- emitted p50/p95/p99, deadlines, throughput, fairness, cache reuse, and
  grammar-validity metrics;
- added focused deterministic, invariant, resource, and JSON tests.

Reference workload result:

- 1,536 decisions and 1,536 radar returns;
- 384 continuous batches with maximum batch size 8;
- p50/p95/p99 latency: 7.774 / 8.988 / 9.017 ms;
- 50.0% deadline attainment under an intentionally tight 8 ms SLA;
- 100% grammar validity and 1.000 Jain completion fairness;
- 36 peak physical KV pages versus 160 without prefix sharing (77.5%
  reduction);
- 3,571 decisions per modeled service second and 27.9x faster than real time.

## Milestone 2 — portfolio replay and release package

Status: accepted

- added a dependency-free Canvas replay with target trails, radar estimates,
  vehicle selection, deadline inspection, autoplay, scrubbing, and keyboard
  controls;
- added mobile and desktop adaptations, visible focus states, reduced-motion
  handling, and trace-load error messaging;
- verified frame stepping and autoplay in Chromium at 1440x1100 and 390x844;
- verified a clean browser console after fixing 64-bit seed serialization;
- added the first portfolio README and quick demo path, later superseded by the
  inference-runtime README;
- added a committed reference trace, Apache-2.0 license, and Linux CI.

Local evidence:

- Apple Clang Debug: 5/5 CTests passed;
- Apple Clang ASan/UBSan: 5/5 CTests passed before final documentation;
- JavaScript syntax and reference JSON validation: passed;
- `git diff --check`: passed.

Remote Linux evidence:

- accepted app: `ap-cw8PqRkkFPfDdYooHZ5SKh`;
- source commit: `b173aa5d8eadcf5c6306c38fab22c5caba5b65ef`;
- x86_64 Linux/gVisor, CMake 3.25.1, Ninja 1.11.1;
- GCC 12.2 warnings-as-errors Debug build: 49/49 targets built, 5/5 CTests
  passed;
- Clang 14 ASan/UBSan build: 49/49 targets built, 5/5 CTests passed;
- total remote gate wall time: 39.548 seconds;
- CPU-only maximum compute cost: $0.0184; August conservative tracker:
  $0.0184;
- compact machine-readable evidence: `docs/pr10-modal-result.json`.

## Release status

PR10 is complete. PR9 GPU evidence remains valid because PR10 did not modify
the accepted CUDA kernels or native/vLLM model backends. The radar code is now
kept as a synthetic workload alongside the inference runtime.
