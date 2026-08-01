# PR 10 progress log

Last updated: 2026-08-01

Branch: `main`

Base: `c3a4aaf`

## Milestone 1 — deterministic autonomy workload

Status: locally complete

- added a public `Velorum::autonomy` C++20 library;
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

Status: locally complete; final Linux gate pending

- added a dependency-free Canvas replay with target trails, radar estimates,
  vehicle selection, deadline inspection, autoplay, scrubbing, and keyboard
  controls;
- added mobile and desktop adaptations, visible focus states, reduced-motion
  handling, and trace-load error messaging;
- verified frame stepping and autoplay in Chromium at 1440x1100 and 390x844;
- verified a clean browser console after fixing 64-bit seed serialization;
- rewrote the README around Velorum, measured evidence, and a quick demo path;
- added a committed reference trace, Apache-2.0 license, and Linux CI.

Local evidence:

- Apple Clang Debug: 5/5 CTests passed;
- Apple Clang ASan/UBSan: 5/5 CTests passed before final documentation;
- JavaScript syntax and reference JSON validation: passed;
- `git diff --check`: passed.
