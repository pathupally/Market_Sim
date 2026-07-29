# PR 4 progress log

Date: 2026-07-28
Scope: pretokenized SmolLM2-135M greedy FP32 CPU decode

This file was updated after every PR 4 milestone. Timestamps use the local
America/New_York timezone.

## Milestone 0 — baseline and scope

- **Timestamp:** 2026-07-28T19:47:27-0400 (EDT)
- **Current status:** Complete. Required project documents and the immutable
  model lock were read before implementation; repository status was inspected.
- **Decisions:**
  - Reuse only locked SmolLM2 revision
    `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
  - Accept explicit token IDs only. Exclude tokenizer implementation, CUDA,
    Qwen weights, performance-tuning work, and generic model abstractions.
  - Use pinned PyTorch/Transformers as the logit/token oracle.
  - Permit at most one bounded, CPU-only Modal run.
- **Files changed:** `docs/pr4-progress.md` (created).
- **Commands/tests run:** `git status --short --branch`; complete reads of
  `README.md`, `docs/inference-runtime-plan.md`, all PR 1–3 reports, and
  `models/model-lock.json`; repository `AGENTS.md` discovery.
- **Results:** Git reports `No commits yet on master` with all project files
  untracked. PRs 1–3 are acceptance-complete. No repository `AGENTS.md` exists.
- **Blockers:** None.
- **Estimated Modal cost:** $0.00 incurred. Optional 600-second CPU ceiling:
  about $0.0184 function compute, below the $24 soft cap with $6 reserved.

## Milestone 1 — implementation design and artifact verification

- **Timestamp:** 2026-07-28T19:50:47-0400 (EDT)
- **Current status:** Complete. Loader/operator seams, tests, oracle environment,
  Modal job, and cached model were inspected.
- **Decisions:**
  - Add one exact `CpuSmolLm2` owner.
  - Materialize unique mapped BF16 tensors once into aligned FP32 storage,
    retain tied embedding/head aliasing, then release the source mapping.
  - Preallocate hidden, final norm, logits, positions, 30-layer K/V, and maximum
    layer workspace before decode.
  - Use the existing complete layer unchanged and a narrow system-BLAS call at
    the existing linear boundary, retaining the scalar fallback.
  - Use pretokenized prompt `[0, 1, 2, 3]` and export all vocabulary logits for
    three steps.
- **Files changed:** `docs/pr4-progress.md`.
- **Commands/tests run:** source/CMake/test inspection; cache discovery;
  checkpoint `shasum`/`stat`; Python version probe; pinned PyTorch trial decode.
- **Results:**
  - Checkpoint is 269,060,552 bytes and matches SHA-256
    `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.
  - `.venv` has PyTorch 2.3.1, Transformers 4.40.1, safetensors 0.4.3, and
    NumPy 1.26.4.
  - Trial tokens are `[198, 198, 504]`; margins are
    `[2.558739, 3.989464, 1.012751]`.
  - No Qwen weight file exists.
- **Blockers:** None.
- **Estimated Modal cost:** $0.00 incurred; optional ceiling remains $0.0184.

## Milestone 2 — full-model path and conformance fixture

- **Timestamp:** 2026-07-28T19:58:55-0400 (EDT)
- **Current status:** Complete. Full 30-layer prefill/decode and initial
  real-checkpoint conformance pass.
- **Decisions:**
  - Limit the API to load/reset/prefill/decode, logits, limits, and memory
    accounting.
  - Match first-index argmax by retaining the lowest token ID on ties.
  - Declare `abs(error) <= 1e-4 + 1e-5 * abs(expected)` for full logits.
  - Verify fixed owned storage/address identity across repeated cycles.
- **Files changed:** `CMakeLists.txt`,
  `include/marketforge/cpu/smollm2.hpp`, `src/cpu/smollm2.cpp`,
  `src/cpu/operators.cpp`, `src/tools/smollm2_conformance.cpp`,
  `tests/unit/cpu_smollm2_tests.cpp`, the PR 4 golden JSON/safetensors files,
  both PR 4 golden Python files, and `docs/pr4-progress.md`.
- **Commands/tests run:** debug configure/build/CTest; fixture export and Python
  checks; debug real-checkpoint conformance with 20 repeats per window.
- **Results:**
  - Exact tokens `[198, 198, 504]`.
  - All 147,456 logits pass; maximum absolute error `7.05719e-05`.
  - Fixture is 590,600 bytes with SHA-256
    `55ef87774be030957fd7fdbcc20d21c8d6c996b37cb10ca0f0e6b4d196093f3f`.
  - Initial buffer-payload accounting and storage address remain fixed.
- **Blockers:** None.
- **Estimated Modal cost:** $0.00 incurred; optional ceiling remains $0.0184.

## Milestone 3 — local acceptance matrix

- **Timestamp:** 2026-07-28T20:01:04-0400 (EDT)
- **Current status:** Complete. Debug, release, sanitizer, Python, artifact, and
  cache checks pass.
- **Decisions:**
  - Retain the declared logit criterion; observed errors leave clear headroom.
  - Run one existing bounded Modal Linux job after local acceptance.
  - Revisit RSS as a leak gate if system-library page residency is not stable.
- **Files changed:** Prior PR 4 sources/tests/fixtures and
  `docs/pr4-progress.md`.
- **Commands/tests run:** all three local presets; native executable; real-model
  conformance; 14 Python tests; locked-cache verify; independent fixture export
  and byte comparison; scope/large-file/whitespace inspections.
- **Results:**
  - 36/36 native tests pass in debug, release, and ASan/UBSan.
  - Real-checkpoint parity passes in all modes with maximum absolute logit error
    `7.05719e-05`.
  - ASan/UBSan reports no issue while materializing and executing the model.
  - All 14 Python tests pass; regenerated fixture is byte-identical.
  - Cache hashes pass; no Qwen access occurred.
- **Blockers:** The formatter was not found on the initial PATH. Milestone 6
  supersedes this note after discovering Apple clang-format through `xcrun`.
- **Estimated Modal cost:** $0.00 incurred. Planned ceiling plus documented
  prior project compute remains far below the $24 soft cap.

## Milestone 4 — bounded Modal Linux validation

- **Timestamp:** 2026-07-28T20:02:18-0400 (EDT)
- **Current status:** Complete. The sole PR 4 Modal run passed; no further run
  was dispatched.
- **Decisions:** Use the existing source-only, two-core/2 GiB, one-container,
  no-GPU, 600-second job. Keep model conformance local because weights and
  downloads are intentionally excluded remotely.
- **Files changed:** `docs/pr4-progress.md`.
- **Commands/tests run:** `.venv/bin/modal run -m tools.modal.cpu_ci
  --month-to-date-usd 0.00149`.
- **Results:**
  - Run `ap-4fda1DkBo0G6FPp5bceqIf`.
  - GCC 12.2 warnings-as-errors: build and 36/36 tests pass.
  - Clang 14.0.6 ASan/UBSan: build and 36/36 tests pass.
  - Linux 4.19 gVisor/x86_64/glibc 2.36; wall time 31.922 seconds.
  - No GPU and no model download.
- **Blockers:** None.
- **Estimated Modal cost:** About $0.000978 function compute; $0.0184 bounded
  ceiling. Documented cumulative project function compute is about $0.00247,
  leaving the $6 reserve untouched. Actual billing/image build is authoritative.

## Milestone 5 — memory-gate refinement, report, and roadmap

- **Timestamp:** 2026-07-28T20:12:39-0400 (EDT)
- **Current status:** Complete. Final exact source passes all acceptance
  criteria; report and PR 5 gate are written.
- **Decisions:**
  - Include the persistent 30-layer view table so total owned storage is exact.
  - Enforce exact owned bytes/capacities/logits address in every build.
  - Use ASan's current-allocated-byte counter as the authoritative leak gate
    with a 1 MiB ceiling. Ordinary all-zone heap and RSS remain observational
    because Accelerate caches and OS page reclamation vary between windows.
  - Do not rerun Modal after the one-run allowance; the post-run change affects
    conformance accounting/reporting, not transformer math or decode behavior.
- **Files changed:** memory accounting and conformance reporter; `README.md`;
  `docs/inference-runtime-plan.md`; `docs/pr4-report.md`;
  `docs/pr4-progress.md`.
- **Commands/tests run:** final sequential debug/release/ASan builds, CTest, and
  40-cycle real-checkpoint conformance; 14 Python tests; fixture hash, Qwen
  cache, large-file, dependency, whitespace, diff, and Git status audits.
- **Results:**
  - Final owned storage: 538,668,544 bytes, unchanged in every measured cycle.
  - Logits storage address and capacities remain identical.
  - ASan current allocated bytes: 538,728,645 before/middle/after; growth 0.
  - Debug/release/ASan each pass exact tokens and all 147,456 logits.
  - 36/36 native tests pass in every preset.
  - 14/14 Python tests pass; fixture SHA-256 remains
    `55ef87774be030957fd7fdbcc20d21c8d6c996b37cb10ca0f0e6b4d196093f3f`.
  - The Qwen cache still contains config only. Git still reports no commits;
    nothing was committed or pushed.
  - PR 4 report status is acceptance-complete; roadmap now gates to PR 5.
- **Blockers:** None.
- **Estimated Modal cost:** Final PR 4 estimate remains about $0.000978 for its
  one CPU run; no GPU cost and no second validation.

## Milestone 6 — PR 4 hardening

- **Timestamp:** 2026-07-28T23:24:32-0400 (EDT)
- **Current status:** Complete. Review follow-up closed the formatter,
  lifecycle-fault, and checkpoint-provenance gaps without changing model math.
- **Decisions:**
  - Use Apple clang-format 16 through `xcrun` and mechanically format the full
    C++ source/header/test set.
  - Keep real-model lifecycle checks opt-in so the default unit suite remains
    model-free.
  - Make `python -m tools.model.conformance` the canonical real-checkpoint
    entry point. It verifies the actual checkpoint and fixture provenance
    against `models/model-lock.json` before launching C++.
  - Do not dispatch Modal; this hardening is fully testable on the development
    machine.
- **Files changed:** formatted C++ sources/headers/tests;
  `src/tools/smollm2_conformance.cpp`; `tools/model/conformance.py`;
  `tools/model/test_conformance.py`; `README.md`; `docs/pr4-report.md`;
  `docs/pr4-progress.md`.
- **Commands/tests run:** Apple clang-format 16 strict dry run; all three local
  CMake build/CTest presets; 17 Python tests; canonical guarded conformance in
  debug, release, and ASan/UBSan; explicit C++/Python whitespace scan.
- **Results:**
  - All C++ files conform to the checked-in style.
  - 36/36 native tests pass in every preset; 17/17 Python tests pass.
  - The guarded path verifies checkpoint SHA-256
    `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
    before inference in every preset.
  - Five lifecycle rejection cases pass: decode before prefill, second
    prefill, invalid prefill/decode tokens, and context overflow. Each preserves
    context length, memory accounting, logits-buffer identity, and logits.
  - Exact tokens, all 147,456 logits, and zero ASan live-heap plateau growth
    remain unchanged.
- **Blockers:** None.
- **Estimated Modal cost:** $0.00; no Modal or network call was made.
