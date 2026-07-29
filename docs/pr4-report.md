# PR 4 evidence report

Date: 2026-07-28
Status: **acceptance complete; merge-ready**

## Delivered

- one exact `CpuSmolLm2` owner, not a generic model hierarchy;
- one-time conversion of the verified mapped BF16 checkpoint into aligned FP32
  storage;
- preservation of tied embedding/output-head aliasing;
- preallocated hidden, final-normalization, full-vocabulary logit, position,
  workspace, and 30-layer contiguous K/V storage;
- full-prompt prefill and one-token decode from explicit token IDs;
- deterministic lowest-ID greedy tie behavior;
- a narrow Accelerate call at the existing `linear_f32` boundary, with the
  readable scalar fallback retained for non-Apple builds;
- a pinned PyTorch full-logit fixture and reproducible exporter;
- a compiled real-checkpoint conformance executable behind a lock-verifying
  Python runner;
- fixed-storage, live-heap, and observational RSS checks across repeated decode.

No tokenizer, CUDA source, Qwen weight access, sampler, scheduler, generic model
registry, custom GEMM, or performance claim was added.

## Full-model oracle

| Contract | Value |
|---|---|
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Revision | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Checkpoint | 269,060,552-byte BF16 safetensors |
| Checkpoint SHA-256 | `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1` |
| PyTorch | 2.3.1 |
| Transformers | 4.40.1 |
| safetensors | 0.4.3 |
| Input token IDs | `[0, 1, 2, 3]` |
| Greedy tokens | `[198, 198, 504]` |
| Logit margins | `[2.558739, 3.989464, 1.012751]` |
| Compared logits | 3 × 49,152 = 147,456 |
| Logit criterion | `abs(error) <= 1e-4 + 1e-5 * abs(expected)` |
| Fixture size | 590,600 bytes |
| Fixture SHA-256 | `55ef87774be030957fd7fdbcc20d21c8d6c996b37cb10ca0f0e6b4d196093f3f` |

The exporter validates the locked checkpoint size and SHA-256 before model
construction, loads the tied checkpoint into the pinned Transformers
`LlamaForCausalLM`, materializes computation to FP32, uses eager attention, and
sets one deterministic CPU thread. It consumes explicit integers and never
loads a tokenizer.

`python -m tools.model.conformance` is the canonical runtime-conformance entry
point. Before launching the compiled executable, it hashes the supplied
checkpoint and compares its size and SHA-256 with `models/model-lock.json`. It
also hashes the fixture, checks it against its manifest, and requires that
manifest to record the same locked checkpoint. Corrupt checkpoints and
fixture/lock mismatches are covered by Python tests that assert the executable
is never invoked.

An independent export to a temporary directory reproduced the safetensors file
byte for byte. The fixture contains full output logits, token IDs, margins, and
tolerance scalars; it contains no model parameters.

## Numerical results

Debug, release, and ASan/UBSan executions selected all three expected tokens
exactly. Every one of the 147,456 logits passed.

| Measurement | Result |
|---|---:|
| Maximum absolute logit error | `7.05719e-05` |
| Maximum tolerance-scaled error, debug/sanitizer | `0.215691` |
| Maximum tolerance-scaled error, release | `0.213134` |
| Minimum fixture margin | `1.012751` |

The greedy comparison is therefore margin-qualified rather than accidental
agreement near an argmax boundary.

## Ownership and repeated decode

For the conformance shape (four prompt tokens, three expected output steps,
seven-token cache capacity), construction allocates:

| Owned storage | Bytes |
|---|---:|
| FP32 weights | 538,060,032 |
| 30-layer FP32 K/V | 322,560 |
| hidden/final/logit execution buffer | 208,128 |
| reusable layer workspace | 58,368 |
| preallocated positions | 16 |
| persistent 30-layer view table | 19,440 |
| **Total** | **538,668,544** |

The source safetensors mapping is released after conversion. The higher
construction-time peak is intentional: source BF16 and destination FP32 coexist
only while materialization is in progress.

After ten warmups, conformance runs two measured windows of 20 complete
reset/prefill/decode cycles each. Every cycle checks:

- exact greedy tokens;
- identical owned-byte accounting;
- the same logits-buffer address;
- unchanged allocation capacities.

The ASan build uses the sanitizer allocator's current-allocated-byte counter and
requires the second window to grow by no more than 1 MiB. It reported **0
bytes** of growth. Ordinary builds report all-zone live heap and RSS only as
diagnostics because Accelerate maintains private caches and macOS can reclaim
and re-fault clean weight pages under memory pressure. Debug and release still
enforce exact MarketForge buffer bytes, capacities, and storage identity on
every cycle. This separation prevents system-library cache behavior from
masking an application allocation under ASan or creating a flaky ordinary-build
failure.

## Local acceptance

Environment: Apple M3 Pro, ARM64, 36 GB unified memory, Apple Clang 16, CMake
4.4.0.

| Preset | Build | Native tests | Real-checkpoint conformance |
|---|---|---:|---|
| `mac-debug` | warnings-as-errors pass | 36/36 pass | pass |
| `mac-release` | warnings-as-errors pass | 36/36 pass | pass |
| `mac-sanitize` | ASan + UBSan pass | 36/36 pass | pass |

Seventeen Python tests pass for fetch safety, the conformance provenance gate,
PR 3/PR 4 fixture provenance and contents, deterministic hashes, and Modal
budget arithmetic.

Additional checks:

- locked cache verification rechecked both SmolLM2 file hashes;
- no default test needs a model or network;
- no Qwen weight exists in the cache and no Qwen request was made;
- no source-tree weight artifact was added;
- the new public header contains no Accelerate, PyTorch, Python, tokenizer,
  CUDA, cuBLAS, or llama.cpp dependency;
- an explicit C++/Python trailing-whitespace inspection passes;
- the complete C++ source/header/test set passes Apple clang-format 16's
  `--dry-run --Werror` check against the checked-in `.clang-format`.

## Modal Linux CPU

Exactly one PR 4 Modal validation was dispatched:

- run: `ap-4fda1DkBo0G6FPp5bceqIf`;
- Linux 4.19 gVisor / x86_64 / glibc 2.36;
- GCC 12.2 warnings-as-errors debug: build pass, 36/36 tests pass;
- Clang 14.0.6 ASan/UBSan: build pass, 36/36 tests pass;
- two physical CPU cores, 2 GiB memory, one container;
- no GPU and no model download;
- function wall time: 31.922 seconds;
- estimated function compute: about $0.000978;
- 600-second preflight ceiling: $0.0184.

The remote job validates the source-only Linux/scalar-fallback path. Full-model
logit conformance remains local by design because model weights are excluded
from the Modal source bundle. Documented cumulative project function compute is
about $0.00247, below the $24 soft cap while preserving the $6 reserve. Modal's
Usage & Billing page remains authoritative for billed compute and image builds.

After the one-run limit was consumed, the local conformance reporter was refined
to include the persistent layer-view allocation and to use ASan's allocator
counter instead of RSS as its leak gate. Transformer math, the scalar fallback,
and the decode path validated by Modal did not change; the final reporting-only
revision was revalidated in all three local presets.

## Acceptance checklist

| Criterion | Evidence |
|---|---|
| Pretokenized SmolLM2 full-model path | exact 30-layer prefill/decode owner |
| Reuse locked 269 MB checkpoint | canonical runner verifies actual checkpoint and fixture provenance before execution |
| Rejected lifecycle calls | decode-before-prefill, second prefill, invalid prefill/decode tokens, and context overflow preserve observable state |
| Logits versus pinned PyTorch | all 147,456 pass declared criterion |
| Greedy tokens versus pinned PyTorch | exact `[198, 198, 504]` |
| Stable repeated-decode memory | fixed buffers/address in all builds; ASan live heap +0 bytes |
| Debug/release/ASan/UBSan | all local matrices and real model pass |
| Bounded Modal validation | one CPU-only run passes |
| No Qwen weight download | cache/source inspection and fetch policy |
| No tokenizer/CUDA/optimization project | scope and dependency inspection |

There are no remaining PR 4 acceptance gates.

## Gate to PR 5

PR 5 may now add an immutable token DFA and minimal market trace. It should use
the existing token-ID/logit boundary and keep grammar/market semantics outside
`CpuSmolLm2`. CUDA remains gated to PR 6, and Qwen weights remain deferred.
