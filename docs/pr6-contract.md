# PR 6 contract: pinned CUDA lifecycle and bounded Modal evidence

Status: frozen as amended

Date: 2026-07-29

Original freeze commit:
`94df1fbe149bd3cdcc4c70be4cf63499b1a1a1bd`

Original frozen-content SHA-256:
`fd5e67b8914dd61438976cc228610c1ad9833218154617dc5bf36115e1d90db6`

PR 6 proves one reproducible CUDA build/run boundary and correct ownership of a
stream and byte-oriented device allocation. It is infrastructure work, not a
model port or performance project.

## Locked toolchain and execution target

- Registry image:
  `nvidia/cuda:12.6.3-devel-ubuntu24.04@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a`
- Expected Linux/AMD64 image manifest:
  `sha256:badf6c452e8b1efea49d0bb956bef78adcf60e7f87ac77333208205f00ac9ade`
- CUDA toolkit: 12.6.3
- Operating system: Ubuntu 24.04, Linux/AMD64
- CMake: 3.30.5
- Ninja: 1.11.1.1
- Modal SDK: 1.3.5
- Canonical GPU: exactly one NVIDIA L4
- Canonical CUDA architecture: `89`
- C++ and CUDA language level: 20

The registry-list and AMD64 digests were resolved on 2026-07-29 with:

```text
docker buildx imagetools inspect nvidia/cuda:12.6.3-devel-ubuntu24.04
```

The machine-readable copy of this lock is
`tools/modal/cuda-toolchain-lock.json`. The remote job must reject a mismatch
between locked and observed toolkit, image, platform, or architecture evidence.

### Compatibility-policy amendment (2026-07-29)

Before implementation, the frozen lock was amended to distinguish versions
that can be pinned before the remote image runs from versions that must be
observed inside that image. The compatibility policy requires:

- GCC identity with major version 13;
- `nvcc` CUDA version 12.6;
- CUDA runtime version 12.6;
- an NVIDIA driver exposing CUDA driver API 12.6 or newer;
- cuBLAS major version 12.

This prerequisite amendment does not change the registry image, CUDA toolkit,
GPU, architecture, resource limits, or acceptance workload. It is necessary
because the image digest fixes those packaged components while the actual
host-compiler, runtime, driver, and cuBLAS version strings are obtainable only
from remote evidence. The schema therefore validates exact identities and
compatible version boundaries without pretending that an unobserved patch
version was pre-locked.

## Build boundary

`MARKETFORGE_ENABLE_CUDA` remains `OFF` by default. The CUDA-off path must not
call `enable_language(CUDA)`, `check_language(CUDA)`, or
`find_package(CUDAToolkit)`, and it must remain buildable on the local ARM64 Mac
when `CUDACXX` points to a nonexistent program.

When CUDA is explicitly enabled:

- CMake enables CUDA and creates one `MarketForge::cuda` target;
- the target uses the CUDA runtime only, with no cuBLAS dependency;
- warnings and compile options are scoped by language;
- fast math is forbidden;
- architecture 89 is explicit;
- CUDA tests are registered only in the CUDA-enabled build;
- no CUDA source enters a portable CPU target.

Public headers under `include/marketforge/cuda/` must compile as ordinary C++20
without CUDA include paths. They must contain no CUDA, NVIDIA, cuBLAS, driver,
runtime, or platform types or headers. CUDA implementation details stay in
private headers and `.cu` files.

## Minimum ownership API

Provide two move-only public RAII types:

1. `CudaStream`
   - fallible construction of a nonblocking CUDA stream;
   - an opaque `StreamHandle` value for explicit operations;
   - explicit `synchronize()` returning typed status;
   - deleted copy operations;
   - `noexcept` move operations and destructor;
   - moved-from and empty states are safe to destroy.
2. `DeviceBuffer`
   - fallible byte allocation;
   - zero bytes is valid and owns no CUDA allocation;
   - byte size and an opaque device address are observable;
   - asynchronous host-to-device and device-to-host copies require an explicit
     valid stream, checked offset, and checked byte count;
   - deleted copy operations;
   - `noexcept` move operations and destructor.

Before any CUDA call, copy operations reject:

- a null host pointer for a nonzero copy;
- an invalid or empty stream;
- offset-plus-count overflow;
- a range outside the allocation;
- a byte count not representable by native `size_t`.

Zero-byte copies are successful no-ops. Move construction empties the source.
Move assignment releases an existing destination exactly once before acquiring
the source and is safe for self-move. Destructors do not throw, synchronize, or
use the implicit default stream.

Add typed CUDA backend/runtime failures to `ErrorCode`. Preserve the numeric
CUDA error in `Status::detail`. Check immediate launch status and surface
asynchronous failures at the explicit synchronization point. Do not call
`cudaDeviceSynchronize`.

## Exact known-answer workload

The only kernel is an integer lifecycle probe. For every tested length:

```text
input[i]  = i XOR 0xA5A5A5A5
output[i] = input[i] * 3 + 7 modulo 2^32
```

Required lengths are `0`, `1`, `255`, `256`, `257`, and `1025`. Use one
explicit nondefault stream for upload, kernel launch, and download. A nonzero
case must protect the output with one sentinel word before and after the active
range. Compare every output bit and both sentinels.

The zero-length case exercises a valid zero allocation without launching a
kernel. Repeat stream creation, allocation, moves, execution, and destruction
at least 100 times under Compute Sanitizer.

The probe executable returns nonzero on any status, result, sentinel, or
manifest inconsistency and emits structured JSON. A prose-only `PASS` is not
acceptance evidence.

## Modal workflow

The trusted host, never an AGY agent, may run two ordered remote stages:

1. `cuda_compile`: pinned CUDA development image, no GPU, at most 2 physical
   cores, 4 GiB, one container, single use, and 600-second timeout. It performs
   a warnings-as-errors CUDA build and CUDA-off portability checks.
2. `gpu_smoke`: dispatched only after the compile stage and all local gates
   pass; exactly one L4, at most 2 physical cores, 4 GiB, one container,
   concurrency one, single use, and 900-second timeout.

The GPU stage runs:

- all required known-answer lengths;
- the 100-cycle lifecycle loop;
- `compute-sanitizer --tool memcheck --leak-check full` with a nonzero
  error-exit code.

Compute Sanitizer availability and a clean result are hard gates.

Probe `nsys` timeline capture and `ncu` counter access. Each capability is
recorded as `available` or `unavailable`. An unavailable result requires the
exact command, version if obtainable, exit status, and nonempty reason. Profiler
availability is not a merge gate; false success is a merge failure. Raw
profiler captures remain outside Git.

## Evidence manifest

The accepted run emits a schema-version-1 JSON manifest containing:

- clean source commit, source bundle SHA-256, and dirty status;
- locked registry-list and AMD64 image digests;
- locked and observed OS, host compiler, CMake, Ninja, CUDA toolkit, `nvcc`,
  CUDA runtime, driver, and cuBLAS versions;
- architecture and exact compile/link flags;
- Modal SDK version and dependency-lock SHA-256;
- GPU model, compute capability, and total memory, but no device UUID;
- exact known-answer lengths and result;
- lifecycle repetition count;
- Compute Sanitizer command and result;
- structured `nsys` and `ncu` capability results;
- function resources, timeouts, and remote call/application identifiers;
- operator-supplied month-to-date cost, maximum planned cost, estimated actual
  compute cost, and billing-report caveat.

A local schema validator rejects missing fields, wrong types, unknown enums,
toolchain inconsistency, dirty sources, a source hash mismatch, or a claimed
pass without all hard evidence.

## Budget and dispatch policy

The monthly budget is $30. The project software soft cap is $24 and the
untouched reserve is $6.

Month-to-date spend is mandatory; there is no default zero. Reject negative,
NaN, infinite, malformed, or otherwise non-finite cost inputs.

Before the no-GPU stage is dispatched, reserve the maximum combined cost of the
entire compile-to-GPU chain. This prevents a split-stage soft-cap bypass. At the
locked standard rates:

- 600-second compile ceiling, 2 cores, 4 GiB: `$0.021048`;
- 900-second L4 ceiling, 2 cores, 4 GiB: `$0.231372`;
- accepted two-stage chain ceiling: `$0.252420`.

Equality with the $24 soft cap is allowed; any amount above it rejects before
either remote call. A dry run prints the complete resources, stages, and maximum
cost while dispatching nothing.

PR 6 has a 60 L4-GPU-minute development ceiling and a $1.00 compute ceiling for
this harness trial. These ceilings do not authorize crossing the $24 monthly
soft cap. Modal billing reports and the Workspace budget remain authoritative;
image construction and storage charges may not be represented by the function
estimate.

### Retry and toolkit-evidence amendment (2026-07-29)

The preceding $1.00 trial compute ceiling is historical and is superseded
narrowly by this amendment. Three no-GPU attempts failed before any L4
allocation:

- attempt 1, source commit `85ec687`, Modal app
  `ap-oL8kepGh5hsJ7BIc8XCS0A`, failed because
  GCC 13 rejected NVCC-generated line markers;
- attempt 2, source commit `2713246`, Modal app
  `ap-DyntjAXVGiGV0L9fdBQHnQ`, failed because
  the replacement script did not set CMake policy `CMP0057`;
- attempt 3, source commit `57d90f4`, Modal app
  `ap-Qj6CMjxGE22AQqW2TOj1Z5`, completed the CUDA build and then failed
  evidence collection because the pinned image does not contain the optional
  `/usr/local/cuda/version.json` file.

The signed trial ledger conservatively retains all three failed full-chain
reservations, totaling `$0.757260` and 45 L4 minutes, even though none of the
attempts allocated an L4. The authoritative month-to-date spend before attempt
3 was `$0.01055264`.

To permit exactly one final corrective run, the PR 6 harness-trial compute
ceiling is raised from `$1.00` to `$1.25`; the 60 L4-GPU-minute ceiling is
unchanged. At most one fourth attempt is authorized. Its full `$0.252420`
reservation brings the conservative ledger total to `$1.009680` and the GPU
reservation total to exactly 60 minutes. This amendment does not authorize a
fifth attempt, crossing the project software soft cap of `$24`, or consuming
the untouched `$6` reserve.

For toolkit identity evidence, the exact locked CUDA toolkit version `12.6.3`
may be obtained from the pinned image's declared `CUDA_VERSION` value. Evidence
collection must cross-check that value against `nvcc` reporting CUDA 12.6 and,
when the GPU stage runs, against the observed CUDA runtime compatibility
evidence. The optional `/usr/local/cuda/version.json` file is not required and
must not be treated as the sole authoritative source. This evidence-source
change does not relax any image-digest, toolkit-version, runtime-version, or
manifest-validation gate.

Cache remote evidence by exact candidate commit and gate identifier. Never
rerun an unchanged accepted remote gate.

## Acceptance gates

- Freeze this contract before implementation.
- Preserve all PR 1–5 debug, release, ASan/UBSan, Python, formatting,
  generation, and guarded real-checkpoint gates.
- Pass a poisoned-`CUDACXX` CUDA-off configure on the Mac with no local `nvcc`.
- Compile every public CUDA header as plain C++20 and prove the required
  copy/move/destructor traits.
- Cover zero allocation, move chains, self-move, move assignment over a live
  allocation, null host pointers, invalid streams, range overflow, out-of-range
  copies, and typed launch/synchronization failures.
- Pass every known-answer length and both sentinels.
- Pass the pinned no-GPU CUDA compile stage.
- Pass the L4 known-answer and Compute Sanitizer stages.
- Validate the evidence manifest and toolchain lock.
- Truthfully record profiler capability limitations.
- Unit-test dry-run/no-dispatch, cost equality, one-cent-over, negative, NaN,
  infinity, and combined-stage budget rejection with mocked remote calls.
- Keep the final worktree clean and free of build trees, profiler captures,
  Python bytecode, secrets, credentials, model weights, and tokenizer files.
- Make no benchmark or performance claim.

## Explicit exclusions

PR 6 does not add:

- model-weight or tokenizer downloads;
- model upload, packed weights, cuBLAS inference, or transformer operators;
- FP16/BF16 model math or fast math;
- CUDA market logic;
- scheduler, KV cache, paging, shared prefixes, CUDA Graphs, or persistent
  kernels;
- PyTorch, Transformers, llama.cpp, Qwen, sampling, or quantization;
- benchmarks, throughput/latency claims, or cross-GPU comparisons;
- public CUDA/NVIDIA types.

cuBLAS FP16 forward begins in PR 7. Custom transformer kernels begin in PR 8.
