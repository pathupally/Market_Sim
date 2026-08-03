# Modal validation jobs

These jobs run Linux and NVIDIA CUDA checks that are unavailable on the local
Apple Silicon development machine. Model and compiler caches remain outside
Git, and each accepted GPU result is tied to a clean source archive.

## Safety properties

- The monthly budget is $30, with a $24 project soft cap and $6 reserve.
- Callers provide current month-to-date spend before a job can launch.
- Every function fixes its CPU, memory, timeout, and GPU allocation.
- GPU entry points compute a worst-case cost before dispatch.
- Source upload uses `git archive` from a clean commit, not the working tree.
- Model revision, size, and SHA-256 are verified before construction.
- Default unit tests import neither vLLM nor a CUDA runtime.

The Python policy is a second line of defense. Modal's enforced workspace
budget should remain the hard outer cap. Rates in `modal_budget.py` are pinned
inputs and should be reviewed against [Modal pricing](https://modal.com/pricing)
before a new billing period.

## Local tests

Run from the repository root so the local `tools.modal` package cannot shadow
the installed Modal SDK:

```sh
.venv/bin/python -B -m pip install -r tools/modal/requirements.txt
.venv/bin/python -m unittest discover -s tools/modal -t . -p 'test_*.py'
```

These tests mock cloud dispatch. They verify cost arithmetic, source identity,
toolchain locks, evidence schemas, command construction, and refusal paths.

## Linux portability

The CPU job runs a warnings-as-errors GCC debug build and a Clang ASan/UBSan
build in one two-core, 2 GiB container with a 600-second timeout. It requests no
GPU.

```sh
.venv/bin/modal run -m tools.modal.cpu_ci --month-to-date-usd 0
```

Replace `0` with current project spend.

## CUDA lifecycle gate

`cuda_ci.py` validates a digest-qualified CUDA 12.6.3 image, performs no-GPU
compile checks first, and then permits one L4 stage. The GPU stage runs
invalid-write and device-leak canaries through the locked Compute Sanitizer
binary before it accepts a clean production result. It also records compiler,
runtime, device, Ninja, Nsight, sanitizer, and executable hashes.

A dry run performs the budget and source checks without importing Modal or
dispatching a job:

```sh
.venv/bin/python -B -m tools.modal.cuda_ci \
  --month-to-date-usd 0 \
  --dry-run
```

The full gate needs the locked SmolLM2 tokenizer at an absolute path outside
the repository:

```sh
.venv/bin/python -B -m tools.modal.cuda_ci \
  --month-to-date-usd 0 \
  --tokenizer-json /absolute/path/to/tokenizer.json \
  --launch
```

The two-stage chain is capped at $0.252420 and 15 L4 minutes. Launch tickets
are source-bound and single use. Accepted results are cached outside Git by
commit and gate so an unchanged candidate is not billed twice.

## Native CUDA and vLLM gate

The combined gate uses a digest-qualified CUDA 12.9 image, vLLM 0.25.1, one
L4, and the locked 269 MB SmolLM2 checkpoint. It builds and tests the native
CUDA runtime, runs native full-model conformance and kernel benchmarks, then
runs the vLLM eager/graph and cold/warm-prefix matrix.

```sh
.venv/bin/modal run --detach -m tools.modal.vllm_modal_app \
  --month-to-date-usd 0
```

The function is capped at $0.239364 and 15 minutes. Detached mode keeps the
remote input alive if the local client disconnects. The app writes its final
`MARKETFORGE_GATE_RESULT:` record to retained logs before returning.

Accepted measurements and their source identity are summarized in
[`BENCHMARKS.md`](../../BENCHMARKS.md). Internal run logs remain outside the
public repository.
