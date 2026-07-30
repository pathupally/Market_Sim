# Modal jobs

These jobs supplement local Apple Silicon development with bounded Linux and
CUDA validation. Modal model or compiler caches must live outside Git; model
weights are never copied into this source bundle.

## Budget policy

- monthly ceiling: **$30**
- project soft cap: **$24**
- untouched reserve: **$6**
- GPU allocation begins with CUDA PR 6; PRs 1–5 use CPU jobs only
- every function must specify CPU, memory, timeout, and container limits
- every GPU function must name one GPU class and estimate its timeout cost
- stop new discretionary jobs when reported month-to-date project spend plus
  the planned ceiling exceeds $24

The code-side preflight uses operator-reported month-to-date spend. Set Modal's
enforced Workspace monthly budget to $30 in **Usage & Billing** as the hard
outer cap.

Rates in `modal_budget.py` are the standard rates published on 2026-07-28.
Review them before the first cloud run of each month:

- <https://modal.com/pricing>
- <https://modal.com/docs/guide/budgets>

## Linux portability gate

The job requests no GPU. It runs a warnings-as-errors GCC debug build and a
warnings-as-errors Clang ASan/UBSan build in one two-core, 2 GiB container with
a 600-second timeout. The function's standard-rate compute ceiling is $0.0184;
image-build overhead is separate.

```bash
.venv/bin/python -B -m pip install -r tools/modal/requirements.txt
.venv/bin/python -B -m unittest tools/modal/test_modal_budget.py
.venv/bin/modal run -m tools.modal.cpu_ci --month-to-date-usd 0
```

Replace `0` with the current project spend reported by Modal. The job is tagged
`project=marketforge` and `purpose=ci` for later cost attribution.

## PR 6 CUDA lifecycle gate

The PR 6 gate uses the digest-qualified CUDA 12.6.3 Ubuntu 24.04 image in
`cuda-toolchain-lock.json`. It passes one clean `git archive` to an ordered
no-GPU compile stage and one L4 smoke stage. The second stage cannot run unless
the first passes. Compute Sanitizer is a hard gate; Nsight Systems and Nsight
Compute are probed and reported truthfully but are not required to be
available. The lock distinguishes the installed `ninja_distribution`
metadata version from the exact `ninja_binary` self-report and validates both.
The image keeps the CUDA compiler/runtime at 12.6.3 and additively installs the
SHA-pinned `cuda-sanitizer-13-0=13.0.85-1` package. Accepted evidence requires
the exact package metadata, docs release, executable path, executable
version/build/channel, byte size, and executable SHA-256 in
`cuda-toolchain-lock.json`.

Before the 100-repetition production probe, the same L4 container must run
separate invalid-global-write and device-leak canaries through the exact locked
sanitizer path and flags. Each canary must exit 97 with matching
tool-generated fault diagnostics and a positive error summary. Production then
must exit zero with `ERROR SUMMARY: 0 errors`. Missing capability, a false
success, a wrapper or alternate path, or a plain-execution fallback is a hard
failure.

Month-to-date spend is mandatory. This pure Python dry run performs the
combined-chain budget preflight. It does not import the Modal SDK, initialize
a Modal application, or dispatch:

```bash
.venv/bin/python -B -m tools.modal.cuda_ci \
  --month-to-date-usd 0.00668019 --dry-run
```

The real gate requires the PR 5 locked SmolLM2 tokenizer at an absolute path
outside the checkout. Its `tokenizer.json` is exactly 2,104,556 bytes; the
generator validates its locked content hash and package identity. This harness
does not download it and never copies it into Git.

The only supported real launch is:

```bash
.venv/bin/python -B -m tools.modal.cuda_ci \
  --month-to-date-usd 0.00668019 \
  --tokenizer-json /absolute/external/smollm2-135m/tokenizer.json \
  --launch
```

The trusted launcher runs the complete poisoned-`CUDACXX` Debug, Release,
ASan/UBSan, Python, guarded checkpoint, generation/reproducibility, and
formatting matrix on one immutable clean-HEAD archive before it initializes
Modal. Generation explicitly checks the archive's committed
`smollm2_market_action_v1.inc` against that external tokenizer. The launcher
then proves the checkout still produces the identical archive, reserves the
entire two-stage cost and 15 L4 minutes in an atomic outside-Git trial ledger,
and issues a one-use source-bound authorization ticket. Directly invoking
`cuda_modal_app.py` is unsupported and cannot dispatch without that ticket and
matching reservation.

Accepted evidence is cached outside Git by exact commit and gate, so an
unchanged accepted candidate is not billed twice. Failed attempts retain their
conservative reservation. Five reservations currently consume `$1.262100` and
75 reserved L4 minutes under the frozen `$2.00` and 120-minute envelope.
Attempts 6 and 7 are mathematically possible, but no further dispatch is
authorized without explicit user approval. Raw profiler captures remain in
remote temporary storage and are deleted after structured capability
verification.

## PR 7 vLLM conformance

The PR 7 baseline uses vLLM 0.25.1 and the locked 269 MB SmolLM2 checkpoint in
a digest-qualified CUDA 12.9 image. It sends token IDs directly, disables
tokenizer initialization and detokenization, generates exactly three greedy
tokens, and requires `[198, 198, 504]` from the existing PR 4 oracle.

The first run uses eager execution so CUDA Graphs cannot obscure basic model
parity. Prefix caching and graph execution are separate PR 8 ablations. The
checkpoint cache is a Modal Volume outside Git; every invocation verifies the
file size and SHA-256 before model construction.

Run all local Python tests from the repository root so the `tools.modal`
package cannot shadow the installed Modal SDK:

```bash
.venv/bin/python -m unittest discover -s . -p 'test_*.py'
```

After committing a clean candidate, the one-container, 15-minute L4 ceiling is
$0.239364:

```bash
.venv/bin/modal run -m tools.modal.vllm_modal_app \
  --month-to-date-usd 1.262100
```

Replace the example spend with the current Modal project spend. The local
entrypoint rejects a dirty tree, creates an immutable Git archive, checks the
$24 project soft cap, and binds the returned inference artifact to the commit
and archive SHA-256. Image build and model-transfer charges, if any, are
separate from the compute ceiling.
