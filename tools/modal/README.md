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
conservative reservation. After four no-GPU failures, the final frozen
envelope permits at most three additional full-chain reservations under the PR
6 cumulative `$2.00` and 120-minute ceilings. Raw profiler captures remain in
remote temporary storage and are deleted after structured capability
verification.
