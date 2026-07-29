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
.venv/bin/python -m pip install -r tools/modal/requirements.txt
.venv/bin/python -m unittest tools/modal/test_modal_budget.py
.venv/bin/modal run -m tools.modal.cpu_ci --month-to-date-usd 0
```

Replace `0` with the current project spend reported by Modal. The job is tagged
`project=marketforge` and `purpose=ci` for later cost attribution.
