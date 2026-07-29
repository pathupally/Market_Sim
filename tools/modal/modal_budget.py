"""Cost estimates and project-level Modal budget policy.

Rates are the standard Modal prices published on 2026-07-28. Estimates do not
replace Modal's enforced Workspace budget because callers must supply current
month-to-date project spend.
"""

from decimal import Decimal

MONTHLY_BUDGET_USD = Decimal("30")
PROJECT_SOFT_CAP_USD = Decimal("24")
RESERVE_USD = MONTHLY_BUDGET_USD - PROJECT_SOFT_CAP_USD

CPU_CORE_SECOND_USD = Decimal("0.0000131")
MEMORY_GIB_SECOND_USD = Decimal("0.00000222")
GPU_SECOND_USD = {
    "T4": Decimal("0.000164"),
    "L4": Decimal("0.000222"),
    "A10": Decimal("0.000306"),
    "A100-40GB": Decimal("0.000583"),
    "H100": Decimal("0.001097"),
}


def estimate_compute_cost(
    *,
    seconds: int,
    physical_cores: Decimal,
    memory_gib: Decimal,
    gpu: str | None = None,
) -> Decimal:
    """Return the standard-rate compute estimate for one container."""
    if seconds < 0 or physical_cores < 0 or memory_gib < 0:
        raise ValueError("resource quantities must be non-negative")
    if gpu is not None and gpu not in GPU_SECOND_USD:
        raise ValueError(f"unknown GPU rate: {gpu}")

    duration = Decimal(seconds)
    cost = duration * (
        physical_cores * CPU_CORE_SECOND_USD
        + memory_gib * MEMORY_GIB_SECOND_USD
    )
    if gpu is not None:
        cost += duration * GPU_SECOND_USD[gpu]
    return cost


def require_project_headroom(
    *, month_to_date_usd: Decimal, planned_cost_usd: Decimal
) -> None:
    """Reject a run whose estimate would cross the project's soft cap."""
    if month_to_date_usd < 0 or planned_cost_usd < 0:
        raise ValueError("costs must be non-negative")
    projected = month_to_date_usd + planned_cost_usd
    if projected > PROJECT_SOFT_CAP_USD:
        raise RuntimeError(
            "planned Modal run would exceed the $24 project soft cap: "
            f"${projected:.4f} projected"
        )
