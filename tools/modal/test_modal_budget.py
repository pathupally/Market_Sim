from decimal import Decimal
import unittest

from tools.modal.modal_budget import (
    PROJECT_SOFT_CAP_USD,
    RESERVE_USD,
    estimate_compute_cost,
    require_project_headroom,
)


class ModalBudgetTests(unittest.TestCase):
    def test_policy_preserves_six_dollar_reserve(self) -> None:
        self.assertEqual(PROJECT_SOFT_CAP_USD, Decimal("24"))
        self.assertEqual(RESERVE_USD, Decimal("6"))

    def test_cpu_ci_timeout_ceiling_is_below_two_cents(self) -> None:
        cost = estimate_compute_cost(
            seconds=600,
            physical_cores=Decimal("2"),
            memory_gib=Decimal("2"),
        )
        self.assertEqual(cost, Decimal("0.01838400"))

    def test_gpu_cost_includes_cpu_and_memory(self) -> None:
        cost = estimate_compute_cost(
            seconds=60,
            physical_cores=Decimal("1"),
            memory_gib=Decimal("4"),
            gpu="L4",
        )
        self.assertEqual(cost, Decimal("0.01463880"))

    def test_soft_cap_rejects_projected_overspend(self) -> None:
        with self.assertRaises(RuntimeError):
            require_project_headroom(
                month_to_date_usd=Decimal("23.99"),
                planned_cost_usd=Decimal("0.02"),
            )

    def test_unknown_gpu_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_compute_cost(
                seconds=1,
                physical_cores=Decimal("1"),
                memory_gib=Decimal("1"),
                gpu="not-a-gpu",
            )


if __name__ == "__main__":
    unittest.main()
