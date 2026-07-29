from copy import deepcopy
from decimal import Decimal
import unittest

from tools.modal.cuda_ci import SourceBundle
from tools.modal.rmsnorm_modal_app import (
    MAXIMUM_COST_USD,
    _strict_json_line,
    _validate_result,
)


def valid_result() -> dict[str, object]:
    return {
        "schema_version": 1,
        "result": "pass",
        "candidate_commit": "a" * 40,
        "source_bundle_sha256": "b" * 64,
        "application_id": "ap-test",
        "function_call_id": "fc-test",
        "gpu": "L4",
        "cuda_architecture": 89,
        "commands": [],
        "benchmark": {},
        "benchmark_output_sha256": "c" * 64,
        "wall_seconds": 1.0,
    }


class RmsNormModalAppTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = SourceBundle(
            commit="a" * 40,
            sha256="b" * 64,
            content=b"bundle",
        )

    def test_cost_ceiling_is_bounded(self) -> None:
        self.assertEqual(MAXIMUM_COST_USD, Decimal("0.1542480"))

    def test_strict_benchmark_json_rejects_ambiguity(self) -> None:
        self.assertEqual(
            _strict_json_line('{"schema_version":1}\n'),
            {"schema_version": 1},
        )
        for output in (
            "",
            "{}\n{}\n",
            '{"value":NaN}\n',
            '{"value":1,"value":2}\n',
        ):
            with self.subTest(output=output), self.assertRaises(
                (RuntimeError, ValueError)
            ):
                _strict_json_line(output)

    def test_local_result_validator_binds_source_and_types(self) -> None:
        _validate_result(valid_result(), self.bundle)
        mutations = {
            "schema_version": True,
            "candidate_commit": "d" * 40,
            "source_bundle_sha256": "e" * 64,
            "cuda_architecture": True,
            "wall_seconds": float("nan"),
            "benchmark_output_sha256": "not-a-sha",
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                value = deepcopy(valid_result())
                value[field] = replacement
                with self.assertRaises(RuntimeError):
                    _validate_result(value, self.bundle)


if __name__ == "__main__":
    unittest.main()
