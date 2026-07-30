from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from tools.modal import vllm_modal_app
from tools.modal.vllm_modal_app import (
    LOCK,
    MAXIMUM_COST_USD,
    MODEL,
    VLLM_VERSION,
    _strict_json_line,
    _validate_benchmark,
    verify_checkpoint,
)


class VllmModalAppTest(unittest.TestCase):
    def test_dependency_and_model_locks_are_consistent(self) -> None:
        self.assertEqual(LOCK["schema_version"], 1)
        self.assertEqual(LOCK["packages"]["vllm"], VLLM_VERSION)
        self.assertEqual(MODEL["id"], "smollm2-135m")
        self.assertEqual(MODEL["checkpoint_bytes"], 269_060_552)
        self.assertEqual(MODEL["vocabulary_size"], 49_152)
        self.assertEqual(
            MODEL["revision"],
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        )
        self.assertEqual(LOCK["packages"]["cmake"], "3.30.5")
        self.assertEqual(LOCK["packages"]["ninja"], "1.11.1.1")
        self.assertEqual(LOCK["packages"]["nvidia-ml-py"], "13.610.43")

    def test_one_l4_run_is_bounded(self) -> None:
        self.assertEqual(MAXIMUM_COST_USD, Decimal("0.23936400"))
        self.assertEqual(LOCK["execution"]["gpu"], "L4")
        self.assertEqual(LOCK["execution"]["max_containers"], 1)
        self.assertEqual(LOCK["execution"]["timeout_seconds"], 900)

    def test_checkpoint_verification_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.write_bytes(b"test checkpoint")
            original_model = vllm_modal_app.MODEL
            try:
                vllm_modal_app.MODEL = {
                    **MODEL,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": (
                        "f908001a4b96aac17dfcc9519072c6282ad28800926524bc"
                        "5178523070356e32"
                    ),
                }
                verify_checkpoint(checkpoint)
                checkpoint.write_bytes(b"corrupt")
                with self.assertRaises(RuntimeError):
                    verify_checkpoint(checkpoint)
            finally:
                vllm_modal_app.MODEL = original_model

    def test_native_benchmark_json_is_strict(self) -> None:
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

    def test_transformer_benchmark_schema_is_strict(self) -> None:
        benchmark = {
            "schema_version": 1,
            "result": "pass",
            "operator": "transformer_elementwise_f16",
            "model_shape": "SmolLM2",
            "gpu": {
                "name": "NVIDIA L4",
                "compute_capability": "8.9",
            },
            "measurements": [
                {
                    "operation": operation,
                    "rows": 1,
                    "elements": 768,
                    "iterations": 20,
                    "average_microseconds": 1.0,
                    "logical_gib_per_second": 1.0,
                }
                for operation in ("rope_f16", "swiglu_f16")
            ],
        }
        _validate_benchmark(
            benchmark,
            operator="transformer_elementwise_f16",
            measurement_count=2,
        )
        benchmark["measurements"][0]["operation"] = "unknown"
        with self.assertRaises(RuntimeError):
            _validate_benchmark(
                benchmark,
                operator="transformer_elementwise_f16",
                measurement_count=2,
            )


if __name__ == "__main__":
    unittest.main()
