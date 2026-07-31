from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import time
import unittest

from tools.modal import vllm_modal_app
from tools.modal.vllm_modal_app import (
    LOCK,
    MAXIMUM_COST_USD,
    MODEL,
    RESULT_PREFIX,
    VLLM_VERSION,
    _build_vllm_ablations,
    _prefixed_json_line,
    _run_vllm_worker,
    _strict_json_line,
    _validate_benchmark,
    _validate_native_inference,
    _validate_restricted_benchmark,
    _validate_restricted_output_head_benchmark,
    _validate_vllm_ablations,
    verify_checkpoint,
)

SOURCE = {"commit": "c" * 40, "bundle_sha256": "d" * 64}


def _inference_payload(
    *,
    mode: str,
    count: int,
    prompt: list[int],
    generated: list[int],
    seconds: float,
    label: str,
) -> dict[str, object]:
    requests = [
        {
            "request_id": f"{label}-{index:02d}",
            "prompt_token_ids": prompt,
            "generated_token_ids": generated,
            "finish_reason": "length",
        }
        for index in range(count)
    ]
    input_tokens = count * len(prompt)
    output_tokens = count * len(generated)
    return {
        "schema_version": 1,
        "result": "pass",
        "source": SOURCE,
        "backend": {
            "name": "vllm",
            "version": VLLM_VERSION,
            "mode": mode,
        },
        "model": {
            "id": MODEL["id"],
            "repository": MODEL["repository"],
            "revision": MODEL["revision"],
            "checkpoint_sha256": MODEL["checkpoint_sha256"],
            "vocabulary_size": MODEL["vocabulary_size"],
        },
        "hardware": {
            "device_name": "NVIDIA L4",
            "compute_capability": "8.9",
            "cuda_version": "12.9",
        },
        "generation": {
            "dtype": "float16",
            "temperature": 0.0,
            "seed": 0,
            "max_output_tokens": 3,
        },
        "features": {
            "prefix_caching": True,
            "cuda_graphs": mode == "cuda_graph",
            "structured_output": False,
        },
        "requests": requests,
        "metrics": {
            "model_load_seconds": 1.0,
            "inference_seconds": seconds,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "requests_per_second": count / seconds,
            "output_tokens_per_second": output_tokens / seconds,
            "peak_gpu_memory_bytes": 1,
        },
    }


def _mode_payload(mode: str) -> dict[str, object]:
    runs = {
        "single": _inference_payload(
            mode=mode,
            count=1,
            prompt=[0, 1, 2, 3],
            generated=[198, 198, 504],
            seconds=0.2,
            label="single",
        ),
        "batch": _inference_payload(
            mode=mode,
            count=16,
            prompt=[0, 1, 2, 3],
            generated=[198, 198, 504],
            seconds=0.4,
            label="batch",
        ),
    }
    if mode == "cuda_graph":
        prefix_prompt = list(range(1, 130))
        runs["prefix_cold"] = _inference_payload(
            mode=mode,
            count=8,
            prompt=prefix_prompt,
            generated=[7, 8, 9],
            seconds=0.3,
            label="prefix",
        )
        runs["prefix_warm"] = _inference_payload(
            mode=mode,
            count=8,
            prompt=prefix_prompt,
            generated=[7, 8, 9],
            seconds=0.1,
            label="prefix",
        )
    return {
        "schema_version": 1,
        "result": "pass",
        "mode": mode,
        "settings": {
            "batch_size": 16,
            "prefix_batch_size": 8,
            "prefix_tokens": 128,
            "prefix_caching": True,
        },
        "runs": runs,
    }


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

    def test_prefixed_worker_json_is_strict(self) -> None:
        prefix = "RESULT="
        self.assertEqual(
            _prefixed_json_line('noise\nRESULT={"schema_version":1}\n', prefix),
            {"schema_version": 1},
        )
        for output in (
            "",
            "RESULT={}\nRESULT={}\n",
            "RESULT={\"value\":NaN}\n",
            "RESULT={\"value\":1,\"value\":2}\n",
        ):
            with self.subTest(output=output), self.assertRaises(
                (RuntimeError, ValueError)
            ):
                _prefixed_json_line(output, prefix)

    def test_worker_artifact_preempts_stalled_process_teardown(self) -> None:
        started = time.monotonic()
        payload = _run_vllm_worker(
            [
                sys.executable,
                "-c",
                (
                    "import time;"
                    f"print('{RESULT_PREFIX}' + "
                    "'{\"schema_version\":1}', flush=True);"
                    "time.sleep(30)"
                ),
            ],
            cwd=Path.cwd(),
            environment={},
            timeout_seconds=2.0,
        )
        self.assertEqual(payload, {"schema_version": 1})
        self.assertLess(time.monotonic() - started, 2.0)

    def test_worker_without_an_artifact_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            _run_vllm_worker(
                [sys.executable, "-c", "print('no artifact')"],
                cwd=Path.cwd(),
                environment={},
                timeout_seconds=2.0,
            )

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

    def test_native_inference_schema_is_strict(self) -> None:
        evidence = {
            "schema_version": 1,
            "result": "pass",
            "backend": "native_cuda_f16",
            "model": "SmolLM2-135M",
            "gpu": {
                "name": "NVIDIA L4",
                "compute_capability": "8.9",
            },
            "prompt_token_ids": [0, 1, 2, 3],
            "generated_token_ids": [198, 198, 504],
            "model_load_seconds": 1.0,
            "inference_seconds": 1.0,
            "context_length": 6,
            "memory": {
                "weight_bytes": 10,
                "kv_bytes": 20,
                "execution_bytes": 30,
                "total_device_bytes": 60,
            },
        }
        _validate_native_inference(evidence)
        evidence["generated_token_ids"] = [198, 198, 198]
        with self.assertRaises(RuntimeError):
            _validate_native_inference(evidence)

    def test_restricted_greedy_benchmark_schema_is_strict(self) -> None:
        benchmark = {
            "schema_version": 1,
            "result": "pass",
            "operator": "restricted_greedy_f16",
            "grammar": "smollm2_market_action_v1",
            "vocabulary_size": 49_152,
            "grammar_states": 10,
            "grammar_arcs": 20,
            "maximum_allowed_tokens": 4,
            "gpu": {
                "name": "NVIDIA L4",
                "compute_capability": "8.9",
            },
            "measurements": [
                {
                    "rows": rows,
                    "total_allowed_candidates": rows * 2,
                    "iterations": 10,
                    "average_microseconds": 1.0,
                    "sequences_per_second": 2.0,
                    "candidate_gib_per_second": 3.0,
                }
                for rows in (1, 16, 256)
            ],
        }
        _validate_restricted_benchmark(benchmark)
        benchmark["measurements"][0]["candidate_gib_per_second"] = 0.0
        with self.assertRaises(RuntimeError):
            _validate_restricted_benchmark(benchmark)

    def test_restricted_output_head_benchmark_schema_is_strict(self) -> None:
        benchmark = {
            "schema_version": 1,
            "result": "pass",
            "operator": "restricted_output_head_f16",
            "model_shape": "SmolLM2-135M",
            "hidden_size": 576,
            "vocabulary_size": 49_152,
            "gpu": {
                "name": "NVIDIA L4",
                "compute_capability": "8.9",
            },
            "measurements": [
                {
                    "rows": rows,
                    "allowed_tokens_per_row": allowed,
                    "iterations": 10,
                    "full_average_microseconds": 10.0,
                    "restricted_average_microseconds": 2.0,
                    "speedup": 5.0,
                    "full_scores": rows * 49_152,
                    "restricted_scores": rows * allowed,
                    "materialized_logit_bytes_avoided": rows * 49_152 * 2,
                    "exact_token_parity": True,
                }
                for rows in (1, 16, 256)
                for allowed in (2, 8, 32, 128)
            ],
        }
        _validate_restricted_output_head_benchmark(benchmark)
        benchmark["measurements"][0]["exact_token_parity"] = False
        with self.assertRaises(RuntimeError):
            _validate_restricted_output_head_benchmark(benchmark)

    def test_vllm_ablation_schema_and_comparisons_are_strict(self) -> None:
        result = _build_vllm_ablations(
            _mode_payload("eager"),
            _mode_payload("cuda_graph"),
            source=SOURCE,
        )
        _validate_vllm_ablations(result)
        self.assertAlmostEqual(
            result["comparisons"]["warm_prefix_speedup"], 3.0
        )
        result["cuda_graph"]["runs"]["batch"]["backend"]["mode"] = "eager"
        with self.assertRaises(RuntimeError):
            _validate_vllm_ablations(result)


if __name__ == "__main__":
    unittest.main()
