from copy import deepcopy
import unittest

from tools.inference.contract import (
    BackendIdentity,
    ContractError,
    GenerationSettings,
    HardwareIdentity,
    InferenceOutput,
    InferenceRun,
    ModelIdentity,
    RunMetrics,
    SourceIdentity,
    load_run_json,
    validate_run_payload,
)


def valid_run() -> InferenceRun:
    return InferenceRun(
        source=SourceIdentity("c" * 40, "d" * 64),
        backend=BackendIdentity("vllm", "0.25.1", "eager"),
        model=ModelIdentity(
            "smollm2-135m",
            "HuggingFaceTB/SmolLM2-135M",
            "a" * 40,
            "b" * 64,
            49_152,
        ),
        hardware=HardwareIdentity("NVIDIA L4", "8.9", "12.9"),
        generation=GenerationSettings("float16", 0.0, 0, 3),
        prefix_caching=False,
        cuda_graphs=False,
        structured_output=False,
        outputs=(
            InferenceOutput(
                "pr4-greedy",
                (0, 1, 2, 3),
                (198, 198, 504),
                "length",
            ),
        ),
        metrics=RunMetrics(1.0, 0.5, 4, 3, 2.0, 6.0, 1_024),
    )


class InferenceContractTest(unittest.TestCase):
    def test_round_trip_is_strict_and_deterministic(self) -> None:
        run = valid_run()
        self.assertEqual(load_run_json(run.to_json()), run.to_payload())
        self.assertEqual(run.to_json(), run.to_json())

    def test_rejects_duplicate_and_non_finite_json(self) -> None:
        for source in (
            '{"schema_version":1,"schema_version":1}',
            '{"value":NaN}',
        ):
            with self.subTest(source=source), self.assertRaises(ContractError):
                load_run_json(source)

    def test_rejects_schema_and_provenance_mutations(self) -> None:
        payload = valid_run().to_payload()
        mutations = (
            ("schema_version", True),
            ("result", "failure"),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):
                value = deepcopy(payload)
                value[field] = replacement
                with self.assertRaises(ContractError):
                    validate_run_payload(value)

        nested_mutations = (
            ("source", "commit", "dirty"),
            ("source", "bundle_sha256", "short"),
            ("backend", "mode", "unknown"),
            ("model", "revision", "moving-main"),
            ("model", "checkpoint_sha256", "short"),
            ("hardware", "compute_capability", "L4"),
            ("generation", "temperature", 0.1),
            ("features", "cuda_graphs", 1),
            ("metrics", "input_tokens", 5),
            ("metrics", "requests_per_second", 3.0),
        )
        for section, field, replacement in nested_mutations:
            with self.subTest(section=section, field=field):
                value = deepcopy(payload)
                value[section][field] = replacement
                with self.assertRaises(ContractError):
                    validate_run_payload(value)

    def test_rejects_token_and_request_inconsistency(self) -> None:
        payload = valid_run().to_payload()
        changes = (
            ("request_id", "pr4-greedy"),
            ("prompt_token_ids", [49_152]),
            ("generated_token_ids", [1, 2, 3, 4]),
            ("finish_reason", "cancelled"),
        )
        for field, replacement in changes:
            with self.subTest(field=field):
                value = deepcopy(payload)
                if field == "request_id":
                    value["requests"].append(deepcopy(value["requests"][0]))
                else:
                    value["requests"][0][field] = replacement
                with self.assertRaises(ContractError):
                    validate_run_payload(value)


if __name__ == "__main__":
    unittest.main()
