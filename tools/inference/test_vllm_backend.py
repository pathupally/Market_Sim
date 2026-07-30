from dataclasses import dataclass
import unittest

from tools.inference.contract import (
    HardwareIdentity,
    InferenceRequest,
    ModelIdentity,
    SourceIdentity,
)
from tools.inference.vllm_backend import (
    VLLM_VERSION,
    VllmAdapterError,
    VllmConfig,
    run_greedy_batch,
)


@dataclass
class FakeCompletion:
    token_ids: tuple[int, ...]
    finish_reason: str | None = "length"


@dataclass
class FakeResult:
    outputs: tuple[FakeCompletion, ...]


class FakeEngine:
    def __init__(self, results: tuple[FakeResult, ...]) -> None:
        self.results = results
        self.prompts: list[list[int]] = []
        self.sampling: object | None = None

    def generate(
        self,
        prompts: list[list[int]],
        sampling_params: object,
        *,
        use_tqdm: bool,
    ) -> tuple[FakeResult, ...]:
        if use_tqdm:
            raise AssertionError("progress output must remain disabled")
        self.prompts = prompts
        self.sampling = sampling_params
        return self.results


class FakeSampling:
    def __init__(self, **values: object) -> None:
        self.values = values


MODEL = ModelIdentity(
    "smollm2-135m",
    "HuggingFaceTB/SmolLM2-135M",
    "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
    49_152,
)
HARDWARE = HardwareIdentity("NVIDIA L4", "8.9", "12.9")
SOURCE = SourceIdentity("c" * 40, "d" * 64)
REQUEST = InferenceRequest("pr4-greedy", (0, 1, 2, 3))


class VllmBackendTest(unittest.TestCase):
    def test_adapts_token_only_greedy_generation(self) -> None:
        engine = FakeEngine(
            (FakeResult((FakeCompletion((198, 198, 504)),)),)
        )
        ticks = iter((10.0, 10.25))
        run = run_greedy_batch(
            engine=engine,
            sampling_params_factory=FakeSampling,
            model=MODEL,
            source=SOURCE,
            hardware=HARDWARE,
            requests=(REQUEST,),
            config=VllmConfig(),
            model_load_seconds=2.0,
            peak_gpu_memory=lambda: 1024,
            clock=lambda: next(ticks),
        )
        payload = run.to_payload()
        self.assertEqual(engine.prompts, [[0, 1, 2, 3]])
        self.assertEqual(
            engine.sampling.values,
            {
                "temperature": 0.0,
                "seed": 0,
                "max_tokens": 3,
                "ignore_eos": True,
                "detokenize": False,
            },
        )
        self.assertEqual(payload["backend"]["version"], VLLM_VERSION)
        self.assertEqual(
            payload["requests"][0]["generated_token_ids"],
            [198, 198, 504],
        )
        self.assertEqual(payload["metrics"]["input_tokens"], 4)
        self.assertEqual(payload["metrics"]["output_tokens"], 3)
        self.assertEqual(payload["metrics"]["requests_per_second"], 4.0)
        self.assertEqual(
            payload["metrics"]["output_tokens_per_second"], 12.0
        )

    def test_rejects_cardinality_and_completion_ambiguity(self) -> None:
        bad_results = (
            (),
            (FakeResult(()),),
            (
                FakeResult(
                    (FakeCompletion((1,)), FakeCompletion((2,)))
                ),
            ),
        )
        for results in bad_results:
            with self.subTest(results=results), self.assertRaises(
                VllmAdapterError
            ):
                ticks = iter((1.0, 2.0))
                run_greedy_batch(
                    engine=FakeEngine(results),
                    sampling_params_factory=FakeSampling,
                    model=MODEL,
                    source=SOURCE,
                    hardware=HARDWARE,
                    requests=(REQUEST,),
                    config=VllmConfig(),
                    model_load_seconds=1.0,
                    peak_gpu_memory=lambda: 0,
                    clock=lambda: next(ticks),
                )

    def test_rejects_empty_work_and_invalid_duration(self) -> None:
        engine = FakeEngine(
            (FakeResult((FakeCompletion((198, 198, 504)),)),)
        )
        with self.assertRaises(VllmAdapterError):
            run_greedy_batch(
                engine=engine,
                sampling_params_factory=FakeSampling,
                model=MODEL,
                source=SOURCE,
                hardware=HARDWARE,
                requests=(),
                config=VllmConfig(),
                model_load_seconds=1.0,
                peak_gpu_memory=lambda: 0,
            )
        with self.assertRaises(VllmAdapterError):
            run_greedy_batch(
                engine=engine,
                sampling_params_factory=FakeSampling,
                model=MODEL,
                source=SOURCE,
                hardware=HARDWARE,
                requests=(REQUEST,),
                config=VllmConfig(),
                model_load_seconds=1.0,
                peak_gpu_memory=lambda: 0,
                clock=lambda: 1.0,
            )


if __name__ == "__main__":
    unittest.main()
