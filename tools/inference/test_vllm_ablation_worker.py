import unittest

from tools.inference.vllm_ablation_worker import (
    BASE_PROMPT,
    BATCH_SIZE,
    PREFIX_BATCH_SIZE,
    PREFIX_TOKENS,
    shared_prefix_requests,
    standard_requests,
)


class VllmAblationWorkerTest(unittest.TestCase):
    def test_standard_batch_is_deterministic_and_model_native(self) -> None:
        requests = standard_requests(BATCH_SIZE, label="batch")
        self.assertEqual(len(requests), BATCH_SIZE)
        self.assertEqual(
            [request.request_id for request in requests],
            [f"batch-{index:02d}" for index in range(BATCH_SIZE)],
        )
        self.assertTrue(
            all(request.prompt_token_ids == BASE_PROMPT for request in requests)
        )

    def test_prefix_batch_has_one_exact_common_prefix(self) -> None:
        requests = shared_prefix_requests(
            PREFIX_BATCH_SIZE,
            prefix_tokens=PREFIX_TOKENS,
            vocabulary_size=49_152,
        )
        self.assertEqual(len(requests), PREFIX_BATCH_SIZE)
        prefixes = {
            request.prompt_token_ids[:PREFIX_TOKENS] for request in requests
        }
        suffixes = {
            request.prompt_token_ids[PREFIX_TOKENS:] for request in requests
        }
        self.assertEqual(len(prefixes), 1)
        self.assertEqual(len(suffixes), PREFIX_BATCH_SIZE)
        self.assertTrue(
            all(
                0 <= token < 49_152
                for request in requests
                for token in request.prompt_token_ids
            )
        )

    def test_request_builders_reject_empty_work(self) -> None:
        for call in (
            lambda: standard_requests(0, label="batch"),
            lambda: standard_requests(1, label=""),
            lambda: shared_prefix_requests(
                0, prefix_tokens=16, vocabulary_size=100
            ),
            lambda: shared_prefix_requests(
                2, prefix_tokens=0, vocabulary_size=100
            ),
        ):
            with self.subTest(call=call), self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
