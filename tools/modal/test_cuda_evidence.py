from copy import deepcopy
import json
from pathlib import Path
import unittest

from tools.modal.cuda_evidence import ValidationError, validate_manifest

ROOT = Path(__file__).resolve().parents[2]
LOCK = json.loads(
    (ROOT / "tools/modal/cuda-toolchain-lock.json").read_text(encoding="utf-8")
)
HASH_A = "a" * 64
HASH_B = "b" * 64
COMMIT = "c" * 40
INDEX = LOCK["registry_image"]["index_digest"]
AMD64 = LOCK["registry_image"]["linux_amd64_digest"]


def valid_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "result": "pass",
        "source": {"commit": COMMIT, "bundle_sha256": HASH_A, "dirty": False},
        "image": {
            "reference": LOCK["registry_image"]["reference"],
            "locked_index_digest": INDEX,
            "observed_index_digest": INDEX,
            "locked_linux_amd64_digest": AMD64,
            "observed_linux_amd64_digest": AMD64,
            "operating_system": "ubuntu24.04",
            "platform": "linux/amd64",
        },
        "toolchain": {
            "locked": deepcopy(LOCK["toolchain"]),
            "observed": {
                "operating_system": "ubuntu24.04",
                "host_compiler": "gcc 13.2.0",
                "cmake": "3.30.5",
                "ninja": "1.11.1.1",
                "cuda_toolkit": "12.6.3",
                "nvcc": "12.6.3",
                "cuda_runtime": "12.6",
                "driver": "570.00",
                "driver_api": "12.8",
                "cublas": "12.6",
            },
            "cuda_architectures": [89],
            "compile_flags": ["-std=c++20", "--generate-code=arch=compute_89"],
            "link_flags": ["-lcudart"],
        },
        "modal": {
            "sdk_version": "1.3.5",
            "dependency_lock_sha256": HASH_B,
            "application_id": "ap-1",
            "compile_call_id": "fc-1",
            "gpu_call_id": "fc-2",
        },
        "gpu": {
            "model": "L4",
            "compute_capability": "8.9",
            "total_memory_mib": 23034,
        },
        "stages": {
            "cuda_compile": {
                "result": "pass",
                "gpu": None,
                "physical_cores": 2,
                "memory_gib": 4,
                "timeout_seconds": 600,
                "max_containers": 1,
                "concurrency": 1,
                "single_use": True,
                "maximum_compute_cost_usd": "0.021048",
                "wall_seconds": 10.0,
            },
            "gpu_smoke": {
                "result": "pass",
                "gpu": "L4",
                "physical_cores": 2,
                "memory_gib": 4,
                "timeout_seconds": 900,
                "max_containers": 1,
                "concurrency": 1,
                "single_use": True,
                "maximum_compute_cost_usd": "0.231372",
                "wall_seconds": 30.0,
            },
        },
        "probe": {
            "result": "pass",
            "known_answer_lengths": [0, 1, 255, 256, 257, 1025],
            "sentinels": "pass",
            "lifecycle_repetitions": 100,
        },
        "compute_sanitizer": {
            "command": [
                "compute-sanitizer",
                "--tool",
                "memcheck",
                "--leak-check",
                "full",
                "--error-exitcode=97",
                "marketforge_cuda_probe",
                "--repetitions",
                "100",
            ],
            "result": "pass",
            "exit_status": 0,
        },
        "profilers": {
            "nsys": {
                "availability": "unavailable",
                "command": ["nsys", "profile", "marketforge_cuda_probe"],
                "version": "2025.1",
                "exit_status": 1,
                "reason": "permission denied",
                "capture_sha256": None,
                "verification_command": [
                    "nsys",
                    "stats",
                    "--report",
                    "cuda_gpu_kern_sum",
                    "capture.nsys-rep",
                ],
                "verification_exit_status": 1,
                "evidence": None,
            },
            "ncu": {
                "availability": "available",
                "command": [
                    "ncu",
                    "--export",
                    "/tmp/capture",
                    "--metrics",
                    "sm__cycles_elapsed.avg",
                    "marketforge_cuda_probe",
                ],
                "version": "2025.1",
                "exit_status": 0,
                "reason": None,
                "capture_sha256": HASH_A,
                "verification_command": [
                    "ncu",
                    "--import",
                    "/tmp/capture.ncu-rep",
                ],
                "verification_exit_status": 0,
                "evidence": "sm__cycles_elapsed.avg=1234.5",
            },
        },
        "budget": {
            "month_to_date_usd": "0",
            "maximum_planned_cost_usd": "0.252420",
            "estimated_actual_compute_cost_usd": "0.00806320",
            "gpu_minutes": "0.5",
            "monthly_budget_usd": "30",
            "project_soft_cap_usd": "24",
            "reserve_usd": "6",
            "billing_report_caveat": (
                "Modal billing is authoritative; image and storage charges "
                "may be absent from this function estimate."
            ),
        },
    }


class CudaEvidenceTests(unittest.TestCase):
    def validate(self, manifest: dict[str, object]) -> None:
        validate_manifest(
            manifest,
            LOCK,
            expected_source_commit=COMMIT,
            expected_source_bundle_sha256=HASH_A,
            expected_dependency_lock_sha256=HASH_B,
        )

    def test_complete_manifest_is_accepted(self) -> None:
        self.validate(valid_manifest())

    def test_missing_and_unknown_fields_are_rejected(self) -> None:
        for mutation in ("missing", "unknown"):
            with self.subTest(mutation=mutation):
                manifest = valid_manifest()
                if mutation == "missing":
                    del manifest["probe"]
                else:
                    manifest["unexpected"] = True
                with self.assertRaises(ValidationError):
                    self.validate(manifest)

    def test_dirty_or_hash_mismatched_source_is_rejected(self) -> None:
        manifest = valid_manifest()
        manifest["source"]["dirty"] = True
        with self.assertRaises(ValidationError):
            self.validate(manifest)
        manifest = valid_manifest()
        manifest["source"]["bundle_sha256"] = "c" * 64
        with self.assertRaises(ValidationError):
            self.validate(manifest)

    def test_toolchain_and_profiler_false_success_are_rejected(self) -> None:
        manifest = valid_manifest()
        manifest["toolchain"]["observed"]["cuda_toolkit"] = "12.7"
        with self.assertRaises(ValidationError):
            self.validate(manifest)
        manifest = valid_manifest()
        manifest["profilers"]["nsys"]["availability"] = "available"
        with self.assertRaises(ValidationError):
            self.validate(manifest)

    def test_claimed_pass_without_hard_evidence_is_rejected(self) -> None:
        manifest = valid_manifest()
        manifest["compute_sanitizer"]["result"] = "fail"
        with self.assertRaises(ValidationError):
            self.validate(manifest)

    def test_mutated_frozen_lock_is_rejected_even_if_manifest_agrees(self) -> None:
        mutations = (
            ("registry_image", "operating_system", "ubuntu25.04"),
            ("registry_image", "platform", "linux/arm64"),
            ("toolchain", "cuda", "12.7.0"),
            ("toolchain", "cmake", "3.31.0"),
            ("modal", "gpu", "T4"),
        )
        for section, field, replacement in mutations:
            with self.subTest(section=section, field=field):
                lock = deepcopy(LOCK)
                lock[section][field] = replacement
                with self.assertRaises(ValidationError):
                    validate_manifest(
                        valid_manifest(),
                        lock,
                        expected_source_commit=COMMIT,
                        expected_source_bundle_sha256=HASH_A,
                        expected_dependency_lock_sha256=HASH_B,
                    )

    def test_frozen_lock_rejects_recursive_json_type_aliases(self) -> None:
        mutations = (
            ("cpp_standard", True),
            ("cuda_architectures", [89.0]),
            ("compatibility_policy", {
                **deepcopy(LOCK["toolchain"]["compatibility_policy"]),
                "host_compiler": {"identity": "gcc", "major": 13.0},
            }),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):
                lock = deepcopy(LOCK)
                lock["toolchain"][field] = replacement
                manifest = valid_manifest()
                manifest["toolchain"]["locked"] = deepcopy(lock["toolchain"])
                with self.assertRaises(ValidationError):
                    validate_manifest(
                        manifest,
                        lock,
                        expected_source_commit=COMMIT,
                        expected_source_bundle_sha256=HASH_A,
                        expected_dependency_lock_sha256=HASH_B,
                    )
        lock = deepcopy(LOCK)
        lock["modal"]["max_containers"] = True
        with self.assertRaises(ValidationError):
            validate_manifest(
                valid_manifest(),
                lock,
                expected_source_commit=COMMIT,
                expected_source_bundle_sha256=HASH_A,
                expected_dependency_lock_sha256=HASH_B,
            )

    def test_embedded_lock_rejects_recursive_json_type_aliases(self) -> None:
        manifest = valid_manifest()
        manifest["toolchain"]["locked"]["cuda_architectures"] = [89.0]
        with self.assertRaises(ValidationError):
            self.validate(manifest)

    def test_resource_cost_and_capture_claims_are_recomputed(self) -> None:
        manifest = valid_manifest()
        manifest["stages"]["gpu_smoke"]["memory_gib"] = 8
        with self.assertRaises(ValidationError):
            self.validate(manifest)
        manifest = valid_manifest()
        manifest["budget"]["estimated_actual_compute_cost_usd"] = "0.008"
        with self.assertRaises(ValidationError):
            self.validate(manifest)
        manifest = valid_manifest()
        manifest["profilers"]["ncu"]["capture_sha256"] = None
        with self.assertRaises(ValidationError):
            self.validate(manifest)

    def test_bool_json_values_and_short_commit_are_rejected(self) -> None:
        manifest = valid_manifest()
        manifest["stages"]["gpu_smoke"]["max_containers"] = True
        with self.assertRaises(ValidationError):
            self.validate(manifest)
        manifest = valid_manifest()
        manifest["probe"]["known_answer_lengths"][1] = True
        with self.assertRaises(ValidationError):
            self.validate(manifest)
        manifest = valid_manifest()
        manifest["source"]["commit"] = "c" * 39
        with self.assertRaises(ValidationError):
            validate_manifest(
                manifest,
                LOCK,
                expected_source_commit="c" * 39,
                expected_source_bundle_sha256=HASH_A,
                expected_dependency_lock_sha256=HASH_B,
            )

    def test_all_fast_math_variants_and_mixed_architectures_are_rejected(self) -> None:
        variants = (
            "--use_fast_math",
            "--ftz=true",
            "--prec-div=false",
            "--prec-sqrt=false",
            "-Ofast",
        )
        for variant in variants:
            with self.subTest(variant=variant):
                manifest = valid_manifest()
                manifest["toolchain"]["compile_flags"].append(variant)
                with self.assertRaises(ValidationError):
                    self.validate(manifest)
        manifest = valid_manifest()
        manifest["toolchain"]["compile_flags"].append(
            "--generate-code=arch=compute_80,code=sm_80"
        )
        with self.assertRaises(ValidationError):
            self.validate(manifest)

    def test_call_ids_must_be_distinct_and_caveat_substantive(self) -> None:
        manifest = valid_manifest()
        manifest["modal"]["gpu_call_id"] = manifest["modal"]["compile_call_id"]
        with self.assertRaises(ValidationError):
            self.validate(manifest)
        manifest = valid_manifest()
        manifest["budget"]["billing_report_caveat"] = "billing varies"
        with self.assertRaises(ValidationError):
            self.validate(manifest)

    def test_observed_toolchain_must_satisfy_locked_compatibility_policy(self) -> None:
        mutations = (
            ("host_compiler", "gcc 12.4.0"),
            ("nvcc", "12.7.1"),
            ("cuda_runtime", "12.5"),
            ("driver_api", "12.5"),
            ("cublas", "11.12.3"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                manifest = valid_manifest()
                manifest["toolchain"]["observed"][field] = value
                with self.assertRaises(ValidationError):
                    self.validate(manifest)
        manifest = valid_manifest()
        manifest["probe"]["lifecycle_repetitions"] = 99
        with self.assertRaises(ValidationError):
            self.validate(manifest)


if __name__ == "__main__":
    unittest.main()
