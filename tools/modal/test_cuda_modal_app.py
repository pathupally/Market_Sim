from decimal import Decimal
from importlib.metadata import PackageNotFoundError
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from tools.modal.cuda_ci import EvidenceCache, GATE_ID
from tools.modal.cuda_evidence import validate_manifest
from tools.modal.cuda_modal_app import (
    LOCK,
    _configure_cuda,
    _installed_distribution_version,
    _manifest,
    _profiler_capability,
    _read_cuda_toolkit_version,
    _validate_compile_result,
    _validate_gpu_result,
    _verify_ncu_output,
    _verify_nsys_output,
)

COMMIT = "c" * 40
SOURCE_HASH = "a" * 64
DEPENDENCY_HASH = "b" * 64


def profiler(name: str) -> dict[str, object]:
    if name == "nsys":
        return {
            "availability": "unavailable",
            "command": ["nsys", "profile", "probe"],
            "version": "2025.1",
            "exit_status": 1,
            "reason": "CUDA timeline capture is unavailable",
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
        }
    return {
        "availability": "available",
        "command": [
            "ncu",
            "--metrics",
            "sm__cycles_elapsed.avg",
            "--export",
            "capture",
            "probe",
        ],
        "version": "2025.1",
        "exit_status": 0,
        "reason": None,
        "capture_sha256": "d" * 64,
        "verification_command": [
            "ncu",
            "--import",
            "capture.ncu-rep",
        ],
        "verification_exit_status": 0,
        "evidence": "sm__cycles_elapsed.avg=1234.5",
    }


def gpu_result() -> dict[str, object]:
    return {
        "result": "pass",
        "candidate_commit": COMMIT,
        "source_bundle_sha256": SOURCE_HASH,
        "remote_call_id": "fc-gpu",
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
        "profilers": {"nsys": profiler("nsys"), "ncu": profiler("ncu")},
        "gpu": {
            "model": "L4",
            "compute_capability": "8.9",
            "total_memory_mib": 23034,
        },
        "runtime": {
            "cuda_runtime": "12.6",
            "driver": "570.00",
            "driver_api": "12.8",
            "cublas": "12.6.4.1",
        },
        "cuda_checks": [{"command": ["ctest"], "exit_status": 0}],
        "wall_seconds": 30.0,
    }


def compile_result() -> dict[str, object]:
    return {
        "result": "pass",
        "candidate_commit": COMMIT,
        "source_bundle_sha256": SOURCE_HASH,
        "remote_call_id": "fc-compile",
        "image": {
            "index_digest": LOCK["registry_image"]["index_digest"],
            "linux_amd64_digest": LOCK["registry_image"][
                "linux_amd64_digest"
            ],
        },
        "observed": {
            "operating_system": "ubuntu24.04",
            "platform": "linux/amd64",
            "host_compiler": "gcc 13.2.0",
            "cmake": "3.30.5",
            "ninja_distribution": "1.11.1.1",
            "ninja_binary": "1.11.1.git.kitware.jobserver-1",
            "cuda_toolkit": "12.6.3",
            "nvcc": "12.6.85",
        },
        "compile_flags": [
            "-std=c++20 --generate-code=arch=compute_89,code=sm_89"
        ],
        "link_flags": ["-lcudart"],
        "cuda_off_checks": [{"command": ["ctest"], "exit_status": 0}],
        "cuda_checks": [{"command": ["cmake", "--build"], "exit_status": 0}],
        "wall_seconds": 10.0,
    }


class ProductionManifestTests(unittest.TestCase):
    def test_ninja_distribution_version_uses_installed_metadata(self) -> None:
        with mock.patch(
            "tools.modal.cuda_modal_app.distribution_version",
            return_value="1.11.1.1",
        ) as version:
            self.assertEqual(
                _installed_distribution_version("ninja"),
                "1.11.1.1",
            )
        version.assert_called_once_with("ninja")

    def test_ninja_distribution_version_rejects_missing_or_invalid_metadata(
        self,
    ) -> None:
        with mock.patch(
            "tools.modal.cuda_modal_app.distribution_version",
            side_effect=PackageNotFoundError("ninja"),
        ):
            with self.assertRaisesRegex(RuntimeError, "distribution is missing"):
                _installed_distribution_version("ninja")
        for value in ("", True, None):
            with self.subTest(value=value), mock.patch(
                "tools.modal.cuda_modal_app.distribution_version",
                return_value=value,
            ):
                with self.assertRaisesRegex(RuntimeError, "version is missing"):
                    _installed_distribution_version("ninja")

    def test_toolkit_version_uses_image_declaration_without_optional_json(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "version.json"
            with mock.patch.dict(
                os.environ, {"CUDA_VERSION": "12.6.3"}, clear=True
            ):
                self.assertEqual(
                    _read_cuda_toolkit_version(
                        "12.6.85",
                        version_file=missing,
                    ),
                    "12.6.3",
                )

    def test_toolkit_version_cross_checks_matching_optional_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            version_file = Path(directory) / "version.json"
            version_file.write_text(
                '{"cuda":{"name":"CUDA SDK","version":"12.6.3"}}',
                encoding="utf-8",
            )
            self.assertEqual(
                _read_cuda_toolkit_version(
                    "12.6",
                    environment={"CUDA_VERSION": "12.6.3"},
                    version_file=version_file,
                ),
                "12.6.3",
            )

    def test_toolkit_version_rejects_missing_and_malformed_versions(
        self,
    ) -> None:
        cases = (
            ({}, "12.6"),
            ({"CUDA_VERSION": ""}, "12.6"),
            ({"CUDA_VERSION": "12.6"}, "12.6"),
            ({"CUDA_VERSION": "12.6.03"}, "12.6"),
            ({"CUDA_VERSION": "12.6.4"}, "12.6"),
            ({"CUDA_VERSION": "12.6.3"}, ""),
            ({"CUDA_VERSION": "12.6.3"}, "12"),
            ({"CUDA_VERSION": "12.6.3"}, "12.06.85"),
        )
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "version.json"
            for environment, nvcc_version in cases:
                with self.subTest(
                    environment=environment, nvcc_version=nvcc_version
                ):
                    with self.assertRaises(RuntimeError):
                        _read_cuda_toolkit_version(
                            nvcc_version,
                            environment=environment,
                            version_file=missing,
                        )

    def test_toolkit_version_rejects_env_nvcc_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "nvcc"):
                _read_cuda_toolkit_version(
                    "12.7.1",
                    environment={"CUDA_VERSION": "12.6.3"},
                    version_file=Path(directory) / "version.json",
                )

    def test_toolkit_version_rejects_optional_json_mismatch_and_schema(
        self,
    ) -> None:
        documents = (
            '{"cuda":{"version":"12.6.4"}}',
            "{}",
            '{"cuda":[]}',
            '{"cuda":{"version":12.63}}',
            '{"cuda":{"version":"12.6"}}',
            '{"cuda":{"version":"12.6.3","version":"12.6.3"}}',
            "not-json",
        )
        with tempfile.TemporaryDirectory() as directory:
            version_file = Path(directory) / "version.json"
            for document in documents:
                with self.subTest(document=document):
                    version_file.write_text(document, encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        _read_cuda_toolkit_version(
                            "12.6.85",
                            environment={"CUDA_VERSION": "12.6.3"},
                            version_file=version_file,
                        )

    def test_production_manifest_round_trips_through_validator(self) -> None:
        source = SimpleNamespace(commit=COMMIT, sha256=SOURCE_HASH)
        dispatcher = SimpleNamespace(
            compile_call_id="fc-compile", gpu_call_id="fc-gpu"
        )
        compile = compile_result()
        manifest = _manifest(
            month_to_date=Decimal("0.00668019"),
            source_bundle=source,
            dispatcher=dispatcher,
            compile_result=compile,
            gpu_result=gpu_result(),
            dependency_lock_sha256=DEPENDENCY_HASH,
            application_id="ap-roundtrip",
            modal_sdk_version="1.3.5",
        )
        validate_manifest(
            manifest,
            LOCK,
            expected_source_commit=COMMIT,
            expected_source_bundle_sha256=SOURCE_HASH,
            expected_dependency_lock_sha256=DEPENDENCY_HASH,
        )
        self.assertNotIn("name", manifest["stages"]["cuda_compile"])
        with tempfile.TemporaryDirectory() as directory:
            cache = EvidenceCache(Path(directory))
            cache.store(
                candidate_commit=COMMIT,
                gate_id=GATE_ID,
                manifest=manifest,
                lock=LOCK,
                expected_source_bundle_sha256=SOURCE_HASH,
                expected_dependency_lock_sha256=DEPENDENCY_HASH,
            )
            self.assertEqual(
                cache.load(candidate_commit=COMMIT, gate_id=GATE_ID),
                manifest,
            )

    def test_gpu_result_rejects_bool_duration_and_provenance_mismatch(self) -> None:
        source = SimpleNamespace(commit=COMMIT, sha256=SOURCE_HASH)
        result = gpu_result()
        result["wall_seconds"] = True
        with self.assertRaises(RuntimeError):
            _validate_gpu_result(
                result,
                source_bundle=source,
                expected_call_id="fc-gpu",
                compile_call_id="fc-compile",
            )
        result = gpu_result()
        result["source_bundle_sha256"] = "e" * 64
        with self.assertRaises(RuntimeError):
            _validate_gpu_result(
                result,
                source_bundle=source,
                expected_call_id="fc-gpu",
                compile_call_id="fc-compile",
            )

    def test_gpu_result_rejects_reused_call_identifier(self) -> None:
        with self.assertRaises(RuntimeError):
            _validate_gpu_result(
                gpu_result(),
                source_bundle=SimpleNamespace(
                    commit=COMMIT, sha256=SOURCE_HASH
                ),
                expected_call_id="fc-gpu",
                compile_call_id="fc-gpu",
            )

    def test_no_gpu_compile_stage_excludes_gpu_runtime_ctests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build"
            with (
                mock.patch(
                    "tools.modal.cuda_modal_app._run",
                    return_value={"output": "", "exit_status": 0},
                ) as run,
                mock.patch(
                    "tools.modal.cuda_modal_app._compile_flags",
                    return_value=(["-std=c++20 compute_89"], ["-lcudart"]),
                ),
            ):
                _configure_cuda(Path("/source"), build, run_gpu_tests=False)
            commands = [call.args[0] for call in run.call_args_list]
            ctest = next(command for command in commands if command[0] == "ctest")
            self.assertIn("--tests-regex", ctest)
            self.assertIn("^marketforge.cuda_public_headers$", ctest)

            with (
                mock.patch(
                    "tools.modal.cuda_modal_app._run",
                    return_value={"output": "", "exit_status": 0},
                ) as run,
                mock.patch(
                    "tools.modal.cuda_modal_app._compile_flags",
                    return_value=(["-std=c++20 compute_89"], ["-lcudart"]),
                ),
            ):
                _configure_cuda(Path("/source"), build, run_gpu_tests=True)
            commands = [call.args[0] for call in run.call_args_list]
            ctest = next(command for command in commands if command[0] == "ctest")
            self.assertNotIn("--tests-regex", ctest)

    def test_compile_result_rejects_bool_duration_and_mixed_architecture(self) -> None:
        source = SimpleNamespace(commit=COMMIT, sha256=SOURCE_HASH)
        result = compile_result()
        result["wall_seconds"] = True
        with self.assertRaises(RuntimeError):
            _validate_compile_result(
                result,
                source_bundle=source,
                expected_call_id="fc-compile",
            )
        result = compile_result()
        result["compile_flags"].append("compute_80 sm_80")
        with self.assertRaises(RuntimeError):
            _validate_compile_result(
                result,
                source_bundle=source,
                expected_call_id="fc-compile",
            )

    def test_compile_result_distinguishes_ninja_distribution_and_binary(
        self,
    ) -> None:
        for field, replacement in (
            ("ninja_distribution", "1.11.1.git.kitware.jobserver-1"),
            ("ninja_binary", "1.11.1.1"),
            ("ninja_distribution", True),
            ("ninja_binary", None),
        ):
            with self.subTest(field=field, replacement=replacement):
                result = compile_result()
                result["observed"][field] = replacement
                with self.assertRaisesRegex(RuntimeError, field):
                    _validate_compile_result(
                        result,
                        source_bundle=SimpleNamespace(
                            commit=COMMIT, sha256=SOURCE_HASH
                        ),
                        expected_call_id="fc-compile",
                    )

    def test_self_move_uses_indirection_not_direct_warning_pattern(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for relative in (
            "src/tools/cuda_probe.cpp",
            "tests/cuda/cuda_tests.cu",
        ):
            text = (root / relative).read_text(encoding="utf-8")
            self.assertNotIn("std::move(stream)", text)
            self.assertNotIn("std::move(second)", text)
            self.assertIn("indirect_move", text)

    def test_profiler_replaces_stale_capture_and_requires_real_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.nsys-rep"
            capture.write_bytes(b"stale")

            def run(command: list[str], **_: object) -> dict[str, object]:
                if command[1] == "profile":
                    self.assertFalse(capture.exists())
                    capture.write_bytes(b"fresh")
                    return {"exit_status": 0, "output": ""}
                return {
                    "exit_status": 0,
                    "output": "lifecycle_kernel,123.5",
                }

            with (
                mock.patch(
                    "tools.modal.cuda_modal_app.shutil.which",
                    return_value="/usr/bin/nsys",
                ),
                mock.patch(
                    "tools.modal.cuda_modal_app._tool_version",
                    return_value="2025.1",
                ),
                mock.patch(
                    "tools.modal.cuda_modal_app._run", side_effect=run
                ),
            ):
                result = _profiler_capability(
                    name="nsys",
                    command=["nsys", "profile", "probe"],
                    capture_candidates=[capture],
                    version_command=["nsys", "--version"],
                    verification_command=["nsys", "stats", str(capture)],
                    verify_output=_verify_nsys_output,
                )
            self.assertEqual(result["availability"], "available")
            self.assertFalse(capture.exists())

        self.assertIsNone(_verify_nsys_output("arbitrary nonempty capture"))
        self.assertIsNone(_verify_ncu_output("sm__cycles_elapsed.avg,n/a"))
        self.assertEqual(
            _verify_ncu_output("sm__cycles_elapsed.avg,987.5"),
            "sm__cycles_elapsed.avg=987.5",
        )


if __name__ == "__main__":
    unittest.main()
