from decimal import Decimal
import hashlib
from importlib.metadata import PackageNotFoundError
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from tools.modal.cuda_ci import EvidenceCache, GATE_ID
from tools.modal.cuda_evidence import validate_manifest
from tools.modal.cuda_modal_app import (
    GPU_BUILD,
    LOCK,
    SANITIZER_IMAGE_INSTALL_COMMAND,
    _configure_cuda,
    _installed_distribution_version,
    _manifest,
    _observe_compute_sanitizer,
    _parse_production_sanitizer,
    _parse_sanitizer_canary,
    _profiler_capability,
    _read_cuda_toolkit_version,
    _require_exact_executable,
    _run_compute_sanitizer_gate,
    _validate_compile_result,
    _validate_gpu_result,
    _verify_ncu_output,
    _verify_nsys_output,
)

COMMIT = "c" * 40
SOURCE_HASH = "a" * 64
DEPENDENCY_HASH = "b" * 64
SANITIZER_PATH = LOCK["compute_sanitizer"]["executable_path"]


def sanitizer_command(target: str, arguments: list[str]) -> list[str]:
    return [
        SANITIZER_PATH,
        "--tool",
        "memcheck",
        "--leak-check",
        "full",
        "--error-exitcode=97",
        target,
        *arguments,
    ]


def sanitizer_evidence() -> dict[str, object]:
    invalid_command = sanitizer_command(
        str(GPU_BUILD / "marketforge_cuda_sanitizer_canary"),
        ["--mode", "invalid-global-write"],
    )
    leak_command = sanitizer_command(
        str(GPU_BUILD / "marketforge_cuda_sanitizer_canary"),
        ["--mode", "device-leak"],
    )
    production_command = sanitizer_command(
        str(GPU_BUILD / "marketforge_cuda_probe"),
        ["--repetitions", "100"],
    )
    return {
        "identity": dict(LOCK["compute_sanitizer"]),
        "canaries": {
            "invalid_global_write": {
                "mode": "invalid-global-write",
                "command": invalid_command,
                "result": "detected",
                "exit_status": 97,
                "error_summary_count": 1,
                "evidence": "invalid __global__ write detected",
                "output_sha256": "1" * 64,
            },
            "device_leak": {
                "mode": "device-leak",
                "command": leak_command,
                "result": "detected",
                "exit_status": 97,
                "error_summary_count": 1,
                "evidence": "256-byte device leak detected",
                "output_sha256": "2" * 64,
            },
        },
        "production": {
            "command": production_command,
            "result": "pass",
            "exit_status": 0,
            "error_summary_count": 0,
            "evidence": "ERROR SUMMARY: 0 errors",
            "output_sha256": "3" * 64,
        },
    }


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
        "compute_sanitizer": sanitizer_evidence(),
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
    def test_import_has_no_process_or_network_side_effects(self) -> None:
        script = """
import socket
import subprocess
import urllib.request
from unittest import mock

def blocked(*args, **kwargs):
    raise AssertionError("import attempted a process or network side effect")

with (
    mock.patch.object(subprocess, "run", side_effect=blocked),
    mock.patch.object(urllib.request, "urlopen", side_effect=blocked),
    mock.patch.object(socket, "create_connection", side_effect=blocked),
):
    import tools.modal.cuda_modal_app

print("import-ok")
"""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertEqual(completed.stdout.strip(), "import-ok")

    def test_image_install_command_is_exact_and_preserves_cuda_12(self) -> None:
        sanitizer = LOCK["compute_sanitizer"]
        command = SANITIZER_IMAGE_INSTALL_COMMAND
        for value in (
            f"{sanitizer['package_name']}={sanitizer['package_version']}",
            sanitizer["package_sha256"],
            sanitizer["executable_path"],
            sanitizer["executable_sha256"],
            str(sanitizer["executable_size_bytes"]),
            "apt-get download",
            "sha256sum --check --strict",
            "dpkg --install",
            'nvcc_before="$(/usr/local/cuda/bin/nvcc --version)"',
            'test "$(/usr/local/cuda/bin/nvcc --version)" = "${nvcc_before}"',
        ):
            with self.subTest(value=value):
                self.assertIn(value, command)
        self.assertIn("readlink -f /usr/local/cuda", command)

    def test_sanitizer_identity_observes_exact_package_and_binary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "compute-sanitizer"
            docs = root / "docs"
            docs.mkdir(parents=True)
            executable = root / "compute-sanitizer"
            executable.write_bytes(b"locked executable")
            executable.chmod(0o755)
            version_file = docs / "VERSION"
            version_file.write_text("13.0.1\n", encoding="utf-8")
            locked = {
                **LOCK["compute_sanitizer"],
                "executable_path": str(executable),
                "executable_size_bytes": executable.stat().st_size,
                "executable_sha256": hashlib.sha256(
                    executable.read_bytes()
                ).hexdigest(),
            }

            def run(command: list[str], **_: object) -> dict[str, object]:
                if command[:2] == ["dpkg-query", "--listfiles"]:
                    output = f"{executable}\n{version_file}\n"
                elif command[:3] == [
                    "dpkg-query",
                    "--show",
                    "--showformat=${Version}",
                ]:
                    output = "13.0.85-1\n"
                elif command[:3] == [
                    "dpkg-query",
                    "--show",
                    "--showformat=${Architecture}",
                ]:
                    output = "amd64\n"
                elif command[:3] == [
                    "dpkg-query",
                    "--show",
                    "--showformat=${Status}",
                ]:
                    output = "install ok installed\n"
                elif command == [str(executable), "--version"]:
                    output = (
                        "NVIDIA (R) Compute Sanitizer\n"
                        "Copyright (c) NVIDIA Corporation\n"
                        "Version 2025.3.1.0 (build 36400806) "
                        "(public-release)\n"
                    )
                else:
                    self.fail(f"unexpected command: {command}")
                return {"output": output, "exit_status": 0}

            with (
                mock.patch(
                    "tools.modal.cuda_modal_app.SANITIZER_LOCK", locked
                ),
                mock.patch(
                    "tools.modal.cuda_modal_app._run", side_effect=run
                ),
            ):
                self.assertEqual(_observe_compute_sanitizer(), locked)

            def duplicate_ownership(
                command: list[str], **kwargs: object
            ) -> dict[str, object]:
                result = run(command, **kwargs)
                if command[:2] == ["dpkg-query", "--listfiles"]:
                    result["output"] = (
                        f"{executable}\n{executable}\n{version_file}\n"
                    )
                return result

            with (
                mock.patch(
                    "tools.modal.cuda_modal_app.SANITIZER_LOCK", locked
                ),
                mock.patch(
                    "tools.modal.cuda_modal_app._run",
                    side_effect=duplicate_ownership,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "uniquely package-owned"
                ):
                    _observe_compute_sanitizer()

    def test_sanitizer_version_rejects_duplicate_or_malformed_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "compute-sanitizer"
            docs = root / "docs"
            docs.mkdir(parents=True)
            executable = root / "compute-sanitizer"
            executable.write_bytes(b"locked executable")
            executable.chmod(0o755)
            version_file = docs / "VERSION"
            version_file.write_text("13.0.1\n", encoding="utf-8")
            locked = {
                **LOCK["compute_sanitizer"],
                "executable_path": str(executable),
                "executable_size_bytes": executable.stat().st_size,
                "executable_sha256": hashlib.sha256(
                    executable.read_bytes()
                ).hexdigest(),
            }
            valid_header = "NVIDIA (R) Compute Sanitizer\n"
            valid_version = (
                "Version 2025.3.1.0 (build 36400806) (public-release)\n"
            )
            outputs = (
                valid_version,
                valid_header + valid_version + valid_version,
                valid_header + valid_header + valid_version,
                valid_header
                + "Version 13.0.1 (build 36400806) (public-release)\n",
            )

            for version_output in outputs:
                with self.subTest(version_output=version_output):
                    def run(
                        command: list[str], **_: object
                    ) -> dict[str, object]:
                        if command[:2] == ["dpkg-query", "--listfiles"]:
                            output = f"{executable}\n{version_file}\n"
                        elif command[:3] == [
                            "dpkg-query",
                            "--show",
                            "--showformat=${Version}",
                        ]:
                            output = "13.0.85-1\n"
                        elif command[:3] == [
                            "dpkg-query",
                            "--show",
                            "--showformat=${Architecture}",
                        ]:
                            output = "amd64\n"
                        elif command[:3] == [
                            "dpkg-query",
                            "--show",
                            "--showformat=${Status}",
                        ]:
                            output = "install ok installed\n"
                        else:
                            output = version_output
                        return {"output": output, "exit_status": 0}

                    with (
                        mock.patch(
                            "tools.modal.cuda_modal_app.SANITIZER_LOCK",
                            locked,
                        ),
                        mock.patch(
                            "tools.modal.cuda_modal_app._run",
                            side_effect=run,
                        ),
                    ):
                        with self.assertRaises(RuntimeError):
                            _observe_compute_sanitizer()

    def test_canary_parsers_require_real_tool_diagnostics(self) -> None:
        canary = str(GPU_BUILD / "marketforge_cuda_sanitizer_canary")
        cases = {
            "invalid-global-write": (
                "MARKETFORGE_SANITIZER_CANARY "
                "mode=invalid-global-write bytes=4\n"
                "========= Invalid __global__ write of size 4 bytes\n"
                "========= at invalid_global_write_kernel(int*)\n"
                "========= ERROR SUMMARY: 1 error\n"
            ),
            "device-leak": (
                "MARKETFORGE_SANITIZER_CANARY mode=device-leak bytes=256\n"
                "========= Leaked 256 bytes at 0x1234\n"
                "========= ERROR SUMMARY: 1 error\n"
            ),
        }
        for mode, output in cases.items():
            with self.subTest(mode=mode):
                command = sanitizer_command(
                    canary, ["--mode", mode]
                )
                evidence = _parse_sanitizer_canary(
                    mode,
                    command,
                    {"exit_status": 97, "output": output},
                )
                self.assertEqual(evidence["result"], "detected")
                self.assertEqual(
                    evidence["output_sha256"],
                    hashlib.sha256(output.encode("utf-8")).hexdigest(),
                )

        valid = cases["invalid-global-write"]
        for replacement in (
            valid.replace(
                "Invalid __global__ write of size 4 bytes",
                "Device not supported",
            ),
            valid + "========= Insufficient permissions\n",
            valid + "========= Internal error\n",
            valid + "========= Error 999\n",
            valid + "========= Errors 999\n",
            valid.replace(
                "========= Invalid __global__ write",
                "= Invalid __global__ write",
            ),
            (
                "MARKETFORGE_SANITIZER_CANARY "
                "mode=invalid-global-write bytes=4\n"
                "========= ERROR SUMMARY: 1 error\n"
            ),
        ):
            with self.subTest(replacement=replacement):
                with self.assertRaises(RuntimeError):
                    _parse_sanitizer_canary(
                        "invalid-global-write",
                        sanitizer_command(
                            canary,
                            ["--mode", "invalid-global-write"],
                        ),
                        {"exit_status": 97, "output": replacement},
                    )

    def test_production_parser_requires_clean_instrumented_probe(self) -> None:
        command = sanitizer_command(
            str(GPU_BUILD / "marketforge_cuda_probe"),
            ["--repetitions", "100"],
        )
        output = (
            '{"schema_version":1,"result":"pass",'
            '"known_answer_lengths":[0,1,255,256,257,1025],'
            '"sentinels":"pass","lifecycle_repetitions":100}\n'
            "========= ERROR SUMMARY: 0 errors\n"
        )
        probe, evidence = _parse_production_sanitizer(
            command, {"exit_status": 0, "output": output}
        )
        self.assertEqual(probe["lifecycle_repetitions"], 100)
        self.assertEqual(evidence["error_summary_count"], 0)
        with self.assertRaises(RuntimeError):
            _parse_production_sanitizer(
                command,
                {
                    "exit_status": 0,
                    "output": output + "========= Device not supported\n",
                },
            )

    def test_sanitizer_gate_short_circuits_after_first_canary_failure(
        self,
    ) -> None:
        with (
            mock.patch(
                "tools.modal.cuda_modal_app._observe_compute_sanitizer",
                return_value=dict(LOCK["compute_sanitizer"]),
            ),
            mock.patch(
                "tools.modal.cuda_modal_app._run_sanitizer_canary",
                side_effect=RuntimeError("canary failed"),
            ) as canary,
            mock.patch("tools.modal.cuda_modal_app._run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "canary failed"):
                _run_compute_sanitizer_gate(
                    GPU_BUILD / "marketforge_cuda_probe",
                    GPU_BUILD / "marketforge_cuda_sanitizer_canary",
                )
        self.assertEqual(canary.call_count, 1)
        run.assert_not_called()

    def test_sanitizer_gate_short_circuits_after_second_canary_failure(
        self,
    ) -> None:
        invalid_write = sanitizer_evidence()["canaries"][
            "invalid_global_write"
        ]
        with (
            mock.patch(
                "tools.modal.cuda_modal_app._observe_compute_sanitizer",
                return_value=dict(LOCK["compute_sanitizer"]),
            ),
            mock.patch(
                "tools.modal.cuda_modal_app._run_sanitizer_canary",
                side_effect=[
                    invalid_write,
                    RuntimeError("leak canary failed"),
                ],
            ) as canary,
            mock.patch("tools.modal.cuda_modal_app._run") as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "leak canary failed"):
                _run_compute_sanitizer_gate(
                    GPU_BUILD / "marketforge_cuda_probe",
                    GPU_BUILD / "marketforge_cuda_sanitizer_canary",
                )
        self.assertEqual(
            [call.args[1] for call in canary.call_args_list],
            ["invalid-global-write", "device-leak"],
        )
        run.assert_not_called()

    def test_sanitizer_gate_runs_identity_canaries_and_production_in_order(
        self,
    ) -> None:
        events: list[object] = []
        fixture = sanitizer_evidence()

        def observe() -> dict[str, object]:
            events.append("identity")
            return dict(LOCK["compute_sanitizer"])

        def run_canary(
            executable: Path, mode: str
        ) -> dict[str, object]:
            events.append((mode, executable))
            key = (
                "invalid_global_write"
                if mode == "invalid-global-write"
                else "device_leak"
            )
            return fixture["canaries"][key]

        production_output = (
            '{"schema_version":1,"result":"pass",'
            '"known_answer_lengths":[0,1,255,256,257,1025],'
            '"sentinels":"pass","lifecycle_repetitions":100}\n'
            "========= ERROR SUMMARY: 0 errors\n"
        )

        def run(
            command: list[str], **_: object
        ) -> dict[str, object]:
            events.append(("production", command))
            return {"exit_status": 0, "output": production_output}

        with (
            mock.patch(
                "tools.modal.cuda_modal_app._observe_compute_sanitizer",
                side_effect=observe,
            ),
            mock.patch(
                "tools.modal.cuda_modal_app._run_sanitizer_canary",
                side_effect=run_canary,
            ),
            mock.patch(
                "tools.modal.cuda_modal_app._run",
                side_effect=run,
            ),
        ):
            probe, evidence = _run_compute_sanitizer_gate(
                GPU_BUILD / "marketforge_cuda_probe",
                GPU_BUILD / "marketforge_cuda_sanitizer_canary",
            )

        self.assertEqual(probe["lifecycle_repetitions"], 100)
        self.assertEqual(evidence["production"]["result"], "pass")
        self.assertEqual(
            events,
            [
                "identity",
                (
                    "invalid-global-write",
                    GPU_BUILD / "marketforge_cuda_sanitizer_canary",
                ),
                (
                    "device-leak",
                    GPU_BUILD / "marketforge_cuda_sanitizer_canary",
                ),
                (
                    "production",
                    sanitizer_command(
                        str(GPU_BUILD / "marketforge_cuda_probe"),
                        ["--repetitions", "100"],
                    ),
                ),
            ],
        )

    def test_runtime_targets_must_be_exact_regular_executables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "probe"
            executable.write_bytes(b"executable")
            executable.chmod(0o755)
            _require_exact_executable(executable, executable)

            with self.assertRaises(RuntimeError):
                _require_exact_executable(executable, root / "other")

            symlink = root / "probe-link"
            symlink.symlink_to(executable)
            with self.assertRaises(RuntimeError):
                _require_exact_executable(symlink, symlink)

            non_executable = root / "not-executable"
            non_executable.write_bytes(b"not executable")
            non_executable.chmod(0o644)
            with self.assertRaises(RuntimeError):
                _require_exact_executable(non_executable, non_executable)

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

    def test_gpu_result_requires_versioned_sanitizer_capability(self) -> None:
        source = SimpleNamespace(commit=COMMIT, sha256=SOURCE_HASH)
        _validate_gpu_result(
            gpu_result(),
            source_bundle=source,
            expected_call_id="fc-gpu",
            compile_call_id="fc-compile",
        )

        identity_mutations = (
            ("package_version", "13.0.1"),
            ("toolkit_docs_release", "13.0.85-1"),
            ("executable_version", "13.0.1"),
            ("executable_build", True),
            ("release_channel", None),
            ("package_sha256", "A" * 64),
            ("executable_sha256", "f" * 63),
            ("executable_size_bytes", True),
            ("executable_path", "compute-sanitizer"),
            ("package_status", "not-installed"),
        )
        for field, replacement in identity_mutations:
            with self.subTest(field=field, replacement=replacement):
                result = gpu_result()
                result["compute_sanitizer"]["identity"][field] = replacement
                with self.assertRaisesRegex(RuntimeError, field):
                    _validate_gpu_result(
                        result,
                        source_bundle=source,
                        expected_call_id="fc-gpu",
                        compile_call_id="fc-compile",
                    )

        for first, second in (
            ("package_version", "executable_version"),
            ("package_sha256", "executable_sha256"),
        ):
            with self.subTest(swapped=(first, second)):
                result = gpu_result()
                identity = result["compute_sanitizer"]["identity"]
                identity[first], identity[second] = (
                    identity[second],
                    identity[first],
                )
                with self.assertRaises(RuntimeError):
                    _validate_gpu_result(
                        result,
                        source_bundle=source,
                        expected_call_id="fc-gpu",
                        compile_call_id="fc-compile",
                    )

        result = gpu_result()
        result["compute_sanitizer"] = {
            "command": ["compute-sanitizer"],
            "result": "pass",
            "exit_status": 0,
        }
        with self.assertRaisesRegex(RuntimeError, "schema"):
            _validate_gpu_result(
                result,
                source_bundle=source,
                expected_call_id="fc-gpu",
                compile_call_id="fc-compile",
            )

        mutations = (
            ("missing-canary", None),
            ("extra-canary", None),
            ("false-success", 0),
            ("bool-exit", True),
            ("bool-count", True),
            ("wrong-fault", "wrong"),
            ("wrong-path", "compute-sanitizer"),
            ("wrong-target", "/tmp/other/marketforge_cuda_probe"),
            ("duplicate-output", "2" * 64),
            ("production-output-reuse", "1" * 64),
            ("production-errors", 1),
        )
        for mutation, replacement in mutations:
            with self.subTest(mutation=mutation):
                result = gpu_result()
                sanitizer = result["compute_sanitizer"]
                if mutation == "missing-canary":
                    del sanitizer["canaries"]["device_leak"]
                elif mutation == "extra-canary":
                    sanitizer["canaries"]["extra"] = dict(
                        sanitizer["canaries"]["device_leak"]
                    )
                elif mutation == "false-success":
                    sanitizer["canaries"]["invalid_global_write"][
                        "exit_status"
                    ] = replacement
                elif mutation == "bool-exit":
                    sanitizer["production"]["exit_status"] = replacement
                elif mutation == "bool-count":
                    sanitizer["canaries"]["device_leak"][
                        "error_summary_count"
                    ] = replacement
                elif mutation == "wrong-fault":
                    sanitizer["canaries"]["device_leak"][
                        "evidence"
                    ] = replacement
                elif mutation == "wrong-path":
                    sanitizer["production"]["command"][0] = replacement
                elif mutation == "wrong-target":
                    sanitizer["production"]["command"][6] = replacement
                elif mutation == "duplicate-output":
                    sanitizer["canaries"]["invalid_global_write"][
                        "output_sha256"
                    ] = replacement
                elif mutation == "production-output-reuse":
                    sanitizer["production"]["output_sha256"] = replacement
                else:
                    sanitizer["production"]["error_summary_count"] = replacement
                with self.assertRaises(RuntimeError):
                    _validate_gpu_result(
                        result,
                        source_bundle=source,
                        expected_call_id="fc-gpu",
                        compile_call_id="fc-compile",
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

        result = compile_result()
        result["observed"]["ninja"] = result["observed"].pop(
            "ninja_distribution"
        )
        with self.assertRaisesRegex(RuntimeError, "toolchain evidence"):
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
