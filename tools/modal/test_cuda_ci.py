from decimal import Decimal
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import tools.modal.cuda_ci as cuda_ci
from tools.modal.cuda_ci import (
    CHAIN_COST_CEILING_USD,
    COMPILE_COST_CEILING_USD,
    AuthorizationTicketStore,
    EvidenceCache,
    GATE_ID,
    GPU_COST_CEILING_USD,
    SourceBundle,
    TRIAL_COMPUTE_CEILING_USD,
    TRIAL_GPU_MINUTE_CEILING,
    TrialLedger,
    create_source_bundle,
    dry_run_manifest,
    extract_source_bundle,
    main,
    modal_executable_for,
    parse_cost,
    require_external_tokenizer,
    require_unchanged_source_bundle,
    run_ordered,
    run_local_gates,
    scan_forbidden_artifacts,
    strict_json_loads,
)

ROOT = Path(__file__).resolve().parents[2]


class RecordingDispatcher:
    def __init__(self, *, compile_result: str = "pass") -> None:
        self.calls: list[str] = []
        self.compile_result = compile_result

    def dispatch_compile(self) -> dict[str, object]:
        self.calls.append("cuda_compile")
        return {"result": self.compile_result}

    def dispatch_gpu(self) -> dict[str, object]:
        self.calls.append("gpu_smoke")
        return {"result": "pass"}


class CudaDispatchTests(unittest.TestCase):
    def _git_fixture(self, root: Path) -> None:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(
            ["git", "-C", str(root), "config", "user.name", "PR6 Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "config",
                "user.email",
                "pr6@example.invalid",
            ],
            check=True,
        )

    def _resign_ledger(
        self, ledger: TrialLedger, value: dict[str, object]
    ) -> None:
        payload = {
            "schema_version": value["schema_version"],
            "reservations": value["reservations"],
        }
        value["signature"] = TrialLedger._signature(
            payload, ledger.key_path.read_bytes()
        )
        ledger.path.write_text(json.dumps(value), encoding="utf-8")

    def test_locked_costs_include_the_combined_chain(self) -> None:
        self.assertEqual(COMPILE_COST_CEILING_USD, Decimal("0.0210480"))
        self.assertEqual(GPU_COST_CEILING_USD, Decimal("0.2313720"))
        self.assertEqual(CHAIN_COST_CEILING_USD, Decimal("0.2524200"))
        self.assertEqual(TRIAL_COMPUTE_CEILING_USD, Decimal("2.00"))
        self.assertEqual(TRIAL_GPU_MINUTE_CEILING, Decimal("120"))

    def test_dry_run_has_complete_resources_and_dispatches_nothing(self) -> None:
        plan = dry_run_manifest(Decimal("0"))
        self.assertFalse(plan["dispatched"])
        self.assertEqual(
            [stage["name"] for stage in plan["ordered_stages"]],
            ["cuda_compile", "gpu_smoke"],
        )
        self.assertEqual(plan["ordered_stages"][1]["gpu"], "L4")
        self.assertEqual(plan["ordered_stages"][0]["memory_gib"], 4)
        self.assertEqual(plan["budget"]["trial_compute_ceiling_usd"], "2.00")
        self.assertEqual(plan["budget"]["trial_gpu_minute_ceiling"], "120")

    def test_combined_cost_equality_with_soft_cap_is_allowed(self) -> None:
        month_to_date = Decimal("24") - CHAIN_COST_CEILING_USD
        dry_run_manifest(month_to_date)

    def test_one_cent_over_soft_cap_is_rejected_before_dispatch(self) -> None:
        dispatcher = RecordingDispatcher()
        month_to_date = Decimal("24") - CHAIN_COST_CEILING_USD + Decimal("0.01")
        with self.assertRaises(RuntimeError):
            run_ordered(
                month_to_date_usd=month_to_date,
                dispatcher=dispatcher,
                local_gates_passed=True,
            )
        self.assertEqual(dispatcher.calls, [])

    def test_compile_failure_prevents_gpu_dispatch(self) -> None:
        dispatcher = RecordingDispatcher(compile_result="fail")
        with self.assertRaises(RuntimeError):
            run_ordered(
                month_to_date_usd=Decimal("0"),
                dispatcher=dispatcher,
                local_gates_passed=True,
            )
        self.assertEqual(dispatcher.calls, ["cuda_compile"])

    def test_local_failure_prevents_all_dispatch(self) -> None:
        dispatcher = RecordingDispatcher()
        with self.assertRaises(RuntimeError):
            run_ordered(
                month_to_date_usd=Decimal("0"),
                dispatcher=dispatcher,
                local_gates_passed=False,
            )
        self.assertEqual(dispatcher.calls, [])

    def test_success_dispatches_in_order(self) -> None:
        dispatcher = RecordingDispatcher()
        result = run_ordered(
            month_to_date_usd=Decimal("0"),
            dispatcher=dispatcher,
            local_gates_passed=True,
        )
        self.assertTrue(result["dispatched"])
        self.assertEqual(dispatcher.calls, ["cuda_compile", "gpu_smoke"])

    def test_invalid_costs_are_rejected(self) -> None:
        for value in (
            "-1",
            "NaN",
            "sNaN",
            "-NaN",
            "Infinity",
            "-Infinity",
            "abc",
            "",
            " 1",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_cost(value)

    def test_strict_json_rejects_extensions_and_duplicate_keys(self) -> None:
        for text in ('{"cost": NaN}', '{"cost": 1, "cost": 2}'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    strict_json_loads(text)

    def test_evidence_cache_is_exact_and_accepts_only_valid_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = EvidenceCache(Path(directory))
            commit = "a" * 40
            manifest = {
                "schema_version": 1,
                "result": "pass",
                **{f"field_{index}": index for index in range(10)},
            }
            self.assertIsNone(
                cache.load(candidate_commit=commit, gate_id=GATE_ID)
            )
            with mock.patch(
                "tools.modal.cuda_evidence.validate_manifest"
            ) as validator:
                path = cache.store(
                    candidate_commit=commit,
                    gate_id=GATE_ID,
                    manifest=manifest,
                    lock={},
                    expected_source_bundle_sha256="b" * 64,
                    expected_dependency_lock_sha256="c" * 64,
                )
                validator.assert_called_once()
            self.assertEqual(
                cache.load(candidate_commit=commit, gate_id=GATE_ID),
                manifest,
            )
            self.assertIsNone(
                cache.load(candidate_commit="b" * 40, gate_id=GATE_ID)
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            record["accepted"] = False
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                cache.load(candidate_commit=commit, gate_id=GATE_ID)
            with self.assertRaises(ValueError):
                cache.store(
                    candidate_commit=commit,
                    gate_id=GATE_ID,
                    manifest={"schema_version": 1, "result": "fail"},
                    lock={},
                    expected_source_bundle_sha256="b" * 64,
                    expected_dependency_lock_sha256="c" * 64,
                )
            with self.assertRaises(ValueError):
                cache.store(
                    candidate_commit=commit,
                    gate_id=GATE_ID,
                    manifest={"schema_version": True, "result": "pass"},
                    lock={},
                    expected_source_bundle_sha256="b" * 64,
                    expected_dependency_lock_sha256="c" * 64,
                )

    def test_source_bundle_requires_clean_immutable_git_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._git_fixture(root)
            (root / "tracked.txt").write_text("locked\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "fixture"],
                check=True,
            )
            bundle = create_source_bundle(root)
            self.assertEqual(len(bundle.commit), 40)
            self.assertEqual(len(bundle.sha256), 64)
            self.assertTrue(bundle.content)
            (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                create_source_bundle(root)

    def test_source_bundle_is_extracted_and_toc_tou_changes_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            root.mkdir()
            self._git_fixture(root)
            tracked = root / "tracked.txt"
            tracked.write_text("first\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "first"],
                check=True,
            )
            bundle = create_source_bundle(root)
            extracted = Path(directory) / "extracted"
            extract_source_bundle(bundle, extracted)
            self.assertEqual(
                (extracted / "tracked.txt").read_text(encoding="utf-8"),
                "first\n",
            )
            tracked.write_text("second\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.txt"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "second"],
                check=True,
            )
            with self.assertRaises(RuntimeError):
                require_unchanged_source_bundle(root, bundle)

    def test_launcher_preserves_virtual_environment_modal_path(self) -> None:
        python = Path("/repo/.venv/bin/python")
        self.assertEqual(
            modal_executable_for(python),
            Path("/repo/.venv/bin/modal"),
        )
        with (
            mock.patch(
                "tools.modal.cuda_ci.sys.executable",
                "/repo/.venv/bin/python",
            ),
            mock.patch(
                "tools.modal.cuda_ci._launch", return_value=0
            ) as launch,
        ):
            self.assertEqual(
                main(
                    [
                        "--month-to-date-usd",
                        "0",
                        "--tokenizer-json",
                        "/external/tokenizer.json",
                        "--launch",
                    ]
                ),
                0,
            )
        self.assertEqual(
            launch.call_args.kwargs["python_executable"],
            Path("/repo/.venv/bin/python"),
        )

    def test_launch_gates_the_archive_and_rechecks_source_before_cache(
        self,
    ) -> None:
        events: list[str] = []
        bundle = SourceBundle(
            commit="a" * 40,
            sha256="b" * 64,
            content=b"archive",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            lock_path = root / "tools/modal/cuda-toolchain-lock.json"
            lock_path.parent.mkdir(parents=True)
            lock_path.write_text("{}", encoding="utf-8")
            cache = mock.Mock()
            cache.load.return_value = {}
            with (
                mock.patch.object(
                    cuda_ci,
                    "scan_forbidden_artifacts",
                    side_effect=lambda unused: events.append("scan"),
                ),
                mock.patch.object(
                    cuda_ci,
                    "require_external_tokenizer",
                    side_effect=lambda unused_root, tokenizer: (
                        events.append("tokenizer") or tokenizer
                    ),
                ),
                mock.patch.object(
                    cuda_ci,
                    "create_source_bundle",
                    side_effect=lambda unused: (
                        events.append("bundle") or bundle
                    ),
                ),
                mock.patch.object(
                    cuda_ci,
                    "run_local_gates",
                    side_effect=lambda unused_bundle, unused_python, unused_tokenizer: (
                        events.append("gates")
                    ),
                ),
                mock.patch.object(
                    cuda_ci,
                    "require_unchanged_source_bundle",
                    side_effect=lambda unused_root, unused_bundle: (
                        events.append("unchanged")
                    ),
                ),
                mock.patch.object(
                    cuda_ci,
                    "source_bundle_member_sha256",
                    return_value="c" * 64,
                ),
                mock.patch.object(
                    cuda_ci,
                    "source_bundle_member_content",
                    return_value=b"{}",
                ),
                mock.patch.object(
                    cuda_ci, "EvidenceCache", return_value=cache
                ),
                mock.patch(
                    "tools.modal.cuda_evidence.validate_manifest"
                ),
            ):
                self.assertEqual(
                    cuda_ci._launch(
                        project_root=root,
                        python_executable=Path("/repo/.venv/bin/python"),
                        tokenizer_json=Path("/external/tokenizer.json"),
                        month_to_date=Decimal("0"),
                    ),
                    0,
                )
        self.assertEqual(
            events,
            ["scan", "tokenizer", "bundle", "gates", "scan", "unchanged"],
        )

    def test_launch_requires_an_external_tokenizer_but_dry_run_does_not(
        self,
    ) -> None:
        self.assertEqual(
            main(["--month-to-date-usd", "0", "--dry-run"]),
            0,
        )
        with self.assertRaisesRegex(SystemExit, "tokenizer-json"):
            main(["--month-to-date-usd", "0", "--launch"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            root.mkdir()
            internal = root / "tokenizer.json"
            internal.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "outside"):
                require_external_tokenizer(root, internal)
            external = Path(directory) / "external-tokenizer.json"
            external.write_text("{}", encoding="utf-8")
            self.assertEqual(
                require_external_tokenizer(root, external),
                external.resolve(),
            )

    def test_missing_tokenizer_stops_local_gates_before_subprocesses(self) -> None:
        bundle = SourceBundle(
            commit="a" * 40,
            sha256="b" * 64,
            content=b"not inspected before tokenizer validation",
        )
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "tokenizer.json"
            with mock.patch("tools.modal.cuda_ci.subprocess.run") as runner:
                with self.assertRaisesRegex(RuntimeError, "mandatory"):
                    run_local_gates(bundle, Path(sys.executable), missing)
                runner.assert_not_called()

    def test_forbidden_artifact_scan_rejects_builds_bytecode_and_models(
        self,
    ) -> None:
        names = (
            "build",
            "__pycache__",
            "capture.nsys-rep",
            "models/model.safetensors",
            "weights.safetensors",
            "stray/CMakeCache.txt",
            "tokenizer.json",
        )
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / name
                if target.suffix:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"artifact")
                else:
                    target.mkdir(parents=True)
                with self.assertRaisesRegex(
                    RuntimeError, "forbidden worktree artifacts"
                ):
                    scan_forbidden_artifacts(root)

    def test_trial_ledger_counts_failed_attempts_and_blocks_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = TrialLedger(Path(directory))
            commit = "a" * 40
            first = ledger.reserve(
                candidate_commit=commit, gate_id=GATE_ID
            )
            with self.assertRaises(RuntimeError):
                ledger.reserve(candidate_commit=commit, gate_id=GATE_ID)
            ledger.finish(first, passed=False)
            for suffix in ("b", "c", "d", "e", "f", "0"):
                reservation = ledger.reserve(
                    candidate_commit=suffix * 40, gate_id=GATE_ID
                )
                ledger.finish(reservation, passed=True)
                with self.assertRaisesRegex(RuntimeError, "passed permanently"):
                    ledger.reserve(
                        candidate_commit=suffix * 40,
                        gate_id=GATE_ID,
                    )
            with self.assertRaisesRegex(RuntimeError, r"\$2\.00"):
                ledger.reserve(
                    candidate_commit="1" * 40, gate_id=GATE_ID
                )

    def test_copied_four_entry_ledger_allows_three_more_and_blocks_eighth(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as original_directory,
            tempfile.TemporaryDirectory() as copied_directory,
        ):
            original = TrialLedger(Path(original_directory))
            for suffix in ("a", "b", "c", "d"):
                reservation = original.reserve(
                    candidate_commit=suffix * 40,
                    gate_id=GATE_ID,
                )
                original.finish(reservation, passed=False)
            original_value = json.loads(
                original.path.read_text(encoding="utf-8")
            )

            copied_root = Path(copied_directory)
            shutil.copy2(original.path, copied_root / original.path.name)
            shutil.copy2(original.key_path, copied_root / original.key_path.name)
            copied = TrialLedger(copied_root)
            for suffix in ("e", "f", "0"):
                reservation = copied.reserve(
                    candidate_commit=suffix * 40,
                    gate_id=GATE_ID,
                )
                copied.finish(reservation, passed=False)

            copied_value = json.loads(
                copied.path.read_text(encoding="utf-8")
            )
            reservations = copied_value["reservations"]
            self.assertEqual(
                reservations[:4],
                original_value["reservations"],
            )
            self.assertEqual(len(reservations), 7)
            reserved_cost = sum(
                (
                    parse_cost(item["reserved_cost_usd"])
                    for item in reservations
                ),
                Decimal("0"),
            )
            reserved_minutes = sum(
                (
                    parse_cost(item["reserved_gpu_minutes"])
                    for item in reservations
                ),
                Decimal("0"),
            )
            self.assertEqual(reserved_cost, Decimal("1.766940"))
            self.assertEqual(reserved_minutes, Decimal("105"))
            self.assertEqual(
                reserved_cost + CHAIN_COST_CEILING_USD,
                Decimal("2.019360"),
            )
            self.assertEqual(
                reserved_minutes + Decimal("15"),
                Decimal("120"),
            )
            with self.assertRaisesRegex(RuntimeError, r"\$2\.00"):
                copied.reserve(
                    candidate_commit="1" * 40,
                    gate_id=GATE_ID,
                )
            self.assertEqual(
                json.loads(copied.path.read_text(encoding="utf-8")),
                copied_value,
            )

    def test_trial_ledger_rejects_tampered_persisted_reservations(self) -> None:
        mutations = {
            "reservation_id": "A" * 32,
            "candidate_commit": True,
            "gate_id": "pr6-other-gate",
            "reserved_cost_usd": "0.25242",
            "reserved_gpu_minutes": "15.0",
            "status": True,
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                ledger = TrialLedger(Path(directory))
                ledger.reserve(
                    candidate_commit="a" * 40,
                    gate_id=GATE_ID,
                )
                value = json.loads(ledger.path.read_text(encoding="utf-8"))
                value["reservations"][0][field] = replacement
                self._resign_ledger(ledger, value)
                with self.assertRaises(ValueError):
                    ledger.reserve(
                        candidate_commit="b" * 40,
                        gate_id=GATE_ID,
                    )

    def test_trial_ledger_rejects_duplicate_reservation_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = TrialLedger(Path(directory))
            ledger.reserve(candidate_commit="a" * 40, gate_id=GATE_ID)
            value = json.loads(ledger.path.read_text(encoding="utf-8"))
            duplicate = dict(value["reservations"][0])
            duplicate["candidate_commit"] = "b" * 40
            duplicate["status"] = "failed"
            value["reservations"].append(duplicate)
            self._resign_ledger(ledger, value)
            with self.assertRaises(ValueError):
                ledger.reserve(candidate_commit="c" * 40, gate_id=GATE_ID)

    def test_trial_ledger_hmac_rejects_valid_shaped_mutations(self) -> None:
        mutations = (
            ("status", "failed"),
            ("candidate_commit", "b" * 40),
        )
        for field, replacement in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                ledger = TrialLedger(Path(directory))
                reservation = ledger.reserve(
                    candidate_commit="a" * 40,
                    gate_id=GATE_ID,
                )
                ledger.finish(reservation, passed=True)
                value = json.loads(ledger.path.read_text(encoding="utf-8"))
                value["reservations"][0][field] = replacement
                ledger.path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "authentication"):
                    ledger.reserve(
                        candidate_commit="a" * 40,
                        gate_id=GATE_ID,
                    )

    def test_trial_ledger_rejects_one_sided_state_loss(self) -> None:
        for missing in ("ledger", "key"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                ledger = TrialLedger(Path(directory))
                ledger.reserve(
                    candidate_commit="a" * 40,
                    gate_id=GATE_ID,
                )
                self.assertEqual(ledger.key_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(ledger.path.stat().st_mode & 0o777, 0o600)
                target = (
                    ledger.path if missing == "ledger" else ledger.key_path
                )
                target.unlink()
                with self.assertRaisesRegex(
                    RuntimeError, "state loss or tampering"
                ):
                    ledger.reserve(
                        candidate_commit="b" * 40,
                        gate_id=GATE_ID,
                    )

    def test_passed_commit_stays_blocked_without_evidence_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = TrialLedger(Path(directory))
            commit = "a" * 40
            reservation = ledger.reserve(
                candidate_commit=commit,
                gate_id=GATE_ID,
            )
            ledger.finish(reservation, passed=True)
            self.assertFalse(
                (Path(directory) / "accepted-evidence.json").exists()
            )
            with self.assertRaisesRegex(RuntimeError, "passed permanently"):
                ledger.reserve(
                    candidate_commit=commit,
                    gate_id=GATE_ID,
                )

    def test_authorization_ticket_is_source_bound_and_one_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = SourceBundle(
                commit="a" * 40,
                sha256="b" * 64,
                content=b"source",
            )
            store = AuthorizationTicketStore(root)
            path = store.issue(
                source_bundle=bundle,
                dependency_lock_sha256="c" * 64,
                month_to_date_usd=Decimal("0.1"),
                reservation_id="reservation",
            )
            ticket = store.consume(path)
            self.assertEqual(ticket["candidate_commit"], bundle.commit)
            self.assertEqual(ticket["source_bundle_sha256"], bundle.sha256)
            with self.assertRaises(RuntimeError):
                store.consume(path)
            tampered = store.issue(
                source_bundle=bundle,
                dependency_lock_sha256="c" * 64,
                month_to_date_usd=Decimal("0.1"),
                reservation_id="reservation-two",
            )
            record = json.loads(tampered.read_text(encoding="utf-8"))
            record["source_bundle_sha256"] = "d" * 64
            tampered.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                store.consume(tampered)

    def test_pure_dry_run_does_not_import_modal(self) -> None:
        script = (
            "import sys;"
            "from tools.modal.cuda_ci import main;"
            "result=main(['--month-to-date-usd','0','--dry-run']);"
            "assert result == 0;"
            "assert 'modal' not in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
