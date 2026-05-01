import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.runner import check_artifact, run
from src.tools import build, utc, validate_period


class HarnessTests(unittest.TestCase):
    def test_backtest_command_has_no_arbitrary_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, config = root / "tool", root / "config.json"
            binary.touch()
            config.write_text("{}")
            command, artifact = build(
                {
                    "tool": "backtest",
                    "args": {
                        "config": str(config),
                        "from": "2024-01-01T00:00:00Z",
                        "to": "2024-02-01T00:00:00Z",
                        "snapshot": "demo",
                        "cache": str(root),
                        "auth-token": "must-not-be-used",
                    },
                },
                root / "out",
                {"backtest": str(binary)},
            )
            self.assertNotIn("must-not-be-used", command)
            self.assertEqual(artifact.name, "result.dibate")
            self.assertTrue(artifact.parent.is_dir())

    def test_merge_cannot_include_own_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fold.dibate").write_bytes(b"fixture")
            binary = root / "tool"
            binary.touch()
            with self.assertRaises(ValueError):
                build(
                    {"tool": "diribbon", "args": {"input": str(root)}},
                    root / "out",
                    {"diribbon": str(binary)},
                )

    def test_statistics_uses_one_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, binary = root / "fixture.dibate", root / "tool"
            source.write_bytes(b"fixture")
            binary.touch()
            command, artifact = build(
                {"tool": "dibate2csv", "args": {"input": str(source)}},
                root / "out",
                {"dibate2csv": str(binary)},
            )
            self.assertEqual(command.count("--modes"), 1)
            self.assertEqual(artifact.name, "statistics.csv")

    def test_empty_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty"
            path.touch()
            with self.assertRaises(ValueError):
                check_artifact(path)

    def test_local_tools_need_permission(self):
        with self.assertRaises(PermissionError):
            run({"steps": [{"tool": "backtest"}]}, "unused")

    def test_timezone_required(self):
        with self.assertRaises(ValueError):
            utc("2024-01-01")

    def test_truncated_period_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "statistics.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Start", "End"])
                writer.writeheader()
                writer.writerow(
                    {"Start": "2024-01-01T00:00:00Z", "End": "2024-01-15T00:00:00Z"}
                )
            with self.assertRaises(ValueError):
                validate_period(path, "2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z")

    def test_zero_exit_without_file_fails_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.runner.subprocess.run") as process:
                process.return_value.returncode = 0
                result = run(
                    {"steps": [{"tool": "demo"}, {"tool": "demo"}]},
                    Path(directory) / "run",
                )
            self.assertFalse(result["ok"])
            self.assertEqual(len(result["steps"]), 1)

    def test_old_directory_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run({"steps": [{"tool": "demo"}]}, directory)
