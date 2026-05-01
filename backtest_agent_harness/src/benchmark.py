"""Измеряем накладные расходы runner, а не выдуманную экономию времени человека."""

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .runner import run

if __name__ == "__main__":
    direct, harness = [], []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i in range(5):
            start = time.perf_counter()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.demo",
                    "--output",
                    str(root / "direct.json"),
                ],
                check=True,
            )
            direct.append(time.perf_counter() - start)
            start = time.perf_counter()
            result = run({"steps": [{"tool": "demo"}]}, root / str(i))
            harness.append(time.perf_counter() - start)
            assert result["ok"]
            assert json.loads((root / "direct.json").read_text()) == json.loads(
                (root / str(i) / "0/metrics.json").read_text()
            )
    result = {
        "runs": 5,
        "direct_median_seconds": statistics.median(direct),
        "harness_median_seconds": statistics.median(harness),
        "matching_results": 5,
        "human_time_saved": None,
        "agent_success_rate": None,
        "scope": "synthetic deterministic demo; no LLM agent or private tools executed",
    }
    out = Path("artifacts/benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
