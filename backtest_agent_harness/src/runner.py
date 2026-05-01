"""Исполнитель проверенного плана: агент предлагает, человек разрешает запуск."""

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .tools import build, validate_period


def check_artifact(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Tool exited without a nonempty artifact")


def run(plan, destination, binaries=None, allow_local=False):
    steps = plan.get("steps", [])
    if not steps or len(steps) > 20:
        raise ValueError("Plan must contain 1..20 steps")
    if any(s.get("tool") != "demo" for s in steps) and not allow_local:
        raise PermissionError("Local tools require explicit --allow-local-tools")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for i, step in enumerate(steps):
        start = time.perf_counter()
        folder = destination / str(i)
        try:
            if step["tool"] == "demo":
                folder.mkdir()
                artifact = folder / "metrics.json"
                command = [sys.executable, "-m", "src.demo", "--output", str(artifact)]
            else:
                command, artifact = build(step, folder, binaries or {})
            timeout = step.get("timeout", 120)
            if not isinstance(timeout, (int, float)) or not 0 < timeout <= 600:
                raise ValueError("Timeout must be within (0, 600]")
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                timeout=timeout,
            )
            # Не сохраняем stdout/stderr закрытого инструмента: даже ошибка
            # может содержать адреса инфраструктуры и чувствительные параметры.
            if result.returncode != 0:
                raise RuntimeError(f"Tool exit code {result.returncode}")
            check_artifact(artifact)
            if step["tool"] == "dibate2csv":
                args = step["args"]
                validate_period(artifact, args["from"], args["to"])
            records.append(
                {
                    "tool": step["tool"],
                    "ok": True,
                    "artifact": str(artifact.relative_to(destination)),
                }
            )
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            subprocess.TimeoutExpired,
        ) as exc:
            records.append(
                {
                    "tool": step.get("tool"),
                    "ok": False,
                    "error_type": type(exc).__name__,
                }
            )
        records[-1]["seconds"] = time.perf_counter() - start
        if not records[-1]["ok"]:
            break
    summary = {
        "ok": all(r["ok"] for r in records),
        "steps": records,
        "note": "runner is not an OS sandbox; trusted plans only",
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("plan")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--allow-local-tools", action="store_true")
    p.add_argument(
        "--binaries", help="Private JSON map of tool name to executable path"
    )
    p.add_argument("--output", default="artifacts")
    a = p.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    if not a.execute:
        # Без execute даже demo не запускается. Агентный план сначала читаем.
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    binaries = json.loads(Path(a.binaries).read_text()) if a.binaries else {}
    destination = Path(a.output) / uuid.uuid4().hex
    result = run(plan, destination, binaries, a.allow_local_tools)
    print(destination, json.dumps(result))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
