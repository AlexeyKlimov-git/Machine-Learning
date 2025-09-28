"""Локальная LLM предлагает план, runner выполняет его и возвращает наблюдения."""

import argparse
import copy
import json
import urllib.request
import uuid
from pathlib import Path

from .runner import run
from .tools import utc

TOOLS = {"demo", "backtest", "dibate2csv", "diviz", "diribbon"}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": sorted(TOOLS)},
                    "args": {
                        "type": "object",
                        "properties": {
                            "window": {"type": "integer"},
                            "cost": {"type": "number"},
                            "seed": {"type": "integer"},
                            "config": {"type": "string"},
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                            "snapshot": {"type": "string"},
                            "cache": {"type": "string"},
                            "input": {"type": "string"},
                            "capital_mode": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
                },
                "required": ["tool"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


def chat(messages, model="qwen3.5:9b", schema=None):
    """Обращение только к локальному Ollama API; ключ и облако не нужны."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }
    if schema is not None:
        payload["format"] = schema
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
    except OSError as exc:
        raise RuntimeError(
            "Ollama unavailable; start it and run `ollama pull qwen3.5:9b`"
        ) from exc
    content = result.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty response")
    return content


def validate_plan(plan):
    """Не доверяем даже JSON из schema mode: проверяем шаги перед запуском."""
    if not isinstance(plan, dict) or set(plan) != {"steps"}:
        raise ValueError("Expected an object with only `steps`")
    steps = plan["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 20:
        raise ValueError("Plan must have 1..20 steps")
    for step in steps:
        if not isinstance(step, dict) or not set(step) <= {"tool", "args", "timeout"}:
            raise ValueError("Invalid step shape")
        tool = step.get("tool")
        if tool not in TOOLS:
            raise ValueError(f"Unknown tool: {tool}")
        timeout = step.get("timeout", 120)
        if type(timeout) is not int or not 0 < timeout <= 600:
            raise ValueError("Invalid timeout")
        args = step.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("Arguments must be an object")
        required = {
            "demo": set(),
            "backtest": {"config", "from", "to", "snapshot", "cache"},
            "dibate2csv": {"input", "from", "to"},
            "diviz": {"input"},
            "diribbon": {"input"},
        }[tool]
        allowed = required | ({"capital_mode"} if tool == "diribbon" else set())
        if tool == "demo":
            allowed = {"window", "cost", "seed"}
        if not required <= set(args) or not set(args) <= allowed:
            raise ValueError(f"Invalid arguments for {tool}")
        if tool == "demo":
            window = args.get("window", 20)
            cost = args.get("cost", 0.0005)
            seed = args.get("seed", 42)
            if (
                type(window) is not int
                or not 2 <= window <= 30
                or type(cost) not in (int, float)
                or not 0 <= cost <= 0.01
                or type(seed) is not int
                or not 0 <= seed <= 1_000_000
            ):
                raise ValueError("Invalid demo parameters")
        elif any(not isinstance(value, str) or not value for value in args.values()):
            raise ValueError("Arguments must be nonempty strings")
        if tool == "backtest" and utc(args["from"]) >= utc(args["to"]):
            raise ValueError("Invalid backtest period")
    return plan


def propose(task, model, allow_local=False):
    allowed_tools = TOOLS if allow_local else {"demo"}
    schema = copy.deepcopy(PLAN_SCHEMA)
    schema["properties"]["steps"]["items"]["properties"]["tool"]["enum"] = sorted(
        allowed_tools
    )
    if not allow_local:
        system = (
            'Return exactly {"steps":[{"tool":"demo","args":{"window":N}}]}. '
            "N is the moving-average window requested by the user, otherwise 20. "
            "The only tool is demo. Optional args: window, cost, seed. "
            "No paths, dates, config, snapshot, strategy or other fields."
        )
        schema["properties"]["steps"]["items"]["properties"]["args"][
            "properties"
        ] = {
            "window": {"type": "integer", "minimum": 2, "maximum": 30},
            "cost": {"type": "number", "minimum": 0, "maximum": 0.01},
            "seed": {"type": "integer", "minimum": 0, "maximum": 1_000_000},
        }
        schema["properties"]["steps"]["maxItems"] = 1
    else:
        system = (
            "You plan backtest experiments. Return exactly one JSON object with steps. "
            f"Allowed tools: {', '.join(sorted(allowed_tools))}. "
            "Use demo for requests without actual local config and input paths. "
            "demo optional args ONLY: integer window 2..30, numeric cost 0..0.01, "
            "integer seed 0..1000000. Use the requested window if specified. "
            "Do not invent paths, dates, snapshots or results. "
            "backtest args: config, from, to, snapshot, cache. "
            "dibate2csv args: input, from, to. diviz args: input. "
            "diribbon args: input, optional capital_mode. "
            "Dates must include a timezone. Use one step unless all later input "
            "paths are already known; outputs are not interpolated between steps. "
            "No shell commands or arbitrary flags. "
            "Execution is a separate explicit step."
        )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    for _ in range(2):
        content = chat(messages, model, schema)
        try:
            plan = validate_plan(json.loads(content))
            if any(step["tool"] not in allowed_tools for step in plan["steps"]):
                raise ValueError("Tool unavailable for this run")
            return plan
        except ValueError as exc:
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"Invalid plan: {exc}. Correct the JSON plan.",
                    },
                ]
            )
    raise ValueError("Local model did not produce a valid plan after two attempts")


def explain(task, plan, result, metrics, model):
    observation = {"task": task, "plan": plan, "run": result, "metrics": metrics}
    return chat(
        [
            {
                "role": "system",
                "content": (
                    "Explain the observed run in Russian in 2-4 sentences. "
                    "Use only the supplied observations. Synthetic demo is not "
                    "evidence of trading profitability. State failures plainly."
                ),
            },
            {"role": "user", "content": json.dumps(observation, ensure_ascii=False)},
        ],
        model,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="What to check, in plain language")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", help="Execute a previously reviewed plan.json")
    parser.add_argument("--allow-local-tools", action="store_true")
    parser.add_argument("--binaries", help="JSON mapping of local tool names to paths")
    parser.add_argument("--output", default="artifacts/agent")
    args = parser.parse_args()
    if args.plan and not args.execute:
        parser.error("--plan requires --execute")
    plan = (
        validate_plan(json.loads(Path(args.plan).read_text()))
        if args.plan
        else propose(args.task, args.model, args.allow_local_tools)
    )
    if not args.allow_local_tools and any(s["tool"] != "demo" for s in plan["steps"]):
        parser.error("Local tools require --allow-local-tools")
    output = Path(args.output) / uuid.uuid4().hex
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    (output / "model.json").write_text(
        json.dumps({"provider": "local Ollama", "model": args.model}, indent=2)
    )
    print(f"Plan: {output / 'plan.json'}", flush=True)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        return
    binaries = json.loads(Path(args.binaries).read_text()) if args.binaries else {}
    result = run(plan, output / "run", binaries, args.allow_local_tools)
    metrics = []
    for step in result["steps"]:
        if step["ok"] and step["tool"] == "demo":
            path = output / "run" / step["artifact"]
            metrics.append(json.loads(path.read_text()))
    print(f"Run: {output / 'run' / 'summary.json'}", flush=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    report = explain(args.task, plan, result, metrics, args.model)
    (output / "report.md").write_text(report + "\n")
    print(f"Report: {output / 'report.md'}\n{report}", flush=True)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
