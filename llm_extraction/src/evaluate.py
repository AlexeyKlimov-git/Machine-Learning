"""Регрессионная оценка промптов; ошибки остаются в знаменателе метрик."""

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

from .backends import Qwen, Rules
from .metrics import score
from .prompts import SYSTEM
from .schema import Ticket, parse


def read(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("Empty dataset")
    for row in rows:
        Ticket.model_validate(row["expected"])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["rules", "qwen"], required=True)
    p.add_argument("--mode", choices=["zero", "few"], default="zero")
    p.add_argument("--data", default="examples/test.jsonl")
    p.add_argument("--dev", default="examples/dev.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--revision", default="main")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows, dev = read(a.data), read(a.dev)
    if {r["text"].strip().lower() for r in rows} & {
        r["text"].strip().lower() for r in dev
    }:
        raise ValueError("Dev/test overlap")
    if a.backend == "rules" and a.mode != "zero":
        p.error("Rules have no few-shot mode")
    backend = Rules() if a.backend == "rules" else Qwen(a.model, a.revision)
    examples = dev if a.mode == "few" else []
    backend.predict("Здравствуйте", examples)  # один отдельный прогрев
    records = []
    for row in rows:
        start = time.perf_counter()
        raw, prediction, error = None, None, None
        try:
            raw = backend.predict(row["text"], examples)
            prediction = parse(raw)
        except Exception as exc:
            error = type(exc).__name__
        records.append(
            {
                **row,
                "raw": raw,
                "prediction": prediction,
                "error": error,
                "seconds": time.perf_counter() - start,
            }
        )
        print(f"{len(records)}/{len(rows)}", flush=True)
    metrics = score(records)
    metrics.update(
        {
            "backend": a.backend,
            "mode": a.mode,
            "model_revision": backend.revision,
            "p50_seconds": statistics.median(r["seconds"] for r in records),
            "platform": platform.platform(),
            "device": "cpu",
            "batch": 1,
            "test_sha256": hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(
                (SYSTEM + json.dumps(examples, sort_keys=True)).encode()
            ).hexdigest(),
            "data_note": "handwritten synthetic fixture; not production quality",
        }
    )
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "predictions.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False)
    )
    (out / "errors.json").write_text(
        json.dumps(
            [r for r in records if r["prediction"] != r["expected"]],
            indent=2,
            ensure_ascii=False,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
