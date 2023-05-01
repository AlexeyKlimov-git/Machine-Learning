"""Одна оценка для обоих backend; ошибки и latency сохраняются вместе с метриками."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from .data import check_splits, read_rows
from .model import Predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = Path(a.data)
    train, valid, test = [
        read_rows(root / (s + ".csv")) for s in ("train", "valid", "test")
    ]
    check_splits(train, valid, test)
    provenance = json.loads((Path(a.model) / "training.json").read_text())
    for split in ("train", "valid"):
        actual = hashlib.sha256((root / (split + ".csv")).read_bytes()).hexdigest()
        if actual != provenance["data_sha256"][split]:
            raise ValueError("Evaluation data does not match training provenance")
    labels = sorted({r["label"] for r in train})
    if not {r["label"] for r in test} <= set(labels):
        raise ValueError("Unseen test label")
    model = Predictor(a.model)
    model.predict(["Where is my card?"])  # прогрев не входит в тайминг
    predictions, timings, errors = [], [], []
    for row in test:
        start = time.perf_counter()
        prediction = model.predict([row["text"]])[0]
        timings.append(1000 * (time.perf_counter() - start))
        predictions.append(prediction)
        if prediction != row["label"]:
            errors.append({**row, "prediction": prediction})
    report = classification_report(
        [r["label"] for r in test],
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "n": len(test),
        "train_n": len(train),
        "valid_n": len(valid),
        "training_data_sha256": provenance["data_sha256"],
        "macro_f1": report["macro avg"]["f1-score"],
        "report": report,
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "device": "cpu",
        "batch": 1,
        "platform": platform.platform(),
        "timing": "warm inference including tokenization; excluding model loading",
        "test_sha256": hashlib.sha256((root / "test.csv").read_bytes()).hexdigest(),
    }
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "errors.json").write_text(json.dumps(errors, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2))


if __name__ == "__main__":
    main()
