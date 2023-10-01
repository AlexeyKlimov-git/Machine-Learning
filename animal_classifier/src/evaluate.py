"""Отложенная оценка: сохраняем не только метрики, но и конкретные ошибки."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .common import load, transform

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/test")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    model, classes = load(a.checkpoint)
    ds = ImageFolder(a.data, transform())
    if ds.classes != classes:
        raise ValueError("Class mapping does not match checkpoint")
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    test_hashes = [
        hashlib.sha256(Path(p).read_bytes()).hexdigest() for p, _ in ds.samples
    ]
    if set(test_hashes) & set(checkpoint["train_valid_hashes"]):
        raise ValueError("Test overlaps checkpoint training data")
    loader = DataLoader(ds, batch_size=16, num_workers=0)
    truth, pred = [], []
    with torch.inference_mode():
        model(torch.zeros(1, 3, 224, 224))
        start = time.perf_counter()
        for x, y in loader:
            pred.extend(model(x).argmax(-1).tolist())
            truth.extend(y.tolist())
    elapsed = time.perf_counter() - start
    metrics = {
        "report": classification_report(
            truth,
            pred,
            labels=list(range(len(classes))),
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            truth, pred, labels=list(range(len(classes)))
        ).tolist(),
        "class_order": classes,
        "n": len(ds),
        "images_per_second": len(ds) / elapsed,
        "device": "cpu",
        "batch_size": 16,
        "timing": "warm; includes image loading and preprocessing",
    }
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    errors = [
        {
            "file": Path(ds.samples[i][0]).name,
            "truth": classes[y],
            "prediction": classes[p],
        }
        for i, (y, p) in enumerate(zip(truth, pred))
        if y != p
    ]
    (out / "errors.json").write_text(json.dumps(errors, indent=2))
    metrics["test_sha256"] = hashlib.sha256(
        "".join(sorted(test_hashes)).encode()
    ).hexdigest()
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)
