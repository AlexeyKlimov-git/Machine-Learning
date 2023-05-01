"""Фиксированный low-data benchmark: одинаковый train для BERT и baseline."""

import argparse
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .data import read_rows

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data")
    p.add_argument("--output", default="data_small")
    p.add_argument("--per-class", type=int, default=5)
    a = p.parse_args()
    if a.per_class < 1:
        p.error("per-class must be positive")
    source, out = Path(a.input), Path(a.output)
    groups = defaultdict(list)
    for row in read_rows(source / "train.csv"):
        groups[row["label"]].append(row)
    rng = random.Random(42)
    rows = []
    for label in sorted(groups):
        rows.extend(rng.sample(groups[label], min(a.per_class, len(groups[label]))))
    out.mkdir(parents=True, exist_ok=False)
    with (out / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    for split in ("valid", "test"):
        shutil.copyfile(source / (split + ".csv"), out / (split + ".csv"))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "seed": 42,
                "train_rows": len(rows),
                "per_class": a.per_class,
                "note": "low-data benchmark, not full-data training",
            },
            indent=2,
        )
    )
