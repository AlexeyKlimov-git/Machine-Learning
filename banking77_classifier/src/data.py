"""Данные делим до обучения; словарь TF-IDF никогда не видит test."""

import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path
from urllib.request import urlopen

from sklearn.model_selection import train_test_split


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(
        not r.get("text", "").strip() or not r.get("label") for r in rows
    ):
        raise ValueError("Expected nonempty CSV with text,label columns")
    return rows


def check_splits(*splits):
    # Нормализация ловит также различия только в регистре и пробелах.
    seen = set()
    for rows in splits:
        texts = {" ".join(r["text"].lower().split()) for r in rows}
        if seen & texts:
            raise ValueError("Duplicate text across splits")
        seen |= texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    # Получаем только публичные CSV. Хеши фиксируют конкретную версию данных,
    # даже если upstream позже изменит master.
    base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
    downloaded, hashes = {}, {}
    for split in ("train", "test"):
        cache = root / (split + ".source.csv")
        if cache.exists():
            raw = cache.read_bytes()
        else:
            for attempt in range(3):
                try:
                    with urlopen(base + split + ".csv", timeout=30) as response:
                        raw = response.read()
                    break
                except OSError:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            cache.write_bytes(raw)
        hashes[split] = hashlib.sha256(raw).hexdigest()
        downloaded[split] = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))

    # В исходном датасете встречаются повторы. Официальный test оставляем
    # неизменным; из train исключаем совпадения с ним и неоднозначные тексты.
    # Это только контроль утечки, не подбор модели по ответам test.
    def key(row):
        return " ".join(row["text"].lower().split())

    test_keys = {key(r) for r in downloaded["test"]}
    grouped = {}
    for row in downloaded["train"]:
        grouped.setdefault(key(row), []).append(row)
    clean = [
        rows[0]
        for text, rows in grouped.items()
        if text not in test_keys and len({r["category"] for r in rows}) == 1
    ]
    removed = len(downloaded["train"]) - len(clean)
    train, valid = train_test_split(
        clean, test_size=0.15, random_state=42, stratify=[r["category"] for r in clean]
    )
    outputs = {"train": train, "valid": valid, "test": downloaded["test"]}
    converted = {
        s: [{"text": r["text"], "label": r["category"]} for r in rows]
        for s, rows in outputs.items()
    }
    check_splits(*converted.values())
    for split, rows in converted.items():
        target = root / (split + ".csv")
        if target.exists():
            raise FileExistsError(
                f"Refusing to overwrite {target}; use a new output directory"
            )
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "source": base,
                "sha256": hashes,
                "seed": 42,
                "removed_train_rows": removed,
                "test_policy": "official test unchanged; normalized overlaps removed from train",
                "sizes": {k: len(v) for k, v in converted.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
