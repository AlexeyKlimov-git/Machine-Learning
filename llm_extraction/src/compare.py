"""Сравнение только одинаковых test; разные объемы train видны в таблице."""

import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Paths to metrics.json")
    args = parser.parse_args()
    rows = [(p, json.loads(Path(p).read_text())) for p in args.runs]
    hashes = {r["test_sha256"] for _, r in rows}
    if len(hashes) != 1:
        raise ValueError("Cannot compare different test sets")
    for path, row in rows:
        score = row.get("macro_f1", row.get("field_micro_f1"))
        if score is None:
            score = row["report"]["macro avg"]["f1-score"]
        print(
            f"{path}: score={score:.6f}, test_n={row['n']}, train_n={row.get('train_n', 'see training.json')}"
        )
