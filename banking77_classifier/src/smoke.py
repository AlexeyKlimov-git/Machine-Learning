"""Полный BERT pipeline без сети: tiny random model, НЕ pretrained baseline."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

if __name__ == "__main__":
    root = Path("artifacts/smoke")
    root.mkdir(parents=True, exist_ok=False)
    base, data = root / "tiny_base", root / "data"
    base.mkdir()
    data.mkdir()
    torch.manual_seed(42)
    labels = ["card", "deposit", "transfer"]
    # Vocab маленький специально: это тест контракта обучения и checkpoint,
    # а не попытка заменить BERT случайными весами в сравнении качества.
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "my",
        "help",
        "please",
    ] + labels
    (base / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizer(vocab_file=str(base / "vocab.txt"))
    tokenizer.save_pretrained(base)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=3,
            id2label=dict(enumerate(labels)),
            label2id={v: i for i, v in enumerate(labels)},
        )
    )
    model.save_pretrained(base)
    for split in ("train", "valid", "test"):
        with (data / (split + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(
                {"text": f"my {label} help please {split} {i}", "label": label}
                for label in labels
                for i in range(3)
            )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--backend",
            "bert",
            "--model",
            str(base),
            "--data",
            str(data),
            "--epochs",
            "1",
            "--output",
            str(root / "model"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--model",
            str(root / "model"),
            "--data",
            str(data),
            "--output",
            str(root / "evaluation"),
        ],
        check=True,
    )
    (root / "NOTICE.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "pretrained": False,
                "purpose": "technical smoke; not BANKING77 metrics",
            }
        )
    )
