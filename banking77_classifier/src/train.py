"""TF-IDF baseline и fine-tuning BERT с выбором checkpoint на validation."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .data import check_splits, read_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["tfidf", "bert"], required=True)
    p.add_argument("--data", default="data")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="google-bert/bert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.epochs < 1 or a.batch < 1 or a.lr <= 0:
        p.error("epochs, batch and lr must be positive")
    random.seed(a.seed)
    np.random.seed(a.seed)
    root, out = Path(a.data), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    train, valid = read_rows(root / "train.csv"), read_rows(root / "valid.csv")
    check_splits(train, valid)
    labels = sorted({r["label"] for r in train})
    if not {r["label"] for r in valid} <= set(labels):
        raise ValueError("Validation contains unseen label")
    x, y = [r["text"] for r in train], [r["label"] for r in train]
    vx, vy = [r["text"] for r in valid], [r["label"] for r in valid]
    if a.backend == "tfidf":
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            LogisticRegression(C=4, max_iter=1000, random_state=a.seed),
        )
        model.fit(x, y)
        best = f1_score(
            vy, model.predict(vx), labels=labels, average="macro", zero_division=0
        )
        joblib.dump(model, out / "model.joblib")
        revision = None
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(4)
        torch.manual_seed(a.seed)
        # CPU работает и на Mac без дополнительных настроек. CUDA используется
        # при наличии; MPS здесь намеренно не включен без отдельной проверки.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            a.model,
            num_labels=len(labels),
            id2label=dict(enumerate(labels)),
            label2id={v: i for i, v in enumerate(labels)},
        )
        revision = getattr(model.config, "_commit_hash", None)
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        best = -1.0
        best_epoch = None
        history = []
        for epoch in range(a.epochs):
            model.train()
            indices = np.random.permutation(len(train))
            for start in range(0, len(indices), a.batch):
                ids = indices[start : start + a.batch]
                batch = tok(
                    [x[i] for i in ids],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(device)
                targets = torch.tensor([labels.index(y[i]) for i in ids], device=device)
                opt.zero_grad()
                loss = model(**batch, labels=targets).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            pred = []
            with torch.inference_mode():
                for start in range(0, len(vx), a.batch):
                    batch = tok(
                        vx[start : start + a.batch],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    ).to(device)
                    pred.extend(
                        labels[i] for i in model(**batch).logits.argmax(-1).tolist()
                    )
            score = f1_score(vy, pred, labels=labels, average="macro", zero_division=0)
            history.append({"epoch": epoch + 1, "valid_macro_f1": float(score)})
            print({"epoch": epoch + 1, "valid_macro_f1": score}, flush=True)
            if score > best:
                best = score
                best_epoch = epoch + 1
                model.save_pretrained(out)
                tok.save_pretrained(out)
    (out / "training.json").write_text(
        json.dumps(
            {
                **vars(a),
                "labels": labels,
                "best_valid_macro_f1": best,
                "best_epoch": best_epoch if a.backend == "bert" else None,
                "history": history if a.backend == "bert" else None,
                "model_revision": revision,
                "data_sha256": {
                    s: hashlib.sha256((root / (s + ".csv")).read_bytes()).hexdigest()
                    for s in ("train", "valid")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
