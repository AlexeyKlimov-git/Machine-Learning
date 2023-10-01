"""ImageFolder train/valid; test не участвует в выборе checkpoint."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .common import network, transform


def main(a):
    if a.epochs < 1 or a.batch < 1:
        raise ValueError("epochs and batch must be positive")
    torch.set_num_threads(2)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    train = ImageFolder(str(Path(a.data) / "train"), transform(a.augment))
    valid = ImageFolder(str(Path(a.data) / "valid"), transform())
    if train.class_to_idx != valid.class_to_idx:
        raise ValueError("Class mapping differs between train and valid")

    def hashes(ds):
        return {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p, _ in ds.samples}

    if hashes(train) & hashes(valid):
        raise ValueError("Identical images across train/valid")
    test = ImageFolder(str(Path(a.data) / "test"), transform())
    if test.class_to_idx != train.class_to_idx:
        raise ValueError("Class mapping differs in test")
    if (hashes(train) | hashes(valid)) & hashes(test):
        raise ValueError("Identical images leak into test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network(a.model, len(train.classes), pretrained=not a.scratch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    criterion = torch.nn.CrossEntropyLoss()
    loaders = [
        DataLoader(ds, batch_size=a.batch, shuffle=i == 0, num_workers=0)
        for i, ds in enumerate([train, valid])
    ]
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("animal-classification")
    with mlflow.start_run():
        mlflow.log_params(
            {
                **vars(a),
                "device": str(device),
                "train_size": len(train),
                "valid_size": len(valid),
            }
        )
        best = -1
        for epoch in range(a.epochs):
            model.train()
            total = 0
            for x, y in loaders[0]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
                total += loss.item() * len(y)
            model.eval()
            truth, pred = [], []
            with torch.inference_mode():
                for x, y in loaders[1]:
                    pred.extend(model(x.to(device)).argmax(-1).cpu().tolist())
                    truth.extend(y.tolist())
            score = f1_score(
                truth,
                pred,
                labels=list(range(len(train.classes))),
                average="macro",
                zero_division=0,
            )
            mlflow.log_metrics(
                {"train_loss": total / len(train), "valid_macro_f1": score}, step=epoch
            )
            print(epoch, score)
            if score > best:
                best = score
                torch.save(
                    {
                        "state": model.cpu().state_dict(),
                        "model": a.model,
                        "classes": train.classes,
                        "train_valid_hashes": sorted(hashes(train) | hashes(valid)),
                    },
                    out / "model.pt",
                )
                model.to(device)
        (out / "training.json").write_text(
            json.dumps(
                {**vars(a), "best_valid_macro_f1": best, "classes": train.classes},
                indent=2,
            )
        )
        mlflow.log_artifact(str(out / "training.json"))
        mlflow.log_artifact(str(out / "model.pt"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "efficientnet"], default="efficientnet")
    p.add_argument("--data", default="data")
    p.add_argument("--output", required=True)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--scratch", action="store_true")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
