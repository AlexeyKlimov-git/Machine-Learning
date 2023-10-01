"""Тот же deterministic transform, что использован при validation и test."""

import argparse
import json

import torch
from PIL import Image

from .common import load, transform

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    a = p.parse_args()
    model, labels = load(a.checkpoint)
    with Image.open(a.image) as image:
        tensor = transform()(image.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(-1)[0]
    index = int(probabilities.argmax())
    # Softmax не является калиброванной уверенностью и не обнаруживает OOD.
    print(
        json.dumps(
            {"label": labels[index], "softmax_score": float(probabilities[index])}
        )
    )
