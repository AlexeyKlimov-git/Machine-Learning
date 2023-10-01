import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from src.common import load, network, transform


class ModelTests(unittest.TestCase):
    def test_eval_transform_is_deterministic(self):
        picture = Image.new("RGB", (80, 90), (100, 120, 140))
        self.assertTrue(torch.equal(transform()(picture), transform()(picture)))
        self.assertEqual(tuple(transform()(picture).shape), (3, 224, 224))

    def test_architectures_backward(self):
        torch.set_num_threads(2)
        for name in ("cnn", "efficientnet"):
            # Без скачивания весов проверяем только shape/градиенты, не качество.
            model = network(name, 3, pretrained=False)
            output = model(torch.randn(2, 3, 64, 64))
            self.assertEqual(tuple(output.shape), (2, 3))
            output.sum().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_checkpoint_roundtrip(self):
        model = network("cnn", 2).eval()
        x = torch.randn(1, 3, 64, 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(
                {
                    "model": "cnn",
                    "classes": ["cat", "dog"],
                    "state": model.state_dict(),
                },
                path,
            )
            restored, classes = load(path)
            self.assertEqual(classes, ["cat", "dog"])
            self.assertTrue(torch.equal(model(x), restored(x)))
