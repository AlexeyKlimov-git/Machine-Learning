"""Единый predict для честного сравнения и HTTP-сервиса."""

from pathlib import Path


class Predictor:
    def __init__(self, path):
        path = Path(path)
        self.baseline = (path / "model.joblib").exists()
        if self.baseline:
            import joblib

            # joblib безопасен только для доверенных собственных артефактов.
            self.model = joblib.load(path / "model.joblib")
        else:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(4)
            self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, local_files_only=True
            ).eval()

    def predict(self, texts):
        if self.baseline:
            return self.model.predict(texts).tolist()
        import torch

        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.inference_mode():
            indices = self.model(**batch).logits.argmax(-1).tolist()
        return [self.model.config.id2label[i] for i in indices]
