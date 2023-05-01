import unittest

from fastapi.testclient import TestClient
from src.api import create_app
from src.data import check_splits


class FakeModel:
    def predict(self, texts):
        return ["card_arrival"] * len(texts)


class ContractTests(unittest.TestCase):
    def test_api_and_validation(self):
        with TestClient(create_app(FakeModel())) as client:
            self.assertEqual(client.get("/health").json(), {"ready": True})
            self.assertEqual(
                client.post("/predict", json={"text": "Where is my card?"}).json(),
                {"label": "card_arrival"},
            )
            for text in ("", "   ", "x" * 4001):
                self.assertEqual(
                    client.post("/predict", json={"text": text}).status_code, 422
                )

    def test_error_is_not_exposed(self):
        class Broken:
            def predict(self, texts):
                raise RuntimeError("private details")

        with TestClient(create_app(Broken())) as client:
            response = client.post("/predict", json={"text": "hello"})
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private", response.text)

    def test_leakage(self):
        with self.assertRaises(ValueError):
            check_splits([{"text": "Hello world"}], [{"text": "HELLO  WORLD"}])
