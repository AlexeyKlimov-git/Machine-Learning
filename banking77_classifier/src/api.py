"""Сервис не скачивает веса при старте и не пишет тексты обращений в лог."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .model import Predictor


class Request(BaseModel):
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("text")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value


def create_app(predictor=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = predictor or Predictor(
            os.getenv("MODEL_PATH", "artifacts/tfidf")
        )
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ready": hasattr(app.state, "predictor")}

    @app.post("/predict")
    def predict(request: Request):
        try:
            return {"label": app.state.predictor.predict([request.text])[0]}
        except Exception:
            # Не логируем exception message: сторонняя библиотека может включить
            # туда пользовательский текст. Клиенту также не отдаем traceback.
            logging.getLogger(__name__).error("Prediction failed")
            raise HTTPException(503, "Prediction unavailable") from None

    return app


app = create_app()
