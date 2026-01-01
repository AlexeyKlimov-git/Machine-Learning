"""Контракт: отсутствие информации — null, а не догадка модели."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    product: Literal["card", "deposit", "transfer"] | None
    amount: float | None = Field(ge=0, allow_inf_nan=False)
    category: Literal["problem", "request", "question"] | None


def parse(raw):
    # Не вырезаем JSON из произвольного ответа и не исправляем ошибки незаметно:
    # иначе valid JSON rate перестанет измерять соблюдение контракта моделью.
    return Ticket.model_validate_json(raw).model_dump()
