"""Открытый игрушечный бэктест. Не содержит стратегий или данных работодателя."""

import argparse
import json
import math
import random
from pathlib import Path


def simulate(prices, window=None, cost=0.0005):
    equity, peak, drawdown, position = 1.0, 1.0, 0.0, 0
    # Решение в t использует данные до t включительно, доходность — t -> t+1.
    # В этой упрощенной модели сделка исполняется по close без проскальзывания.
    for t in range(30, len(prices) - 1):
        target = (
            1
            if window is None
            else int(prices[t] > sum(prices[t - window + 1 : t + 1]) / window)
        )
        turnover = abs(target - position)
        ret = prices[t + 1] / prices[t] - 1
        equity *= 1 + target * ret - turnover * cost
        position = target
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)
    equity *= 1 - abs(position) * cost  # закрытие позиции тоже стоит денег
    drawdown = max(drawdown, 1 - equity / peak)
    return {"return": equity - 1, "max_drawdown": drawdown}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rng = random.Random(42)
    prices = [100.0]
    for _ in range(600):
        prices.append(prices[-1] * math.exp(rng.gauss(0.0002, 0.01)))
    # Window фиксирован заранее; никакого перебора на test в этом demo нет.
    test = prices[370:]
    result = {
        "synthetic": True,
        "seed": 42,
        "test_bars": len(test),
        "buy_and_hold": simulate(test),
        "moving_average_20": simulate(test, 20),
        "note": "plumbing demo, not evidence of trading alpha",
    }
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
