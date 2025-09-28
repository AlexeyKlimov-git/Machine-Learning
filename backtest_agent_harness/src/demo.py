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
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--cost", type=float, default=0.0005)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if not 2 <= a.window <= 30 or not 0 <= a.cost <= 0.01:
        p.error("window must be 2..30 and cost must be 0..0.01")
    rng = random.Random(a.seed)
    prices = [100.0]
    for _ in range(600):
        prices.append(prices[-1] * math.exp(rng.gauss(0.0002, 0.01)))
    # Window фиксирован заранее; никакого перебора на test в этом demo нет.
    test = prices[370:]
    result = {
        "synthetic": True,
        "seed": a.seed,
        "window": a.window,
        "cost": a.cost,
        "test_bars": len(test),
        "buy_and_hold": simulate(test, cost=a.cost),
        f"moving_average_{a.window}": simulate(test, a.window, cost=a.cost),
        "note": "plumbing demo, not evidence of trading alpha",
    }
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
