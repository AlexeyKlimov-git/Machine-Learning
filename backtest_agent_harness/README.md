# Запуск бэктестов

## Запуск

Нужен Python 3.11+, дополнительные библиотеки не нужны.
Команды из папки проекта:

```bash
python3 -m src.runner examples/demo.json
python3 -m src.runner examples/demo.json --execute
python3 -m src.benchmark
```

Без `--execute` выводится только план.
Результаты запуска: `artifacts/<run-id>/summary.json` и `0/metrics.json` внутри той же папки.

Для своего локального бэктестера подготовьте `local-plan.json` по
`examples/local-plan.template.json`, а в `local-binaries.json` укажите
абсолютные пути к нужным инструментам: backtest, dibate2csv, diviz, diribbon.

```bash
python3 -m src.runner local-plan.json --execute --allow-local-tools --binaries local-binaries.json
```

Проверьте план перед запуском: runner не изолирует выполняемые программы.

## Результат

Пять парных запусков на искусственных ценах, macOS ARM64:

| Измерение | Значение |
|---|---:|
| Медиана прямого запуска | 33.61 мс |
| Медиана через harness | 34.97 мс |
| Совпадение результатов | 5 из 5 |

Накладные расходы — около 1.36 мс. Экономия времени человека и качество
LLM-агента не измерены. Встроенного LLM-планировщика нет.
Локальные закрытые инструменты в этом тесте не запускались.

Исходные измерения: [benchmark.json](reports/benchmark.json).

