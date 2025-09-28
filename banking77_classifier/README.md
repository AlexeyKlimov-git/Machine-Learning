# Классификация банковских обращений

## Запуск

Python 3.11–3.13, команды из папки проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data
python -m src.train --backend tfidf --output artifacts/tfidf
python -m src.evaluate --model artifacts/tfidf --output artifacts/eval_tfidf
```

Сравнение с BERT на маленькой выборке:

```bash
python -m src.subset
python -m src.train --backend tfidf --data data_small --output artifacts/tfidf_small
python -m src.train --backend bert --data data_small --epochs 8 --batch 8 --output artifacts/bert_small_8epochs
python -m src.evaluate --model artifacts/tfidf_small --data data_small --output artifacts/eval_tfidf_small
python -m src.evaluate --model artifacts/bert_small_8epochs --data data_small --output artifacts/eval_bert_small
```

Данные и веса скачиваются при первом запуске. Для повторного обучения и оценки используйте новые выходные папки.

Запустить API с TF-IDF:

```bash
MODEL_PATH=artifacts/tfidf uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Откройте http://127.0.0.1:8000/docs или выполните в другом терминале:

```bash
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @examples/request.json
```

Для BERT замените MODEL_PATH на `artifacts/bert_small_8epochs`.

## Результат

BANKING77: 77 категорий, 3080 обращений в test.
TF-IDF + Logistic Regression на 8493 обучающих примерах: **macro-F1 0.8847**.

Сравнение на одинаковых 385 обучающих примерах — по 5 на категорию:

| Модель | Test macro-F1 | Задержка p50 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.5626 | 0.187 мс |
| BERT-base, 8 эпох | 0.4351 | 13.898 мс |

На этой маленькой выборке TF-IDF оказался лучше.
Задержка: CPU, macOS ARM64, batch=1, после прогрева, включая токенизацию,
без загрузки весов. Это ориентировочный замер, не нагрузочный тест API.

Проверка API с TF-IDF вернула HTTP 200 и категорию `card_arrival`
для запроса из `examples/request.json`.

Подробные результаты: [отчёт](reports/RESULTS.md).

