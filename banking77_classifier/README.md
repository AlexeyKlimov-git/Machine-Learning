# Маршрутизация обращений: BANKING77

## Задача и данные

Предсказать один из 77 банковских intent по английскому обращению.
Источник: [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data).


## Что реализовано

- TF-IDF (1–2-граммы) + Logistic Regression как дешевый baseline.
- Fine-tuning google-bert/bert-base-uncased, CrossEntropyLoss, AdamW, clipping.
- Выбор лучшего checkpoint только по validation macro-F1.
- Общая оценка, список ошибок и warm CPU latency при batch=1.
- FastAPI с проверкой входа, обработкой ошибок и Docker без root.

Это объем реализации проекта, а не утверждение о коммерческом внедрении.

## Запуск

Python 3.11–3.13. Из этой папки:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.data
python -m src.train --backend tfidf --output artifacts/tfidf
python -m src.evaluate --model artifacts/tfidf --output artifacts/eval_tfidf
python -m src.train --backend bert --epochs 3 --output artifacts/bert
python -m src.evaluate --model artifacts/bert --output artifacts/eval_bert
MODEL_PATH=artifacts/bert uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Первый запуск скачивает данные и веса; BERT на CPU заметно медленнее baseline.
Выходные каталоги обучения/оценки должны быть новыми, чтобы не смешивать эксперименты.

```bash
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @examples/request.json
docker build -t banking77 .
docker run --rm -p 127.0.0.1:8000:8000 -e MODEL_PATH=/models -v "$PWD/artifacts/bert:/models:ro" banking77
```

Docker-образ не содержит датасеты и веса. API не логирует текст запроса.
Это локальный демонстрационный сервис без аутентификации, rate limiting и очереди GPU.

## Короткий сравнительный эксперимент



```bash
python -m src.subset
python -m src.train --backend tfidf --data data_small --output artifacts/tfidf_small
python -m src.train --backend bert --data data_small --epochs 8 --batch 8 --output artifacts/bert_small_8epochs
python -m src.evaluate --model artifacts/tfidf_small --data data_small --output artifacts/eval_tfidf_small
python -m src.evaluate --model artifacts/bert_small_8epochs --data data_small --output artifacts/eval_bert_small
```



