# Маршрутизация обращений: BANKING77

## Задача и данные

Предсказать один из 77 банковских intent по английскому обращению.
Источник: [PolyAI BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data).
Это публичные исследовательские данные, не обращения клиентов работодателя.
Условия использования проверяйте в исходном репозитории.

Официальный test сохраняется. Из исходного train удаляются нормализованные дубликаты,
тексты с противоречивыми метками и пересечения с test. Оставшийся train делится
стратифицированно 85/15 с seed=42. Число удалений и SHA256 записываются в manifest.
Такой протокол отличается от обучения на полном официальном train: это надо указывать при сравнении с чужими цифрами.

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

Для первого запуска на ноутбуке можно взять по 5 примеров на класс (385 train),
не меняя valid/test. Оба метода обязательно обучаются на одной подвыборке:

```bash
python -m src.subset
python -m src.train --backend tfidf --data data_small --output artifacts/tfidf_small
python -m src.train --backend bert --data data_small --epochs 8 --batch 8 --output artifacts/bert_small_8epochs
python -m src.evaluate --model artifacts/tfidf_small --data data_small --output artifacts/eval_tfidf_small
python -m src.evaluate --model artifacts/bert_small_8epochs --data data_small --output artifacts/eval_bert_small
```

Это low-data pilot с маленьким бюджетом обучения, не итоговое сравнение полностью
настроенных моделей. После одной эпохи validation показал недообучение; бюджет
увеличен до 8 эпох до просмотра BERT test. TF-IDF все равно может оказаться сильнее.
Offline-проверка всего нейросетевого pipeline: `python -m src.smoke`.

## Оценка и ошибки

Отчет: artifacts/eval_*/metrics.json; конкретные ошибки: errors.json.
Сравнивайте macro-F1, per-class precision/recall/F1, p50/p95 при одинаковом CPU.
Сначала фиксируйте настройки на valid; не подбирайте C/эпохи по test.
Результаты реально выполненных запусков — reports/RESULTS.md.

Типичные гипотезы для анализа: card_arrival vs card_delivery_tracking,
cash_withdrawal_charge vs extra_charge_on_statement. Это ожидаемые трудные пары,
не выдаваемые за измеренные ошибки: фактические пары смотрите в errors.json.

## Ограничения

Английский single-intent датасет; модель не решает русские обращения и multi-label.
Нет отдельного unknown intent или калибровки. Длинные тексты обрезаются до 128 токенов.
Latency не включает старт процесса и загрузку весов; результаты CPU не переносятся на GPU.
