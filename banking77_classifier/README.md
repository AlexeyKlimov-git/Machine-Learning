# Классификация банковских обращений

## Запуск

Команды из папки проекта. Если окружение уже создано, начните с `source`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --backend bert --epochs 5 --batch 32 --lr 2e-5 --output artifacts/bert_full_lr2e5_b32
```

Данные должны лежать в `data/train.csv` и `data/valid.csv`.
Веса BERT загружаются при первом запуске. Папка `--output` должна быть новой.

Запустить API с обученным BERT:

```bash
MODEL_PATH=artifacts/bert_full_lr2e5_b32 uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Откройте http://127.0.0.1:8000/docs или выполните в другом терминале:

```bash
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @examples/request.json
```

## Результат

Обучен `bert-base-uncased` на полном train: 8493 обращения, 77 категорий.
Проверка шла на 1499 обращениях из `valid`. CPU, batch 32, learning rate 2e-5.

| Эпоха | Macro-F1 на valid |
|---:|---:|
| 1 | 0.5619 |
| 2 | 0.7710 |
| 3 | 0.8577 |
| 4 | 0.8992 |
| 5 | **0.9098** |

Лучший checkpoint сохранён в `artifacts/bert_full_lr2e5_b32`.
На `test` эту модель пока не оценивали, поэтому сравнивать её `valid` F1 с
`test` F1 бейзлайна нельзя. Ранее TF-IDF получил **0.8847 на test** при обучении
на тех же 8493 примерах.

История обучения: `artifacts/bert_full_lr2e5_b32/training.json`.
