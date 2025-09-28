# Классификация животных

## Запуск

Python 3.11+, команды из папки проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.prepare --output data/pets
python -m src.subset
python -m src.train --model cnn --scratch --data data/pets3 --epochs 3 --output artifacts/cnn_pets3
python -m src.train --model efficientnet --augment --data data/pets3 --epochs 3 --output artifacts/eff_pets3
python -m src.evaluate --checkpoint artifacts/cnn_pets3/model.pt --data data/pets3/test --output artifacts/eval_cnn_pets3
python -m src.evaluate --checkpoint artifacts/eff_pets3/model.pt --data data/pets3/test --output artifacts/eval_eff_pets3
```

Данные и веса скачиваются при первом запуске. Для повторного запуска задайте новые выходные папки.

Проверить своё фото (замените `photo.jpg` на путь к файлу):

```bash
python -m src.predict --checkpoint artifacts/eff_pets3/model.pt --image photo.jpg
```

Посмотреть эксперименты:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

## Результат

298 тестовых фото Oxford-IIIT Pet, три породы: Abyssinian, beagle, pug.
Это не оценка на всех 37 породах.

| Модель | Macro-F1 | Accuracy | Фото/сек |
|---|---:|---:|---:|
| CNN с нуля | 0.2895 | 36.58% | 122.8 |
| EfficientNet-B0, предобучение + аугментации | 0.9900 | 98.99% | 14.1 |

Скорость измерена на CPU, batch=16, после прогрева, с чтением и подготовкой фото.
У моделей различаются не только архитектуры, но и предобучение и аугментации.

Команда predict выводит породу и softmax score.
Подробные метрики: [CNN](reports/cnn_pets3.json), [EfficientNet](reports/eff_aug_pets3.json).

