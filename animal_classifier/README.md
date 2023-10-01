# Классификация животных: CNN и EfficientNet-B0

## Задача и данные

Классификация 37 пород кошек и собак на [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/).
Датасет около 800 МБ; CC BY-SA 4.0, права на фото остаются у владельцев.
Официальный test отдельно; trainval делится 80/20 стратифицированно, seed=42.
Веса [EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
предобучены на ImageNet.

## Что реализовано

- Собственная CNN с тремя сверточными блоками против EfficientNet-B0.
- Полное дообучение, сравнение с/без аугментаций.
- Единые resize/normalization для train, valid, test и predict; случайные аугментации только train.
- Выбор checkpoint по valid macro-F1; MLflow сохраняет параметры, метрики и веса.
- Проверка точных дубликатов между всеми split, SHA256 test, confusion matrix и список ошибок.

Сравнение оценивает практические решения, а не изолированный эффект архитектуры:
CNN стартует с нуля, EfficientNet — с ImageNet.

## Установка и быстрый smoke

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.prepare --demo --output data/demo
python -m src.train --model cnn --scratch --data data/demo --epochs 1 --output artifacts/cnn
python -m src.evaluate --checkpoint artifacts/cnn/model.pt --data data/demo/test --output artifacts/eval_demo
python -m src.predict --checkpoint artifacts/cnn/model.pt --image data/demo/test/cat/0.png
```

Smoke использует цветные картинки. Его цифры НЕ являются качеством распознавания животных.

## Полноценный эксперимент

```bash
python -m src.prepare --output data/pets
python -m src.train --model cnn --scratch --data data/pets --output artifacts/cnn_pets
python -m src.train --model efficientnet --data data/pets --output artifacts/eff_pets
python -m src.train --model efficientnet --augment --data data/pets --output artifacts/eff_aug_pets
python -m src.evaluate --checkpoint artifacts/cnn_pets/model.pt --data data/pets/test --output artifacts/eval_cnn_pets
python -m src.evaluate --checkpoint artifacts/eff_pets/model.pt --data data/pets/test --output artifacts/eval_eff_pets
python -m src.evaluate --checkpoint artifacts/eff_aug_pets/model.pt --data data/pets/test --output artifacts/eval_eff_aug_pets
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

Для сравнения аугментаций держите остальные параметры одинаковыми; лучше повторить на трех seed.
Не запускайте команды повторно в существующую output-папку: задайте новое имя.
По умолчанию 5 эпох; это стартовый бюджет, а не гарантия оптимального качества.

## Результаты и ограничения

Короткий реальный benchmark на трех породах, выбранных заранее:

```bash
python -m src.subset
python -m src.train --model cnn --scratch --data data/pets3 --epochs 3 --output artifacts/cnn_pets3
python -m src.train --model efficientnet --augment --data data/pets3 --epochs 3 --output artifacts/eff_pets3
python -m src.evaluate --checkpoint artifacts/cnn_pets3/model.pt --data data/pets3/test --output artifacts/eval_cnn_pets3
python -m src.evaluate --checkpoint artifacts/eff_pets3/model.pt --data data/pets3/test --output artifacts/eval_eff_pets3
```

Подвыборка Abyssinian/beagle/pug сохраняет официальное разделение данных.
Это не результат на всех 37 породах. Здесь сравниваются готовые решения
(pretraining и аугментации тоже различаются), не чистая ablation архитектуры.

Реальные выполненные проверки — reports/RESULTS.md. Оценка пишет metrics.json и errors.json.
Метрики: macro-F1, accuracy, матрица ошибок; throughput включает чтение изображений и preprocessing.
Тесты проверяют градиенты обеих архитектур, сохранение/загрузку и deterministic preprocessing.

Трудные случаи: похожие породы, маленькое животное в кадре, фон, несколько животных.
Нет OOD-класса, детектора или гарантии правильности softmax confidence.
Хеши обнаруживают только точные дубликаты, не near-duplicates и не одного питомца на разных фото.
