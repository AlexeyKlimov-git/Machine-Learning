# Реально выполненные эксперименты — 27 сентября 2026

macOS ARM64, CPU, Python 3.13.14. Версии в constraints-tested.txt.
77 intent; официальный test: 3080 обращений, valid: 1499.
Из исходного train исключено 11 повторных/неоднозначных/пересекающихся строк.
Хеши данных присутствуют в JSON рядом с отчетом.

## Основной baseline

TF-IDF + Logistic Regression на 8493 train: test macro-F1 **0.884696**.
Артефакт: tfidf_full.json. Эта строка не сравнивается напрямую с BERT ниже,
потому что объемы обучения разные.

## Честное low-data сравнение

По 5 примеров на класс, всего 385 train. Идентичные train/valid/test у обоих методов.

| Метод | Test macro-F1 | Warm p50 CPU |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.562623 | 0.187 мс |
| BERT-base-uncased, 8 эпох | 0.435119 | 13.898 мс |

Первый пробный бюджет в одну эпоху дал valid F1=0.000258 — явное недообучение.
До оценки BERT на test бюджет увеличен до 8 эпох. Лучший valid F1=0.421819,
checkpoint выбран по validation. Test не использован для подбора эпох.
Веса BERT дообучались целиком, а не только classifier head.

## Вывод

На 385 train-примерах и таком бюджете TF-IDF оказался лучше BERT. У проекта нет
основания заявлять, что Transformer улучшил качество. Следующий эксперимент —
обучение на полном train с настройками по valid, а не подбор по уже увиденному test.
Один seed и короткий запуск не определяют лучший класс моделей вообще.
Latency включает токенизацию, не включает загрузку весов. Во время некоторых
измерений выполнялись другие задачи: это ориентир, не изолированный hardware benchmark.

## Реальные ошибки полного TF-IDF baseline

- virtual_card_not_working → getting_virtual_card: 7 случаев.
- contactless_not_working → card_not_working: 5 случаев.
- card_acceptance → card_not_working: 5 случаев.
- unable_to_verify_identity → verify_my_identity: 5 случаев.

В artifacts/eval_tfidf_verified/errors.json сохранены исходные публичные примеры.
Три unit-теста проходят; реальный TF-IDF через FastAPI TestClient вернул 200
и card_arrival для примера из examples. Docker build не проверен: Docker не установлен.
Отдельно прошел tiny random BERT smoke; его результаты не входят в таблицы выше.

Команды повторения — раздел «Короткий сравнительный эксперимент» в README.
