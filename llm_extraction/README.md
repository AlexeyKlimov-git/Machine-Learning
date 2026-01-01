# Извлечение полей из обращений: Qwen3-0.6B

## Задача и данные

Получить JSON с product, amount и category из русского обращения.
Пропущенные сведения — null. Несколько неоднозначных продуктов/сумм — null.
product: card/deposit/transfer; category: problem/request/question.

examples/dev.jsonl — 4 few-shot примера. examples/test.jsonl — 16 отдельных
синтетических примеров, написанных специально для проекта. Это технический fixture,
а не репрезентативный банковский benchmark. Рабочих и персональных данных нет.
Инструкции разметки: examples/ANNOTATION.md.

## Что реализовано

- Консервативный rules baseline.
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), Transformers,
  zero-shot и few-shot с одинаковой моделью и test.
- Веса не обучаются: меняется контекст, градиентов и training loss здесь нет.
- Строгая схема Pydantic; без «починки» ответов и скрытого удаления неудачных примеров.
- Автоматическая оценка, SHA256 prompt/test, revision модели, latency и ошибки.

## Запуск

Python 3.11–3.13; первый Qwen-запуск скачает около 1.2 ГБ весов, CPU требует несколько ГБ RAM.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.evaluate --backend rules --output artifacts/rules
python -m src.evaluate --backend qwen --mode zero --output artifacts/qwen_zero
python -m src.evaluate --backend qwen --mode few --output artifacts/qwen_few
```

После изменения промпта повторите обе оценки в новых output-каталогах.
Если эта папка опубликована отдельным GitHub-репозиторием, workflow в .github
автоматически запускает тесты и все три оценки при push/PR. Он сохраняет отчеты,
но не объявляет улучшение по одному F1 и не задает искусственно удобных порогов.
В этой сессии проверены локальные команды; сам GitHub Actions ещё не запускался.
Для воспроизводимости укажите --revision с commit SHA из первого отчета.
Не подбирайте промпт на test: используйте отдельный расширенный dev.
Вся генерация локальная; обращения не отправляются внешнему API.
Декодирование greedy, thinking выключен; это фиксированный режим эксперимента,
а не обещание лучших настроек для Qwen.

## Как считается качество

- Field micro-F1: точное совпадение пар (имя поля, значение), null не считается сущностью.
- Неверная сумма дает один FP и один FN.
- Невалидный ответ дает FN по всем заполненным эталонным полям.
- Отдельно: valid_response_rate, exact match всей записи, accuracy по полям,
  false_fill_rate для полей, где в эталоне null.
- p50 wall time на CPU, batch=1, после одного прогрева; загрузка весов исключена.

Результаты: artifacts/<run>/metrics.json, predictions.json, errors.json.
Фактические измерения — reports/RESULTS.md.

## Ограничения и ошибки

Суммы прописью, несколько операций, косвенные просьбы, prompt injection.
Валидация JSON проверяет структуру, а не достоверность содержимого.
Greedy-генерация не гарантирует валидный JSON. Нет constrained decoding, валюты,
конвертации единиц и поддержки отрицательных сумм. Не используйте результат
для автоматических платежей или решений без проверки человеком.
На 16 синтетических примерах нельзя делать вывод о качестве в production.
