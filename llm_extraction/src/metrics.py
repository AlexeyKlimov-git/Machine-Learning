"""Micro-F1 по точным парам (имя поля, значение), без награды за null-null."""


def score(records):
    if not records:
        raise ValueError("Empty evaluation set")
    tp = fp = fn = 0
    exact = valid = absent = false_fill = 0
    per_field = {k: 0 for k in ("product", "amount", "category")}
    for record in records:
        gold = record["expected"]
        pred = record["prediction"]
        good = pred is not None
        valid += good
        exact += good and gold == pred
        predicted = pred or {}
        truth_pairs = {(k, v) for k, v in gold.items() if v is not None}
        pred_pairs = {(k, v) for k, v in predicted.items() if v is not None}
        tp += len(truth_pairs & pred_pairs)
        fp += len(pred_pairs - truth_pairs)
        fn += len(truth_pairs - pred_pairs)
        for key, value in gold.items():
            per_field[key] += good and predicted.get(key) == value
            if value is None:
                absent += 1
                false_fill += predicted.get(key) is not None
    # Невалидный JSON получает FN за все заполненные gold-поля. Отдельный
    # valid_response_rate не позволяет «выиграть», возвращая мусор на null-примерах.
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "field_micro_precision": precision,
        "field_micro_recall": recall,
        "field_micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "valid_response_rate": valid / len(records),
        "record_exact_match": exact / len(records),
        "false_fill_rate": false_fill / absent if absent else None,
        "field_accuracy": {k: v / len(records) for k, v in per_field.items()},
    }
