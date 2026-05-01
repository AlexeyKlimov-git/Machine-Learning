"""Тонкие CLI-адаптеры. Закрытые бинарники и данные НЕ входят в проект."""

import csv
from datetime import datetime, timezone
from pathlib import Path


def utc(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("Timezone is required")
    return date.astimezone(timezone.utc)


def build(step, output, binaries):
    tool, args = step["tool"], step.get("args", {})
    if tool not in {"backtest", "dibate2csv", "diviz", "diribbon"}:
        raise ValueError(f"Unsupported tool: {tool}")
    binary = Path(binaries[tool]).expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"Missing local executable: {tool}")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [str(binary)]
    if tool == "backtest":
        config = Path(args["config"]).expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError("Missing strategy config")
        if utc(args["from"]) >= utc(args["to"]):
            raise ValueError("Empty or reversed period")
        # Не принимаем произвольные CLI-флаги и не подставляем credentials.
        # Даже local-провайдер может загрузить отсутствующий кэш из сети:
        # это НЕ гарантия офлайн-работы или изоляции.
        command += [
            "--config-path",
            str(config),
            "--save-dir-artifacts",
            str(output),
            "--dibate-name",
            "result",
            "--date-from",
            args["from"],
            "--date-to",
            args["to"],
            "--snapshot-label",
            args["snapshot"],
            "--data-provider-type",
            "local",
            "--local-cache-directory",
            str(Path(args["cache"]).expanduser().resolve()),
        ]
        expected = output / "result.dibate"
    elif tool == "diribbon":
        folder = Path(args["input"]).expanduser().resolve()
        inputs = sorted(folder.glob("*.dibate"))
        if not inputs or any(not p.is_file() for p in inputs):
            raise ValueError("Expected nonempty directory of dibate files")
        if output == folder or folder in output.parents:
            raise ValueError("Merged report must stay outside input directory")
        mode = args.get("capital_mode", "first-sum")
        if mode not in {"first-sum", "sum", "max"}:
            raise ValueError("Unknown capital convention")
        expected = output / "merged.dibate"
        command += [
            str(folder),
            "--output",
            str(expected),
            "--itm-mode",
            mode,
            "--save-strategy-reports",
            "true",
        ]
    else:
        source = Path(args["input"]).expanduser().resolve()
        if not source.is_file() or not source.stat().st_size:
            raise ValueError("Missing or empty dibate")
        if tool == "dibate2csv":
            # Ровно один mode: несколько режимов могут частично завершиться,
            # оставив неоднозначный набор выходных файлов.
            command += [
                "--dibate-path",
                str(source),
                "--modes",
                "statistics",
                "--result-dir",
                str(output),
            ]
            expected = output / "statistics.csv"
        else:
            expected = output / "report.png"
            command += [str(source), "--output", str(expected)]
    return command, expected


def validate_period(csv_path, start, end, tolerance_seconds=60):
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("Expected exactly one aggregate statistics row")
    actual_start, actual_end = utc(rows[0]["Start"]), utc(rows[0]["End"])
    # Для минутных данных последняя свеча может быть на минуту раньше границы.
    if abs((actual_start - utc(start)).total_seconds()) > tolerance_seconds:
        raise ValueError("Actual start differs from requested period")
    if abs((actual_end - utc(end)).total_seconds()) > tolerance_seconds:
        raise ValueError(
            "Actual end differs from requested period; snapshot may be truncated"
        )
    if actual_end < actual_start:
        raise ValueError("Reversed report period")
