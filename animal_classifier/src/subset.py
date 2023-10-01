"""Быстрый трехклассовый эксперимент с сохранением исходных границ split."""

import argparse
import json
import shutil
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/pets")
    p.add_argument("--output", default="data/pets3")
    p.add_argument("--classes", nargs="+", default=["Abyssinian", "beagle", "pug"])
    a = p.parse_args()
    source, target = Path(a.input), Path(a.output)
    # Классы задаем до просмотра качества. Test не прореживаем по сложности
    # и не переносим фото между split, иначе сравнение станет нечестным.
    for split in ("train", "valid", "test"):
        for label in a.classes:
            if Path(label).name != label or not (source / split / label).is_dir():
                raise ValueError("Unknown or invalid class")
    target.mkdir(parents=True, exist_ok=False)
    for split in ("train", "valid", "test"):
        for label in a.classes:
            shutil.copytree(source / split / label, target / split / label)
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "classes": a.classes,
                "synthetic": False,
                "note": "subset, not the full 37-class benchmark",
            },
            indent=2,
        )
    )
