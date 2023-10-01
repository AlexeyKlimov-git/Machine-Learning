"""Oxford-IIIT Pet с официальным test; demo отдельно и явно помечен."""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/pets")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    if a.demo:
        # Эти цвета проверяют обучение, сохранение весов и оценку. Они ничего
        # не говорят о способности распознавать настоящие фотографии животных.
        rng = random.Random(42)
        for split, count in (("train", 8), ("valid", 4), ("test", 4)):
            for name, color in (("cat", (190, 30, 30)), ("dog", (30, 190, 30))):
                folder = out / split / name
                folder.mkdir(parents=True)
                for i in range(count):
                    image = Image.new("RGB", (64, 64), color)
                    for _ in range(60):
                        image.putpixel(
                            (rng.randrange(64), rng.randrange(64)),
                            tuple(rng.randrange(256) for _ in range(3)),
                        )
                    image.save(folder / f"{i}.png")
        metadata = {"synthetic": True, "seed": 42, "purpose": "pipeline smoke only"}
    else:
        from torchvision.datasets import OxfordIIITPet

        download_root = out / "download"
        OxfordIIITPet(download_root, split="trainval", download=True)
        OxfordIIITPet(download_root, split="test", download=True)
        source = download_root / "oxford-iiit-pet"

        def annotations(split):
            return [
                line.split()
                for line in (source / "annotations" / (split + ".txt"))
                .read_text()
                .splitlines()
                if line and not line.startswith("#")
            ]

        pool, test = annotations("trainval"), annotations("test")
        train, valid = train_test_split(
            pool, test_size=0.2, random_state=42, stratify=[r[1] for r in pool]
        )
        hashes = {}
        for split, rows in (("train", train), ("valid", valid), ("test", test)):
            for row in rows:
                name = row[0]
                breed = name.rsplit("_", 1)[0]
                dest = out / split / breed
                dest.mkdir(parents=True, exist_ok=True)
                file = source / "images" / (name + ".jpg")
                shutil.copyfile(file, dest / file.name)
                hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        metadata = {
            "source": "Oxford-IIIT Pet",
            "synthetic": False,
            "seed": 42,
            "split": "official test; stratified 80/20 trainval",
            "image_sha256": hashes,
        }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
