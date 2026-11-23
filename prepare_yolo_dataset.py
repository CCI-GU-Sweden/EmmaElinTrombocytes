"""Prepare a YOLO detection dataset from download_all_data_for_training.py output.

Edit the settings below and run this file. Split counts are rounded to the
nearest whole image (half up). Test and validation assignments are saved in the
source folder and reused on later runs. Training subsets use a fixed seed and
are nested: a smaller training set is contained in every larger one.

Test and validation sets are fixed at 20% and 10% of the manifest images, respectively.

DOUBLE CHECK THIS: Test and validation sets are identical in each training run.

YOLO labels use class 0 for cell and 1 for granule, followed by normalized
center-x, center-y, width and height. For 80% training, validation images are
included in training and no validation images are exported.
"""

SOURCE_DIRECTORY = "training_data"
TARGET_DIRECTORY = "yolo_dataset"
TRAIN_SPLIT = 20  # 20, 40, 60, 70, or 80 percent of all manifest images
RANDOM_SEED = 42

import csv
import math
import random
import shutil
from pathlib import Path

from PIL import Image

CLASSES = {"cell": 0, "granule": 1}
ID_COLUMNS = ("dataset_id", "image_id")


def read_ids(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not set(ID_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{path} must have dataset_id and image_id columns")
        ids = []
        for row in reader:
            try:
                pair = tuple(int(row[key]) for key in ID_COLUMNS)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid IDs in {path}: {row}") from exc
            if any(value <= 0 for value in pair):
                raise ValueError(f"IDs must be positive in {path}: {row}")
            ids.append(pair)
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image entries in {path}")
    return ids


def write_ids(path, ids):
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(ID_COLUMNS)
        writer.writerows(ids)


def count(fraction, total):
    return math.floor(fraction * total + 0.5)


def load_splits(source, all_ids):
    """Keep existing assignments; generate missing lists from unassigned IDs."""
    test_path = source / "test_images.csv"
    val_path = source / "validation_images.csv"
    test = read_ids(test_path) if test_path.exists() else None
    val = read_ids(val_path) if val_path.exists() else None
    available = set(all_ids)
    for path, ids in ((test_path, test), (val_path, val)):
        if ids is not None and not set(ids) <= available:
            raise ValueError(f"{path} contains images absent from the manifest")
    if test is not None and val is not None and set(test) & set(val):
        raise ValueError("Test and validation lists overlap")
    rng = random.Random(RANDOM_SEED)
    if test is None:
        candidates = sorted(available - set(val or []))
        rng.shuffle(candidates)
        size = count(0.2, len(all_ids))
        if len(candidates) < size:
            raise ValueError("Not enough images to create the test split")
        test = candidates[:size]
        write_ids(test_path, test)
    if val is None:
        candidates = sorted(available - set(test))
        rng.shuffle(candidates)
        size = count(0.1, len(all_ids))
        if len(candidates) < size:
            raise ValueError("Not enough images to create the validation split")
        val = candidates[:size]
        write_ids(val_path, val)
    return test, val


def label_lines(path, width, height):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"class", "x", "y", "width", "height"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {', '.join(sorted(required))}")
        lines = []
        for row in reader:
            if row["class"] not in CLASSES:
                raise ValueError(f"Unknown class in {path}: {row['class']!r}")
            try:
                x, y, w, h = (float(row[key]) for key in ("x", "y", "width", "height"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid box in {path}: {row}") from exc
            if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
                raise ValueError(f"Box outside image or invalid in {path}: {row}")
            lines.append(f"{CLASSES[row['class']]} {(x + w / 2) / width:.8f} {(y + h / 2) / height:.8f} {w / width:.8f} {h / height:.8f}\n")
        return lines


def export_split(source, target, split, ids):
    image_dir = target / split / "images"
    label_dir = target / split / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for dataset_id, image_id in ids:
        stem = f"{dataset_id}_{image_id}"
        image_path = source / f"{stem}.png"
        with Image.open(image_path) as image:
            width, height = image.size
        lines = label_lines(source / f"{stem}.csv", width, height)
        shutil.copy2(image_path, image_dir / image_path.name)
        (label_dir / f"{stem}.txt").write_text("".join(lines), encoding="utf-8")


def main():
    if TRAIN_SPLIT not in (20, 40, 60, 70, 80):
        raise ValueError("TRAIN_SPLIT must be 20, 40, 60, 70 or 80")
    source = Path(SOURCE_DIRECTORY).expanduser().resolve()
    target = Path(TARGET_DIRECTORY).expanduser().resolve()
    if source == target or source in target.parents:
        raise ValueError("TARGET_DIRECTORY must be outside SOURCE_DIRECTORY")
    all_ids = read_ids(source / "manifest.csv")
    if not all_ids:
        raise ValueError("The manifest is empty")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(f"Target directory must be empty: {target}")
    test, val = load_splits(source, all_ids)
    candidates = sorted(set(all_ids) - set(test) - set(val))
    random.Random(RANDOM_SEED).shuffle(candidates)
    train_size = count(TRAIN_SPLIT / 100, len(all_ids))
    if TRAIN_SPLIT == 80:
        train = candidates + val
        # Existing split files are authoritative, even if the manifest changes.
        if len(train) != train_size:
            print(f"Using all {len(train)} non-test images for 80% training")
        exported_val = []
    else:
        if train_size > len(candidates):
            raise ValueError("Not enough images outside the saved test and validation splits")
        train = candidates[:train_size]
        exported_val = val
    for name, ids in (("train", train), ("val", exported_val), ("test", test)):
        export_split(source, target, name, ids)
        print(f"{name}: {len(ids)} images")
    # Ultralytics expects a validation path; for the 80% set the folder is empty.
    (target / "data.yaml").write_text(
        f"path: {target.as_posix()!r}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: cell\n  1: granule\n",
        encoding="utf-8",
    )
    print(f"Dataset written to {target}")


if __name__ == "__main__":
    main()
