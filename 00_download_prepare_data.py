"""
Prepare a configurable mudra dataset for training.

Recommended usage:
  python 00_download_prepare_data.py --source hf --out mudra_dataset --max-per-class 2500

This creates:
  mudra_dataset/
    train/
      pataka/
      tripataka/
      ...
    val/
      ...
    test/
      ...

If the Hugging Face dataset column names differ, run:
  python 00_download_prepare_data.py --source hf --list-classes

If you already downloaded/cloned a dataset locally, use:
  python 00_download_prepare_data.py --source local --local-dir /path/to/dataset --out mudra_dataset --max-per-class 2500
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from difflib import SequenceMatcher
from PIL import Image
from tqdm import tqdm

DEFAULT_TARGETS = {
    "pataka": ["pataka", "pathaka", "pataaka", "pathakam"],
    "tripataka": ["tripataka", "tripathaka", "tri pataka", "tripataaka"],
    "alapadma": ["alapadma", "alapadmam", "alapadma hasta", "alapadm"],
    "musti": ["musti", "mushti", "mushthi", "musti hasta", "mushti hasta"],
    "ardhapataka": ["ardhapataka", "ardha pataka", "ardhapatakam"],
    "ardhachandra": ["ardhachandra", "ardha chandra", "ardhachandram"],
    "mayura": ["mayura", "mayuram", "mayura hasta"],
    "shikhara": ["shikhara", "sikhara", "shikharam"],
    "kapittha": ["kapittha", "kapitha", "kapittham"],
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def normalize_name(s: str) -> str:
    return (
        str(s).lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("/", "")
        .replace("\\", "")
    )

def canonical_label(raw_label: str, targets):
    n = normalize_name(raw_label)

    # Exact/alias match first
    for canonical, aliases in targets.items():
        for a in aliases:
            if normalize_name(a) == n:
                return canonical

    # Fuzzy fallback
    best_label = None
    best_score = 0.0
    for canonical, aliases in targets.items():
        for a in aliases:
            score = SequenceMatcher(None, normalize_name(a), n).ratio()
            if score > best_score:
                best_score = score
                best_label = canonical

    if best_score >= 0.78:
        return best_label

    return None

def safe_clear_out(out: Path, targets):
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for cls in targets:
            (out / split / cls).mkdir(parents=True, exist_ok=True)

def save_image(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(img, Image.Image):
        img.convert("RGB").save(path, quality=95)
    else:
        Image.open(img).convert("RGB").save(path, quality=95)

def split_items(items, seed=42, train_ratio=0.75, val_ratio=0.15):
    random.seed(seed)
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }

def prepare_from_local(local_dir: Path, out: Path, targets, seed=42, max_per_class=None):
    """
    Expects a dataset folder with class folders somewhere inside.
    It recursively searches image files and uses the parent folder as the class label.
    """
    all_items = {k: [] for k in targets}

    for img_path in local_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        raw_class = img_path.parent.name
        canon = canonical_label(raw_class, targets)
        if canon:
            all_items[canon].append(img_path)

    for cls, paths in all_items.items():
        print(f"{cls}: found {len(paths)} images")

    if max_per_class is not None and max_per_class > 0:
        rng = random.Random(seed)
        for cls, paths in all_items.items():
            if len(paths) > max_per_class:
                all_items[cls] = rng.sample(paths, k=max_per_class)
                print(f"{cls}: capped to {max_per_class} images")

    safe_clear_out(out, targets)

    for cls, paths in all_items.items():
        split_map = split_items(paths, seed=seed)
        for split, split_paths in split_map.items():
            for i, src in enumerate(tqdm(split_paths, desc=f"{split}/{cls}")):
                dst = out / split / cls / f"{cls}_{i:05d}{src.suffix.lower()}"
                shutil.copy2(src, dst)

def prepare_from_hf(out: Path, targets, seed=42, list_classes=False, max_per_class=None):
    from datasets import load_dataset, Image as HFImage, ClassLabel

    ds = load_dataset("Samarth0710/bharatanatyam-mudra-dataset")

    # Find image and label columns robustly
    sample_split_name = list(ds.keys())[0]
    features = ds[sample_split_name].features

    image_col = None
    label_col = None

    for col, feature in features.items():
        if isinstance(feature, HFImage):
            image_col = col
        if isinstance(feature, ClassLabel):
            label_col = col

    # Fallback names
    if image_col is None:
        for c in ["image", "img", "file", "filepath"]:
            if c in features:
                image_col = c
                break

    if label_col is None:
        for c in ["label", "labels", "class", "category"]:
            if c in features:
                label_col = c
                break

    if image_col is None or label_col is None:
        print("Could not automatically identify columns.")
        print("Available columns/features:", features)
        raise RuntimeError("Update image_col/label_col manually in this script.")

    label_feature = features[label_col]

    def label_to_name(x):
        if isinstance(label_feature, ClassLabel):
            return label_feature.int2str(x)
        return str(x)

    if list_classes:
        names = set()
        for split_name in ds.keys():
            for row in ds[split_name]:
                names.add(label_to_name(row[label_col]))
        print("Classes found:")
        for name in sorted(names):
            print(" -", name)
        return

    all_items = {k: [] for k in targets}

    # Some HF datasets already have train split only; we collect then split ourselves
    for split_name in ds.keys():
        for idx, row in enumerate(tqdm(ds[split_name], desc=f"Scanning HF split {split_name}")):
            raw_label = label_to_name(row[label_col])
            canon = canonical_label(raw_label, targets)
            if canon:
                all_items[canon].append(row[image_col])

    for cls, items in all_items.items():
        print(f"{cls}: found {len(items)} images")

    if any(len(v) == 0 for v in all_items.values()):
        print("\nWARNING: One or more target classes had 0 images.")
        print("Run with --list-classes to inspect exact dataset class names.")

    if max_per_class is not None and max_per_class > 0:
        rng = random.Random(seed)
        for cls, items in all_items.items():
            if len(items) > max_per_class:
                all_items[cls] = rng.sample(items, k=max_per_class)
                print(f"{cls}: capped to {max_per_class} images")

    safe_clear_out(out, targets)

    for cls, items in all_items.items():
        split_map = split_items(items, seed=seed)
        for split, split_items_list in split_map.items():
            for i, img in enumerate(tqdm(split_items_list, desc=f"Saving {split}/{cls}")):
                dst = out / split / cls / f"{cls}_{i:05d}.jpg"
                save_image(img, dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hf", "local"], default="hf")
    parser.add_argument("--local-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default="mudra_dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list-classes", action="store_true")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS.keys()),
        help="Subset of mudras to include. Default includes 9 mudras."
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=2500,
        help="Maximum samples to keep per class before train/val/test split."
    )
    args = parser.parse_args()

    out = Path(args.out)

    unknown = [t for t in args.targets if t not in DEFAULT_TARGETS]
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Valid options: {list(DEFAULT_TARGETS.keys())}")
    targets = {k: DEFAULT_TARGETS[k] for k in args.targets}

    if args.source == "hf":
        prepare_from_hf(
            out,
            targets=targets,
            seed=args.seed,
            list_classes=args.list_classes,
            max_per_class=args.max_per_class
        )
    else:
        if not args.local_dir:
            raise ValueError("--local-dir is required when --source local")
        prepare_from_local(
            Path(args.local_dir),
            out,
            targets=targets,
            seed=args.seed,
            max_per_class=args.max_per_class
        )

    print("\nDone. Dataset prepared at:", out.resolve())

if __name__ == "__main__":
    main()
