from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

random.seed(42)

CLASS_NAMES = ["fall", "not-fall"]
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_images(folder: Path):
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def augment_image(img: Image.Image) -> Image.Image:
    """Apply lightweight augmentations to simulate a minority-class oversampling."""
    # Random horizontal flip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    # Random rotation
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Random brightness/contrast tweaks
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.3))

    # Mild blur sometimes
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    # Small zoom/crop effect
    if random.random() < 0.6:
        width, height = img.size
        new_w = max(32, int(width * random.uniform(0.88, 1.0)))
        new_h = max(32, int(height * random.uniform(0.88, 1.0)))
        left = random.randint(0, max(0, width - new_w))
        top = random.randint(0, max(0, height - new_h))
        img = img.crop((left, top, left + new_w, top + new_h))
        img = img.resize((width, height), Image.BILINEAR)

    return img.convert("RGB")


def balance_dataset(source_dir: Path, output_dir: Path, dry_run: bool = False):
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()

    counts = {}
    for cls in CLASS_NAMES:
        cls_dir = source_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Thiếu folder: {cls_dir}")
        counts[cls] = len(read_images(cls_dir))

    max_count = max(counts.values())
    print("Dataset before balancing:")
    for cls in CLASS_NAMES:
        print(f" - {cls}: {counts[cls]} ảnh")
    print(f"Target count for each class: {max_count}")

    if dry_run:
        print("\nDry run: no files are written.")
        for cls in CLASS_NAMES:
            missing = max_count - counts[cls]
            if missing > 0:
                print(f" -> {cls} cần augment thêm {missing} ảnh")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cls in CLASS_NAMES:
        class_in_dir = source_dir / cls
        class_out_dir = output_dir / cls
        class_out_dir.mkdir(parents=True, exist_ok=True)

        original_files = read_images(class_in_dir)
        for idx, img_path in enumerate(original_files):
            img = Image.open(img_path).convert("RGB")
            img.save(class_out_dir / f"{img_path.stem}_orig{img_path.suffix}")

        if len(original_files) < max_count:
            needed = max_count - len(original_files)
            print(f"\nGenerating {needed} augmented images for class '{cls}'...")
            candidate_files = original_files[:]
            for i in range(needed):
                base_image_path = random.choice(candidate_files)
                base_img = Image.open(base_image_path).convert("RGB")
                augmented = augment_image(base_img)
                new_name = f"{base_image_path.stem}_aug_{i}{base_image_path.suffix}"
                augmented.save(class_out_dir / new_name)

    print(f"\nBalanced dataset saved to: {output_dir}")
    print("Final counts:")
    for cls in CLASS_NAMES:
        print(f" - {cls}: {len(read_images(output_dir / cls))} ảnh")


def parse_args():
    parser = argparse.ArgumentParser(description="Balance fall/not-fall dataset using augmentation.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("fall_dataset/images"),
        help="Folder containing subfolders fall and not-fall",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fall_dataset/images_balanced"),
        help="Folder to save balanced dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many images need augmentation without writing files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    balance_dataset(args.source_dir, args.output_dir, dry_run=args.dry_run)
