from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a labeled image dataset into train/val/test folders."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("fall_dataset/images"),
        help="Folder containing class subfolders, e.g. fall/ and not-fall/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fall_dataset/splits"),
        help="Output root folder that will contain train/val/test folders.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio for train split. Default: 0.7",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Ratio for validation split. Default: 0.15",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio for test split. Default: 0.15",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them. Default behavior copies, not moves.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned split without copying or moving files.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not (abs(total - 1.0) < 1e-9):
        raise ValueError(
            f"Tổng ratio phải bằng 1.0, hiện tại: train={train_ratio}, val={val_ratio}, test={test_ratio}, total={total}"
        )
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Các ratio phải dương.")


def list_image_files(class_dir: Path):
    files = []
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return files


def split_files(files, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    # Ensure at least one file in each split when possible
    if total > 0:
        if total == 1:
            return {"train": [shuffled[0]], "val": [], "test": []}
        if train_count == 0:
            train_count = 1
            val_count = max(0, int(total * val_ratio))
            test_count = total - train_count - val_count
        if val_count == 0 and total >= 2:
            val_count = 1
            test_count = total - train_count - val_count

    if test_count < 0:
        test_count = 0
    if val_count < 0:
        val_count = 0

    if total > 0:
        idx_train = train_count
        idx_val = train_count + val_count
        train_files = shuffled[:idx_train]
        val_files = shuffled[idx_train:idx_val]
        test_files = shuffled[idx_val:]
    else:
        train_files = []
        val_files = []
        test_files = []

    return {"train": train_files, "val": val_files, "test": test_files}


def ensure_split_dirs(output_dir: Path):
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)


def copy_or_move_file(src: Path, dst: Path, move: bool) -> None:
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def main() -> int:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    class_dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class directories found under {input_dir}.")

    output_dir = args.output_dir.resolve()
    ensure_split_dirs(output_dir)

    total_summary = {"train": 0, "val": 0, "test": 0}
    print(f"[INFO] Input directory: {input_dir}")
    print(f"[INFO] Train/Val/Test ratios: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"[INFO] Seed: {args.seed}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = list_image_files(class_dir)
        if not files:
            print(f"[WARN] Class '{class_name}' has no images, skipped.")
            continue

        split = split_files(files, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

        for split_name, split_files_list in split.items():
            target_dir = output_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for file_path in split_files_list:
                target_file = target_dir / file_path.name
                print(
                    f"[{split_name.upper()}] {class_name}: {file_path.name} -> {target_file.relative_to(output_dir.parent)}"
                )

                if not args.dry_run:
                    if target_file.exists():
                        target_file.unlink()
                    copy_or_move_file(file_path, target_file, args.move)

            total_summary[split_name] += len(split_files_list)

    print("\nSummary:")
    for split_name, count in total_summary.items():
        print(f"  {split_name}: {count}")

    if args.dry_run:
        print("\n[INFO] Dry run only. No files were copied or moved.")
    else:
        print(f"\n[INFO] Dataset split completed. Output root: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
