"""Compress JPG/JPEG images using OpenCV.

Examples:
    python scripts/compress_jpg.py photo.jpg
    python scripts/compress_jpg.py photo.jpg -o photo_small.jpg -q 70
    python scripts/compress_jpg.py images/ -o compressed/ -q 75 --max-width 1600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


JPG_EXTENSIONS = {".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress JPG/JPEG file sizes.")
    parser.add_argument("input", type=Path, help="Input JPG/JPEG file or directory.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output file or directory. Defaults to '<name>_compressed.jpg' for a "
            "file input, or '<input>_compressed' for a directory input."
        ),
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=75,
        help="JPEG quality from 1 to 100. Lower means smaller file. Default: 75.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        help="Resize image if it is wider than this value, preserving aspect ratio.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input is a directory, include subdirectories.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100.")

    if args.max_width is not None and args.max_width < 1:
        raise SystemExit("--max-width must be greater than 0.")

    if args.input.is_file() and args.input.suffix.lower() not in JPG_EXTENSIONS:
        raise SystemExit("Input file must have .jpg or .jpeg extension.")


def default_output_for_file(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_compressed.jpg")


def default_output_for_directory(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.name}_compressed")


def iter_jpg_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in JPG_EXTENSIONS
    )


def resize_if_needed(image, max_width: int | None):
    if max_width is None:
        return image

    height, width = image.shape[:2]
    if width <= max_width:
        return image

    scale = max_width / width
    new_size = (max_width, max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def compress_file(input_path: Path, output_path: Path, quality: int, max_width: int | None) -> int:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    image = resize_if_needed(image, max_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not write image: {output_path}")

    return output_path.stat().st_size


def compress_directory(args: argparse.Namespace) -> None:
    output_dir = args.output or default_output_for_directory(args.input)
    files = iter_jpg_files(args.input, args.recursive)
    if not files:
        raise SystemExit(f"No JPG/JPEG files found in: {args.input}")

    for input_path in files:
        relative_path = input_path.relative_to(args.input)
        output_path = (output_dir / relative_path).with_suffix(".jpg")
        before = input_path.stat().st_size
        after = compress_file(input_path, output_path, args.quality, args.max_width)
        print_result(input_path, output_path, before, after)


def compress_single_file(args: argparse.Namespace) -> None:
    output_path = args.output or default_output_for_file(args.input)
    if output_path.exists() and output_path.resolve() == args.input.resolve():
        raise SystemExit("Output file cannot be the same as input file.")

    before = args.input.stat().st_size
    after = compress_file(args.input, output_path, args.quality, args.max_width)
    print_result(args.input, output_path, before, after)


def print_result(input_path: Path, output_path: Path, before: int, after: int) -> None:
    saved = before - after
    saved_percent = (saved / before * 100) if before else 0
    print(
        f"{input_path} -> {output_path} | "
        f"{before / 1024:.1f} KB -> {after / 1024:.1f} KB "
        f"({saved_percent:.1f}% saved)"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.input.is_dir():
        compress_directory(args)
    else:
        compress_single_file(args)


if __name__ == "__main__":
    main()
