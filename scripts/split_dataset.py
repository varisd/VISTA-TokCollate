#!/usr/bin/env python3
"""Split a downloaded TSV corpus into train/validation text files.

The input file is expected to contain the downloaded columns from the corpus
builder, usually: bytes, language, text. This script keeps only the final text
column and writes:
  <input-stem>.train.txt
  <input-stem>.valid.txt

The validation split is intentionally kept small by default so that train.txt is
much larger than valid.txt.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List

DEFAULT_VALID_RATIO = 0.01
DEFAULT_MIN_VALID_SAMPLES = 100
DEFAULT_MAX_VALID_SAMPLES = 5000
DEFAULT_SEED = 42


def extract_texts(path: Path) -> List[str]:
    """Return only non-empty text values from a TSV corpus file."""
    if path.suffix.lower() != ".tsv":
        raise ValueError(f"Unsupported corpus format for {path}. Expected a .tsv file.")

    texts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            fields = [part.strip() for part in line.split("\t")]
            if len(fields) < 3:
                raise ValueError(
                    f"TSV row in {path} at line {line_number} does not contain the required columns: bytes, language, text."
                )

            if fields[0].lower() == "bytes" and fields[1].lower() in {"lang", "language"}:
                continue

            text = fields[-1]
            if text:
                texts.append(text)
    return texts


def compute_valid_count(total: int, valid_ratio: float, min_valid_samples: int, max_valid_samples: int) -> int:
    if total <= 1:
        return 0

    raw = total * valid_ratio
    valid_count = max(1, int(round(raw)))
    valid_count = max(min_valid_samples, valid_count)
    valid_count = min(valid_count, max_valid_samples)
    valid_count = min(valid_count, total - 1)
    return valid_count


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row}\n")


def split_dataset(
    input_path: Path,
    *,
    valid_ratio: float = DEFAULT_VALID_RATIO,
    min_valid_samples: int = DEFAULT_MIN_VALID_SAMPLES,
    max_valid_samples: int = DEFAULT_MAX_VALID_SAMPLES,
    seed: int = DEFAULT_SEED,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Shuffle and split a TSV dataset into train/valid text files."""
    if not input_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {input_path}")

    texts = extract_texts(input_path)
    if not texts:
        raise ValueError(f"No usable text rows were found in {input_path}.")

    rng = random.Random(seed)
    rng.shuffle(texts)

    valid_count = compute_valid_count(len(texts), valid_ratio, min_valid_samples, max_valid_samples)
    train_rows = texts[valid_count:]
    valid_rows = texts[:valid_count]

    if output_dir is None:
        output_dir = input_path.parent

    train_path = output_dir / f"{input_path.stem}.train.txt"
    valid_path = output_dir / f"{input_path.stem}.valid.txt"

    write_lines(train_path, train_rows)
    write_lines(valid_path, valid_rows)

    return train_path, valid_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a TSV corpus into train/valid text files using only the text column.")
    parser.add_argument("--input", type=Path, required=True, help="TSV corpus file produced by the download step.")
    parser.add_argument("--output-dir", type=Path, help="Optional directory to write the split files into; defaults to the input file's directory.")
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO, help="Fraction of rows reserved for validation; default 0.01.")
    parser.add_argument("--min-valid-samples", type=int, default=DEFAULT_MIN_VALID_SAMPLES, help="Minimum validation rows to keep.")
    parser.add_argument("--max-valid-samples", type=int, default=DEFAULT_MAX_VALID_SAMPLES, help="Cap the validation split size.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed used for shuffling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.input.parent
    train_path, valid_path = split_dataset(
        args.input,
        valid_ratio=args.valid_ratio,
        min_valid_samples=args.min_valid_samples,
        max_valid_samples=args.max_valid_samples,
        seed=args.seed,
        output_dir=output_dir,
    )

    print(f"Input rows: {sum(1 for _ in open(args.input, 'r', encoding='utf-8')) - 1:,}")
    print(f"Train rows: {sum(1 for _ in open(train_path, 'r', encoding='utf-8')):,}")
    print(f"Valid rows: {sum(1 for _ in open(valid_path, 'r', encoding='utf-8')):,}")
    print(f"Train output: {train_path}")
    print(f"Valid output: {valid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
