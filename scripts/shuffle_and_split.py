#!/usr/bin/env python3
"""Shuffle a downloaded corpus and split it into train/dev text files.

The input may be:
- JSONL with a `text` field
- TSV with columns: bytes, lang, text
- plain text, one example per line

The output defaults to:
  data/corpus/train.txt
  data/corpus/dev.txt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List


DEFAULT_OUTPUT_DIR = Path("data/corpus")
DEFAULT_DEV_RATIO = 0.01
DEFAULT_SEED = 42


def extract_texts(path: Path) -> List[str]:
    """Read and return only the textual content from a corpus file."""
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        texts: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    text = obj.get("text")
                    if text is None:
                        raise ValueError(f"JSONL record in {path} did not contain a 'text' field.")
                    texts.append(str(text))
                else:
                    raise ValueError(f"Unsupported JSONL record in {path}: expected an object, got {type(obj).__name__}.")
        return texts

    if suffix == ".tsv":
        texts = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) < 3:
                    raise ValueError(f"TSV row in {path} does not contain the required columns: bytes, language, text.")
                texts.append(fields[-1])
        return texts

    if suffix in {".txt", ".text"}:
        texts = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                s = line.strip()
                if s:
                    texts.append(s)
        return texts

    raise ValueError(f"Unsupported corpus format for {path}. Use JSONL, TSV, or plain text.")


def split_indices(total: int, dev_ratio: float, min_dev_samples: int, max_dev_samples: int) -> int:
    if total <= 1:
        return 0

    raw = total * dev_ratio
    dev_count = max(1, int(round(raw)))
    dev_count = max(min_dev_samples, dev_count)
    dev_count = min(dev_count, max_dev_samples)
    dev_count = min(dev_count, total - 1)
    return dev_count


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle a corpus and split it into train/dev text files.")
    parser.add_argument("--input", type=Path, required=True, help="Corpus file: JSONL, TSV, or plain text.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write the split files into.")
    parser.add_argument("--dev-ratio", type=float, default=DEFAULT_DEV_RATIO, help="Fraction of rows to reserve for dev, default 0.01.")
    parser.add_argument("--min-dev-samples", type=int, default=100, help="Minimum dev rows to keep.")
    parser.add_argument("--max-dev-samples", type=int, default=5000, help="Cap the dev split size.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed used for shuffling.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {input_path}")

    texts = extract_texts(input_path)
    if not texts:
        raise ValueError(f"No usable text rows were found in {input_path}.")

    rng = random.Random(args.seed)
    rng.shuffle(texts)

    dev_count = split_indices(len(texts), args.dev_ratio, args.min_dev_samples, args.max_dev_samples)
    train_rows = texts[dev_count:]
    dev_rows = texts[:dev_count]

    train_path = args.output_dir / "train.txt"
    dev_path = args.output_dir / "dev.txt"

    write_lines(train_path, train_rows)
    write_lines(dev_path, dev_rows)

    print(f"Input rows: {len(texts):,}")
    print(f"Train rows: {len(train_rows):,}")
    print(f"Dev rows: {len(dev_rows):,}")
    print(f"Train output: {train_path}")
    print(f"Dev output: {dev_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
