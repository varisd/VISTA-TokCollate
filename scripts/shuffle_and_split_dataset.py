#!/usr/bin/env python3
"""Shuffle a large TSV corpus and split it into train/validation text files.

This stage intentionally performs the randomization after download, because the
final corpus file can be tens of GB and should not be fully resident in memory.
The shuffle is implemented with a chunked external shuffle: we stream the input,
randomize chunks in memory, and then merge the shuffled chunks in a random order.
This keeps memory bounded while still giving a much better mix than local shard
batch shuffles performed during download.
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

DEFAULT_VALID_RATIO = 0.01
DEFAULT_MIN_VALID_SAMPLES = 100
DEFAULT_MAX_VALID_SAMPLES = 5000
DEFAULT_SEED = 42
DEFAULT_SHUFFLE_CHUNK_SIZE = 100_000


def iter_tsv_texts(path: Path) -> Iterator[str]:
    """Yield non-empty text values from a TSV corpus file."""
    if path.suffix.lower() != ".tsv":
        raise ValueError(f"Unsupported corpus format for {path}. Expected a .tsv file.")

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
                yield text


def compute_valid_count(total: int, valid_ratio: float, min_valid_samples: int, max_valid_samples: int) -> int:
    if total <= 1:
        return 0

    raw = total * valid_ratio
    valid_count = max(1, int(round(raw)))
    valid_count = max(min_valid_samples, valid_count)
    valid_count = min(valid_count, max_valid_samples)
    valid_count = min(valid_count, total - 1)
    return valid_count


def count_usable_rows(input_path: Path) -> int:
    return sum(1 for _ in iter_tsv_texts(input_path))


def shuffle_tsv_file(
    input_path: Path,
    output_path: Path,
    *,
    seed: int = DEFAULT_SEED,
    chunk_size: int = DEFAULT_SHUFFLE_CHUNK_SIZE,
    temp_dir: Path | None = None,
    keep_temp: bool = False,
) -> None:
    """Shuffle a TSV file with bounded memory via chunked external shuffling."""
    if not input_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {input_path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    base_temp_dir = temp_dir if temp_dir is not None else input_path.parent
    base_temp_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    chunk_paths: list[Path] = []
    current_chunk: list[str] = []
    chunk_dir = None

    try:
        chunk_dir = Path(tempfile.mkdtemp(prefix=f"{input_path.stem}_shuffle_chunk_", dir=str(base_temp_dir)))
        with input_path.open("r", encoding="utf-8") as source_handle:
            for raw_line in source_handle:
                if not raw_line.strip():
                    continue
                current_chunk.append(raw_line)
                if len(current_chunk) >= chunk_size:
                    chunk_paths.append(_write_shuffled_chunk(current_chunk, rng, chunk_dir))
                    current_chunk = []

        if current_chunk:
            chunk_paths.append(_write_shuffled_chunk(current_chunk, rng, chunk_dir))

        if not chunk_paths:
            raise ValueError(f"No usable rows found in {input_path}.")

        if len(chunk_paths) > 1:
            rng.shuffle(chunk_paths)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as output_handle:
            for chunk_path in chunk_paths:
                with chunk_path.open("r", encoding="utf-8") as chunk_handle:
                    shutil.copyfileobj(chunk_handle, output_handle)
    finally:
        if not keep_temp and chunk_dir is not None and chunk_dir.exists():
            shutil.rmtree(chunk_dir, ignore_errors=True)


def _write_shuffled_chunk(lines: list[str], rng: random.Random, temp_dir: Path) -> Path:
    rng.shuffle(lines)
    chunk_path = temp_dir / f"chunk_{len(list(temp_dir.iterdir())):04d}.tsv"
    with chunk_path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)
    return chunk_path


def shuffle_and_split_dataset(
    input_path: Path,
    *,
    valid_ratio: float = DEFAULT_VALID_RATIO,
    min_valid_samples: int = DEFAULT_MIN_VALID_SAMPLES,
    max_valid_samples: int = DEFAULT_MAX_VALID_SAMPLES,
    seed: int = DEFAULT_SEED,
    output_dir: Path | None = None,
    shuffle_chunk_size: int = DEFAULT_SHUFFLE_CHUNK_SIZE,
    temp_dir: Path | None = None,
    keep_temp: bool = False,
) -> tuple[Path, Path]:
    """Shuffle a TSV corpus externally, then split it into train/validation text files."""
    if not input_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {input_path}")

    total_rows = count_usable_rows(input_path)
    if total_rows == 0:
        raise ValueError(f"No usable text rows were found in {input_path}.")

    valid_count = compute_valid_count(total_rows, valid_ratio, min_valid_samples, max_valid_samples)

    if output_dir is None:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if temp_dir is None:
        temp_dir = input_path.parent
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = Path(tempfile.mkdtemp(prefix=f"{input_path.stem}_shuffle_", dir=str(temp_dir)))
    try:
        shuffled_path = tmpdir / f"{input_path.stem}.shuffled.tsv"
        shuffle_tsv_file(
            input_path,
            shuffled_path,
            seed=seed,
            chunk_size=shuffle_chunk_size,
            temp_dir=tmpdir,
            keep_temp=True,
        )

        train_path = output_dir / f"{input_path.stem}.train.txt"
        valid_path = output_dir / f"{input_path.stem}.valid.txt"

        with train_path.open("w", encoding="utf-8") as train_handle, \
             valid_path.open("w", encoding="utf-8") as valid_handle:
            for row_index, text in enumerate(iter_tsv_texts(shuffled_path)):
                if row_index < valid_count:
                    valid_handle.write(f"{text}\n")
                else:
                    train_handle.write(f"{text}\n")
    finally:
        if not keep_temp and tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)

    return train_path, valid_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shuffle a TSV corpus with bounded memory and split it into train/validation text files."
    )
    parser.add_argument("--input", type=Path, required=True, help="TSV corpus file produced by the download step.")
    parser.add_argument("--output-dir", type=Path, help="Optional directory to write the split files into; defaults to the input file's directory.")
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO, help="Fraction of rows reserved for validation; default 0.01.")
    parser.add_argument("--min-valid-samples", type=int, default=DEFAULT_MIN_VALID_SAMPLES, help="Minimum validation rows to keep.")
    parser.add_argument("--max-valid-samples", type=int, default=DEFAULT_MAX_VALID_SAMPLES, help="Cap the validation split size.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed used for shuffling.")
    parser.add_argument(
        "--shuffle-chunk-size",
        type=int,
        default=DEFAULT_SHUFFLE_CHUNK_SIZE,
        help="Number of lines to keep in RAM per shuffle chunk; lower values use less memory.",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        help="Directory for temporary shuffle chunks. Defaults to the input file's directory.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary chunk directory on disk after the run for debugging; default is to delete it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.input.parent
    train_path, valid_path = shuffle_and_split_dataset(
        args.input,
        valid_ratio=args.valid_ratio,
        min_valid_samples=args.min_valid_samples,
        max_valid_samples=args.max_valid_samples,
        seed=args.seed,
        output_dir=output_dir,
        shuffle_chunk_size=args.shuffle_chunk_size,
        temp_dir=args.temp_dir,
        keep_temp=args.keep_temp,
    )

    print(f"Input rows: {sum(1 for _ in open(args.input, 'r', encoding='utf-8')) - 1:,}")
    print(f"Train rows: {sum(1 for _ in open(train_path, 'r', encoding='utf-8')):,}")
    print(f"Valid rows: {sum(1 for _ in open(valid_path, 'r', encoding='utf-8')):,}")
    print(f"Train output: {train_path}")
    print(f"Valid output: {valid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
