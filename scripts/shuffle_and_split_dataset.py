#!/usr/bin/env python3
"""Shuffle a large TSV corpus externally and split it into train/validation TSV files.

Expected TSV format (no header or optional header):
    bytes\tlanguage\ttext
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

DEFAULT_VALID_RATIO = 0.01
DEFAULT_MIN_VALID_SAMPLES = 100
DEFAULT_MAX_VALID_SAMPLES = 5000
DEFAULT_SEED = 42
DEFAULT_SHUFFLE_CHUNK_SIZE = 100_000


def iter_tsv_rows(path: Path) -> Iterator[str]:
    """Yield validated raw TSV lines (bytes\tlanguage\ttext)."""
    if path.suffix.lower() != ".tsv":
        raise ValueError(f"Unsupported corpus format for {path}. Expected a .tsv file.")

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            fields = line.split("\t", 2)
            if len(fields) < 3:
                raise ValueError(
                    f"TSV row in {path} at line {line_number} must have 3 tab-separated columns: bytes, language, text."
                )

            if fields[0].strip().lower() == "bytes" and fields[1].strip().lower() in {"lang", "language"}:
                continue

            if fields[2].strip():
                yield f"{fields[0].strip()}\t{fields[1].strip()}\t{fields[2]}\n"


def compute_valid_count(total: int, valid_ratio: float, min_valid_samples: int, max_valid_samples: int) -> int:
    if total <= 1:
        return 0
    valid_count = max(1, int(round(total * valid_ratio)))
    valid_count = max(min_valid_samples, valid_count)
    valid_count = min(valid_count, max_valid_samples, total - 1)
    return valid_count


def _write_shuffled_chunk(lines: list[str], rng: random.Random, temp_dir: Path, chunk_idx: int) -> Path:
    rng.shuffle(lines)
    chunk_path = temp_dir / f"chunk_{chunk_idx:05d}.tsv"
    with chunk_path.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)
    return chunk_path


def shuffle_tsv_file(
    input_path: Path,
    output_path: Path,
    *,
    seed: int = DEFAULT_SEED,
    chunk_size: int = DEFAULT_SHUFFLE_CHUNK_SIZE,
    temp_dir: Path | None = None,
    keep_temp: bool = False,
) -> int:
    """Shuffle a TSV file with bounded memory via chunked external shuffling. Returns total usable rows."""
    if not input_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {input_path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    base_temp_dir = temp_dir if temp_dir is not None else input_path.parent
    base_temp_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    chunk_paths: list[Path] = []
    current_chunk: list[str] = []
    total_rows = 0
    chunk_dir = None

    try:
        chunk_dir = Path(tempfile.mkdtemp(prefix=f"{input_path.stem}_shuffle_chunk_", dir=str(base_temp_dir)))
        for row in iter_tsv_rows(input_path):
            current_chunk.append(row)
            total_rows += 1
            if len(current_chunk) >= chunk_size:
                chunk_paths.append(_write_shuffled_chunk(current_chunk, rng, chunk_dir, len(chunk_paths)))
                current_chunk = []

        if current_chunk:
            chunk_paths.append(_write_shuffled_chunk(current_chunk, rng, chunk_dir, len(chunk_paths)))

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

    return total_rows


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
) -> tuple[Path, Path, int, int]:
    """Shuffle a TSV corpus externally and split it into .train.tsv and .valid.tsv files."""
    if not input_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {input_path}")

    out_dir = output_dir if output_dir is not None else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    scratch_base = Path(temp_dir) if temp_dir is not None else out_dir
    scratch_base.mkdir(parents=True, exist_ok=True)

    tmpdir = Path(tempfile.mkdtemp(prefix=f"{input_path.stem}_shuffle_", dir=str(scratch_base)))
    try:
        shuffled_path = tmpdir / f"{input_path.stem}.shuffled.tsv"
        total_rows = shuffle_tsv_file(
            input_path,
            shuffled_path,
            seed=seed,
            chunk_size=shuffle_chunk_size,
            temp_dir=tmpdir,
            keep_temp=True,
        )

        valid_count = compute_valid_count(total_rows, valid_ratio, min_valid_samples, max_valid_samples)
        train_count = total_rows - valid_count

        train_path = out_dir / f"{input_path.stem}.train.tsv"
        valid_path = out_dir / f"{input_path.stem}.valid.tsv"

        with shuffled_path.open("r", encoding="utf-8") as src_handle, \
             train_path.open("w", encoding="utf-8") as train_handle, \
             valid_path.open("w", encoding="utf-8") as valid_handle:
            for row_index, line in enumerate(src_handle):
                if row_index < valid_count:
                    valid_handle.write(line)
                else:
                    train_handle.write(line)
    finally:
        if not keep_temp and tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)

    return train_path, valid_path, train_count, valid_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shuffle a TSV corpus (bytes\tlanguage\ttext) and split into .train.tsv / .valid.tsv."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input .tsv corpus file.")
    parser.add_argument("--output-dir", type=Path, help="Output directory (defaults to input file directory).")
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO, help="Validation fraction (default: 0.01).")
    parser.add_argument("--min-valid-samples", type=int, default=DEFAULT_MIN_VALID_SAMPLES, help="Minimum validation rows.")
    parser.add_argument("--max-valid-samples", type=int, default=DEFAULT_MAX_VALID_SAMPLES, help="Maximum validation rows.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--shuffle-chunk-size", type=int, default=DEFAULT_SHUFFLE_CHUNK_SIZE, help="Rows per shuffle chunk in RAM.")
    parser.add_argument("--temp-dir", type=Path, help="Temporary directory for external shuffle chunks.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary chunks for debugging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_path, valid_path, train_count, valid_count = shuffle_and_split_dataset(
        args.input,
        valid_ratio=args.valid_ratio,
        min_valid_samples=args.min_valid_samples,
        max_valid_samples=args.max_valid_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        shuffle_chunk_size=args.shuffle_chunk_size,
        temp_dir=args.temp_dir,
        keep_temp=args.keep_temp,
    )

    logger.info("Total usable rows: %d", train_count + valid_count)
    logger.info("Train rows: %d -> %s", train_count, train_path)
    logger.info("Valid rows: %d -> %s", valid_count, valid_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
