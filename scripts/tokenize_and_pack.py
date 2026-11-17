#!/usr/bin/env python3
"""Tokenize a TSV corpus split and pack token IDs into fixed-length sequences."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_BATCH_SIZE = 4096
BOS_CANDIDATES = ("<bos>", "<cls>", "<s>")
EOS_CANDIDATES = ("<eos>", "<sep>", "</s>")


def iter_tsv_texts(path: Path) -> Iterator[str]:
    """Yield non-empty text records from a `bytes\tlanguage\ttext` TSV file."""
    if path.suffix.lower() != ".tsv":
        raise ValueError(f"Unsupported input format for {path}. Expected a .tsv file.")

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            parts = line.split("\t", maxsplit=2)
            if len(parts) < 3:
                raise ValueError(
                    f"TSV row in {path} at line {line_number} must have 3 columns: bytes, language, text."
                )

            if parts[0].strip().lower() == "bytes" and parts[1].strip().lower() in {"lang", "language"}:
                continue

            text = parts[2]
            if text.strip():
                yield text


def resolve_special_token_id(tokenizer: Tokenizer, candidates: tuple[str, ...]) -> int | None:
    """Return the first matching special token ID present in the tokenizer vocabulary."""
    for token in candidates:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return int(token_id)
    return None


def iter_batches(records: Iterator[str], batch_size: int) -> Iterator[list[str]]:
    """Group an iterator of strings into fixed-size lists."""
    batch: list[str] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def pack_tsv_to_token_ids(
    *,
    input_tsv: Path,
    tokenizer: Tokenizer,
    output_path: Path,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_segments: int = -1,
    prepend_cls: bool = False,
    include_sep: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, int]:
    """Stream TSV documents, batch-encode them, and write packed sequences of length `max_seq_len`."""
    if not input_tsv.exists():
        raise FileNotFoundError(f"Input TSV not found: {input_tsv}")
    if max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    bos_id = resolve_special_token_id(tokenizer, BOS_CANDIDATES) if prepend_cls else None
    eos_id = resolve_special_token_id(tokenizer, EOS_CANDIDATES) if include_sep else None

    if prepend_cls and bos_id is None:
        logger.warning("Requested --prepend-cls, but none of %s exist in vocabulary.", BOS_CANDIDATES)
    if include_sep and eos_id is None:
        logger.warning("Requested --include-sep, but none of %s exist in vocabulary.", EOS_CANDIDATES)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    segments_written = 0
    docs_processed = 0

    with output_path.open("w", encoding="utf-8") as out_handle:
        for text_batch in iter_batches(iter_tsv_texts(input_tsv), batch_size=batch_size):
            if max_segments > 0 and segments_written >= max_segments:
                break

            encodings = tokenizer.encode_batch(text_batch, add_special_tokens=False)
            for encoding in encodings:
                if max_segments > 0 and segments_written >= max_segments:
                    break

                doc_ids = encoding.ids
                if not doc_ids:
                    continue

                if bos_id is not None:
                    buffer.append(bos_id)
                buffer.extend(doc_ids)
                if eos_id is not None:
                    buffer.append(eos_id)
                docs_processed += 1

                while len(buffer) >= max_seq_len:
                    if max_segments > 0 and segments_written >= max_segments:
                        break
                    chunk = buffer[:max_seq_len]
                    buffer = buffer[max_seq_len:]
                    out_handle.write(" ".join(map(str, chunk)) + "\n")
                    segments_written += 1

    return docs_processed, segments_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize a TSV dataset split and pack token IDs into fixed-length sequences."
    )
    parser.add_argument(
        "--input-tsv",
        type=Path,
        required=True,
        help="Input TSV split file (bytes\tlanguage\ttext).",
    )
    parser.add_argument(
        "--tokenizer-prefix",
        type=Path,
        default=Path("data/tokenizers-natural/base"),
        help="Prefix path used when saving tokenizers (<prefix>.<tokenizer-type>.<vocab-size>/tokenizer.json).",
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        help="Tokenizer family name (e.g. bpe, wordpiece, parity_aware_bpe).",
    )
    parser.add_argument(
        "--tokenizer-vocabulary",
        type=int,
        required=True,
        help="Vocabulary size of the trained tokenizer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output path. Defaults to <input-tsv-dir>/<stem>.<type>.<vocab>.packed.txt",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Fixed token sequence length per packed output line.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=-1,
        help="Maximum packed sequences to write (-1 for unlimited).",
    )
    parser.add_argument(
        "--prepend-cls",
        action="store_true",
        help="Prepend BOS/CLS token ID before each document.",
    )
    parser.add_argument(
        "--include-sep",
        action="store_true",
        help="Append EOS/SEP token ID after each document.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of documents encoded per batch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer_path = Path(
        f"{args.tokenizer_prefix}.{args.tokenizer_type}.{args.tokenizer_vocabulary}/tokenizer.json"
    ).resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    output_path = (
        args.output
        if args.output is not None
        else args.input_tsv.parent
        / f"{args.input_tsv.stem}.{args.tokenizer_type}.{args.tokenizer_vocabulary}.packed.txt"
    )

    logger.info("Loading tokenizer: %s", tokenizer_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    logger.info("Packing %s -> %s (max_seq_len=%d)", args.input_tsv, output_path, args.max_seq_len)
    docs_processed, segments_written = pack_tsv_to_token_ids(
        input_tsv=args.input_tsv.resolve(),
        tokenizer=tokenizer,
        output_path=output_path.resolve(),
        max_seq_len=args.max_seq_len,
        max_segments=args.max_segments,
        prepend_cls=args.prepend_cls,
        include_sep=args.include_sep,
        batch_size=args.batch_size,
    )

    logger.info(
        "Completed packing: %d documents -> %d packed sequences of %d tokens (%s)",
        docs_processed,
        segments_written,
        args.max_seq_len,
        output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
