#!/usr/bin/env python3
"""Train a tokenizer from a TSV corpus (bytes\tlanguage\ttext) and save it under an output prefix.

Supported tokenizer families:
- bpe: standard byte-level BPE
- wordpiece: byte-level WordPiece tokenizer
- parity_aware_bpe: byte-level Parity-Aware BPE (using native ParityBpeTrainer and --ratios JSON)

Default save location:
    data/tokenizers/tokenizer.<tokenizer_type>.<vocab_size>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import Replace
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer, ParityBpeTrainer, WordPieceTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PREFIX = Path("data/tokenizers/tokenizer")
DEFAULT_SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]
MAX_DOC_CHARS = 50_000


def iter_tsv_records(
    corpus_path: Path,
    *,
    max_lines: int | None = None,
    max_bytes: int | None = None,
    target_lang: str | None = None,
) -> Iterator[tuple[str, str]]:
    """Stream (language, text) pairs from a .tsv file (bytes\tlanguage\ttext)."""
    if corpus_path.suffix.lower() != ".tsv":
        raise ValueError(f"Expected a .tsv corpus file, got: {corpus_path}")

    bytes_read = 0
    lines_read = 0

    with corpus_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if max_lines is not None and lines_read >= max_lines:
                break

            line_bytes = len(raw_line.encode("utf-8"))
            if max_bytes is not None and (bytes_read + line_bytes) > max_bytes:
                break

            bytes_read += line_bytes
            lines_read += 1

            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue

            fields = line.split("\t", 2)
            if len(fields) < 3:
                raise ValueError(
                    f"TSV row in {corpus_path} at line {line_number} must have 3 columns: bytes, language, text."
                )

            if fields[0].strip().lower() == "bytes" and fields[1].strip().lower() in {"lang", "language"}:
                continue

            lang = fields[1].strip()
            if target_lang is not None and lang != target_lang:
                continue

            text = fields[2]
            if text.strip():
                if len(text) > MAX_DOC_CHARS:
                    text = text[:MAX_DOC_CHARS]
                yield lang, text


def batch_iterator(
    corpus_path: Path,
    batch_size: int = 10_000,
    max_lines: int | None = None,
    max_bytes: int | None = None,
    target_lang: str | None = None,
) -> Iterator[list[str]]:
    """Stream batches of text strings from a TSV corpus without loading it into RAM."""
    batch: list[str] = []
    for _, text in iter_tsv_records(
        corpus_path, max_lines=max_lines, max_bytes=max_bytes, target_lang=target_lang
    ):
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_ratios(ratios_path: Path) -> dict[str, float]:
    """Load flat {lang: ratio} mapping from a .ratios.json file."""
    if not ratios_path.exists():
        raise FileNotFoundError(f"Ratios file not found: {ratios_path}")
    data = json.loads(ratios_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected a non-empty JSON object in {ratios_path}")
    ratios = {str(k): float(v) for k, v in data.items() if float(v) > 0.0}
    if not ratios:
        raise ValueError(f"No positive language ratios found in {ratios_path}")
    return ratios


def build_bytelevel_bpe() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def build_bytelevel_wordpiece() -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = DecoderSequence([
        Replace("##", ""),
        ByteLevelDecoder(),
    ])
    return tokenizer


def train_bpe(
    tokenizer: Tokenizer,
    corpus_path: Path,
    vocab_size: int,
    min_frequency: int,
    special_tokens: list[str],
    batch_size: int,
    max_lines: int | None,
    max_bytes: int | None,
) -> None:
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    iterator = batch_iterator(
        corpus_path, batch_size=batch_size, max_lines=max_lines, max_bytes=max_bytes
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)


def train_wordpiece(
    tokenizer: Tokenizer,
    corpus_path: Path,
    vocab_size: int,
    min_frequency: int,
    special_tokens: list[str],
    batch_size: int,
    max_lines: int | None,
    max_bytes: int | None,
) -> None:
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        continuing_subword_prefix="##",
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    iterator = batch_iterator(
        corpus_path, batch_size=batch_size, max_lines=max_lines, max_bytes=max_bytes
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)


def train_parity_aware_bpe(
    tokenizer: Tokenizer,
    corpus_path: Path,
    vocab_size: int,
    min_frequency: int,
    special_tokens: list[str],
    batch_size: int,
    max_lines: int | None,
    max_bytes: int | None,
    ratios_path: Path,
) -> None:
    corpus_ratios = load_ratios(ratios_path)
    languages = list(corpus_ratios.keys())
    ratios = list(corpus_ratios.values())

    logger.info("Parity-Aware BPE target sampling ratios: %s", corpus_ratios)

    initial_alphabet = ByteLevel.alphabet()
    num_merges = max(0, vocab_size - len(special_tokens) - len(initial_alphabet))

    train_iterators = [
        batch_iterator(
            corpus_path,
            batch_size=batch_size,
            max_lines=max_lines,
            max_bytes=max_bytes,
            target_lang=lang,
        )
        for lang in languages
    ]

    trainer = ParityBpeTrainer(
        num_merges=num_merges,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=initial_alphabet,
        show_progress=False,
    )
    trainer.train_from_iterator(tokenizer, train_iterators, ratio=ratios)


def add_hf_metadata(
    tokenizer: Tokenizer,
    output_dir: Path,
    *,
    corpus_path: Path,
    tokenizer_type: str,
    target_vocab_size: int,
    seed: int,
    max_bytes: int | None,
    max_lines: int | None,
    ratios_path: Path | None = None,
) -> None:
    config = {
        "tokenizer_class": tokenizer_type,
        "target_vocab_size": target_vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "corpus": str(corpus_path),
        "ratios_file": str(ratios_path) if ratios_path else None,
        "max_bytes": max_bytes,
        "max_lines": max_lines,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tokenizer from a TSV corpus (bytes\tlanguage\ttext)."
    )
    parser.add_argument("--corpus", type=Path, required=True, help="Input .tsv corpus file.")
    parser.add_argument(
        "--ratios",
        type=Path,
        default=None,
        help="Path to .ratios.json mapping language codes to proportions (required for parity_aware_bpe; ignored by bpe/wordpiece).",
    )
    parser.add_argument(
        "--tokenizer-type",
        choices=["bpe", "wordpiece", "parity_aware_bpe"],
        default="bpe",
        help="Tokenizer family to train.",
    )
    parser.add_argument("--vocab-size", type=int, default=32768, help="Target vocabulary size.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix (<output-prefix>.<tokenizer-type>.<vocab-size>).",
    )
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--special-tokens", nargs="*", default=DEFAULT_SPECIAL_TOKENS, help="Special tokens.")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Lines yielded per batch.")
    parser.add_argument("--max-bytes", type=int, default=None, help="Optional max UTF-8 bytes to read.")
    parser.add_argument("--max-lines", type=int, default=None, help="Optional max lines to read.")
    args = parser.parse_args()

    if args.tokenizer_type == "parity_aware_bpe" and args.ratios is None:
        parser.error("--ratios is required when --tokenizer-type=parity_aware_bpe.")
    return args


def main() -> int:
    args = parse_args()
    corpus_path = args.corpus.resolve()
    if not corpus_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {corpus_path}")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive.")

    output_dir = Path(f"{args.output_prefix}.{args.tokenizer_type}.{args.vocab_size}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = list(dict.fromkeys(args.special_tokens))
    tokenizer_type = args.tokenizer_type
    ratios_path = args.ratios.resolve() if args.ratios else None

    logger.info("Training %s tokenizer (vocab_size=%d, max_bytes=%s)", tokenizer_type, args.vocab_size, args.max_bytes)
    if ratios_path is not None and tokenizer_type != "parity_aware_bpe":
        logger.info("--ratios is ignored for --tokenizer-type=%s.", tokenizer_type)

    if tokenizer_type == "bpe":
        tokenizer = build_bytelevel_bpe()
        train_bpe(
            tokenizer, corpus_path, args.vocab_size, args.min_frequency,
            special_tokens, args.batch_size, args.max_lines, args.max_bytes,
        )
    elif tokenizer_type == "wordpiece":
        tokenizer = build_bytelevel_wordpiece()
        train_wordpiece(
            tokenizer, corpus_path, args.vocab_size, args.min_frequency,
            special_tokens, args.batch_size, args.max_lines, args.max_bytes,
        )
    elif tokenizer_type == "parity_aware_bpe":
        assert ratios_path is not None
        tokenizer = build_bytelevel_bpe()
        train_parity_aware_bpe(
            tokenizer, corpus_path, args.vocab_size, args.min_frequency,
            special_tokens, args.batch_size, args.max_lines, args.max_bytes,
            ratios_path=ratios_path,
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    add_hf_metadata(
        tokenizer,
        output_dir,
        corpus_path=corpus_path,
        tokenizer_type=tokenizer_type,
        target_vocab_size=args.vocab_size,
        seed=args.seed,
        max_bytes=args.max_bytes,
        max_lines=args.max_lines,
        ratios_path=ratios_path if tokenizer_type == "parity_aware_bpe" else None,
    )

    logger.info("Saved tokenizer: %s", tokenizer_path)
    logger.info("Target vocab size: %d | Actual vocab size: %d", args.vocab_size, tokenizer.get_vocab_size())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
