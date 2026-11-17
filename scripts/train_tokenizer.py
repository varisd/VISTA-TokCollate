#!/usr/bin/env python3
"""Train a tokenizer on a corpus and save it under an output prefix.

Supported tokenizer families:
- bpe: standard byte-level BPE
- wordpiece: WordPiece tokenizer (with ByteLevel pre-tokenization & lossless decoding)
- parity_aware_bpe: true fair-max Parity-Aware BPE (via HF tokenizers >= 0.23.2 ParityBpeTrainer)

Default save location:
    data/tokenizers/tokenizer.<tokenizer_type>.<vocab_size>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import Replace
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer, WordPieceTrainer

try:
    from tokenizers.trainers import ParityBpeTrainer
except ImportError:
    ParityBpeTrainer = None

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


def add_hf_metadata(
    tokenizer: Tokenizer,
    output_dir: Path,
    *,
    corpus_path: Path,
    tokenizer_type: str,
    target_vocab_size: int,
    seed: int,
    max_bytes: Optional[int],
    max_lines: Optional[int],
) -> None:
    config = {
        "tokenizer_class": tokenizer_type,
        "target_vocab_size": target_vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "corpus": str(corpus_path),
        "max_bytes": max_bytes,
        "max_lines": max_lines,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_corpus_line(raw_line: str, is_tsv: bool) -> Optional[tuple[str, str]]:
    """Return (language, text) from either a TSV (bytes\tlanguage\ttext) or plain-text line."""
    line = raw_line.rstrip("\r\n")
    if not line.strip():
        return None

    if is_tsv:
        parts = line.split("\t", 2)
        if len(parts) < 3:
            return None
        if parts[0].strip().lower() == "bytes" and parts[1].strip().lower() in {"lang", "language"}:
            return None
        lang = parts[1].strip() or "unknown"
        text = parts[2]
    else:
        lang = "unknown"
        text = line

    if not text.strip():
        return None
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS]
    return lang, text


def batch_iterator(
    corpus_path: Path,
    batch_size: int = 10_000,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> Iterator[List[str]]:
    """Stream batches of clean text documents from a .tsv or .txt corpus."""
    is_tsv = corpus_path.suffix.lower() == ".tsv"
    batch: List[str] = []
    bytes_read = 0
    lines_read = 0

    with corpus_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if max_lines is not None and lines_read >= max_lines:
                break
            line_bytes = len(raw_line.encode("utf-8"))
            if max_bytes is not None and (bytes_read + line_bytes) > max_bytes:
                break

            bytes_read += line_bytes
            lines_read += 1

            parsed = parse_corpus_line(raw_line, is_tsv)
            if parsed is None:
                continue

            _, text = parsed
            batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def discover_languages(
    corpus_path: Path,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> List[str]:
    """Scan the corpus slice once to discover all language codes present."""
    is_tsv = corpus_path.suffix.lower() == ".tsv"
    if not is_tsv:
        raise ValueError(
            f"parity_aware_bpe requires a .tsv corpus with (bytes, language, text) columns; got {corpus_path}"
        )

    langs: dict[str, int] = {}
    bytes_read = 0
    lines_read = 0

    with corpus_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if max_lines is not None and lines_read >= max_lines:
                break
            line_bytes = len(raw_line.encode("utf-8"))
            if max_bytes is not None and (bytes_read + line_bytes) > max_bytes:
                break
            bytes_read += line_bytes
            lines_read += 1

            parsed = parse_corpus_line(raw_line, is_tsv=True)
            if parsed is not None:
                lang, text = parsed
                langs[lang] = langs.get(lang, 0) + len(text.encode("utf-8"))

    ordered = sorted(langs.keys())
    logger.info("Discovered languages for Parity-Aware BPE (bytes in slice): %s", langs)
    return ordered


def language_stream_iterator(
    corpus_path: Path,
    target_lang: str,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> Iterator[str]:
    """Stream documents for a single language from a .tsv corpus without loading into RAM."""
    bytes_read = 0
    lines_read = 0

    with corpus_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if max_lines is not None and lines_read >= max_lines:
                break
            line_bytes = len(raw_line.encode("utf-8"))
            if max_bytes is not None and (bytes_read + line_bytes) > max_bytes:
                break
            bytes_read += line_bytes
            lines_read += 1

            parsed = parse_corpus_line(raw_line, is_tsv=True)
            if parsed is not None and parsed[0] == target_lang:
                yield parsed[1]


def build_bpe_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def build_wordpiece_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = DecoderSequence([
        Replace("##", ""),
        ByteLevelDecoder(),
    ])
    return tokenizer


def build_parity_aware_tokenizer() -> Tokenizer:
    """Build a byte-level BPE tokenizer for official fair-max Parity-Aware BPE."""
    return build_bpe_tokenizer()


def train_bpe(
    tokenizer: Tokenizer,
    corpus_path: Path,
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str],
    batch_size: int,
    max_lines: Optional[int],
    max_bytes: Optional[int],
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
    special_tokens: List[str],
    batch_size: int,
    max_lines: Optional[int],
    max_bytes: Optional[int],
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


def train_parity_aware(
    tokenizer: Tokenizer,
    corpus_path: Path,
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str],
    max_lines: Optional[int],
    max_bytes: Optional[int],
    dev_corpus_path: Optional[Path] = None,
    variant: str = "base",
) -> None:
    if ParityBpeTrainer is None:
        raise RuntimeError(
            "ParityBpeTrainer not found in installed `tokenizers` package. "
            "Upgrade with `pip install -U 'tokenizers>=0.23.2'` to use native Parity-Aware BPE."
        )

    languages = discover_languages(corpus_path, max_lines=max_lines, max_bytes=max_bytes)
    if not languages:
        raise ValueError(f"No languages discovered in {corpus_path}")

    # Total vocab = special_tokens + 256 base bytes + num_merges
    base_vocab = len(special_tokens) + len(ByteLevel.alphabet())
    num_merges = max(1, vocab_size - base_vocab)

    trainer = ParityBpeTrainer(
        num_merges=num_merges,
        variant=variant,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )

    train_iterators = [
        language_stream_iterator(
            corpus_path, lang, max_lines=max_lines, max_bytes=max_bytes
        )
        for lang in languages
    ]

    if dev_corpus_path is not None:
        if not dev_corpus_path.exists():
            raise FileNotFoundError(f"Parity dev corpus not found: {dev_corpus_path}")
        logger.info("Balancing merges using parallel dev corpus: %s", dev_corpus_path)
        dev_iterators = [
            language_stream_iterator(dev_corpus_path, lang)
            for lang in languages
        ]
        trainer.train_from_iterator(
            tokenizer,
            train_iterators=train_iterators,
            dev_iterators=dev_iterators,
        )
    else:
        # Equal target compression ratio (1.0 per language) enforces strict token-length parity
        # without requiring a multi-parallel dev set.
        equal_ratios = [1.0] * len(languages)
        logger.info("Balancing merges using equal target compression ratios: %s", dict(zip(languages, equal_ratios)))
        trainer.train_from_iterator(
            tokenizer,
            train_iterators=train_iterators,
            ratio=equal_ratios,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tokenization model on a .tsv or .txt corpus via streaming."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Training corpus (.tsv with bytes\tlang\ttext or plain-text .txt).",
    )
    parser.add_argument(
        "--dev-corpus",
        type=Path,
        default=None,
        help="Optional parallel dev .tsv corpus (e.g., FLORES) for parity_aware_bpe calibration.",
    )
    parser.add_argument(
        "--parity-variant",
        choices=["base", "window"],
        default="base",
        help="Merge balancing variant for ParityBpeTrainer (default: base).",
    )
    parser.add_argument(
        "--tokenizer-type",
        choices=["bpe", "wordpiece", "parity_aware_bpe"],
        default="bpe",
        help="Tokenizer family to train.",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32768, help="Target vocabulary size."
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output directory prefix. Final directory name is <output-prefix>.<tokenizer-type>.<vocabulary-size>",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency for training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stochastic training routines.",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=DEFAULT_SPECIAL_TOKENS,
        help="Special tokens to reserve in the tokenizer vocabulary.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Number of lines yielded per batch during streaming.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=500_000_000,
        help="Maximum UTF-8 bytes to read from the corpus (default: 500MB).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Maximum number of lines to read from the corpus.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_path = args.corpus.resolve()
    if not corpus_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {corpus_path}")

    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive.")

    output_dir = Path(
        f"{args.output_prefix}.{args.tokenizer_type}.{args.vocab_size}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = list(dict.fromkeys(args.special_tokens))
    tokenizer_type = args.tokenizer_type

    logger.info("Training tokenizer=%s | target_vocab_size=%d", tokenizer_type, args.vocab_size)
    if tokenizer_type == "bpe":
        tokenizer = build_bpe_tokenizer()
        train_bpe(
            tokenizer,
            corpus_path,
            args.vocab_size,
            args.min_frequency,
            special_tokens,
            batch_size=args.batch_size,
            max_lines=args.max_lines,
            max_bytes=args.max_bytes,
        )
    elif tokenizer_type == "wordpiece":
        tokenizer = build_wordpiece_tokenizer()
        train_wordpiece(
            tokenizer,
            corpus_path,
            args.vocab_size,
            args.min_frequency,
            special_tokens,
            batch_size=args.batch_size,
            max_lines=args.max_lines,
            max_bytes=args.max_bytes,
        )
    elif tokenizer_type == "parity_aware_bpe":
        tokenizer = build_parity_aware_tokenizer()
        train_parity_aware(
            tokenizer,
            corpus_path,
            args.vocab_size,
            args.min_frequency,
            special_tokens,
            max_lines=args.max_lines,
            max_bytes=args.max_bytes,
            dev_corpus_path=args.dev_corpus.resolve() if args.dev_corpus else None,
            variant=args.parity_variant,
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {args.tokenizer_type}")

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
    )

    logger.info("Saved tokenizer: %s", tokenizer_path)
    logger.info(
        "Target vocab size: %d | Actual vocab size: %d",
        args.vocab_size,
        tokenizer.get_vocab_size(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
