#!/usr/bin/env python3
"""Train a tokenizer on a TSV or plain-text corpus and save it under an output prefix.

Supported tokenizer families:
- bpe: standard byte-level BPE
- unigram: unigram LM tokenizer (with Metaspace pre-tokenization & decoding)
- wordpiece: WordPiece tokenizer (with ByteLevel pre-tokenization & lossless decoding)
- parity_aware_bpe: byte-level BPE with temperature-balanced language/script sampling

Supports both:
- TSV corpora (`bytes\tlanguage\ttext`), using exact language labels for `parity_aware_bpe`
  while stripping metadata columns before feeding text to the trainer.
- Plain-text corpora (`.txt`), one document per line (falling back to Unicode script detection
  for `parity_aware_bpe`).

Default save location:
    data/tokenizers/tokenizer.<tokenizer_type>.<vocab_size>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, Replace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel, Metaspace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

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


def _dominant_script(text: str) -> str:
    """Fallback classifier by dominant Unicode script when explicit language tags are absent."""
    counts: dict[str, int] = defaultdict(int)
    for ch in text[:512]:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            script = name.split(" ", 1)[0] if name else "OTHER"
            counts[script] += 1
    return max(counts, key=counts.get) if counts else "OTHER"


def iter_corpus_records(
    corpus_path: Path,
    *,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> Iterator[Tuple[str, str]]:
    """Stream (language_or_script, text) pairs from either a `.tsv` or plain-text file.

    Budget (`max_bytes` / `max_lines`) is tracked strictly on the extracted `text` payload.
    """
    is_tsv = corpus_path.suffix.lower() == ".tsv"
    bytes_read = 0
    lines_read = 0

    with corpus_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            if max_lines is not None and lines_read >= max_lines:
                break

            stripped = raw_line.rstrip("\r\n")
            if not stripped.strip():
                continue

            if is_tsv:
                fields = stripped.split("\t", maxsplit=2)
                if len(fields) < 3:
                    continue
                col0, lang, text = fields[0].strip(), fields[1].strip(), fields[2].strip()
                if col0.lower() == "bytes" and lang.lower() in {"lang", "language"}:
                    continue
                if not text:
                    continue
            else:
                text = stripped
                lang = ""

            if len(text) > MAX_DOC_CHARS:
                text = text[:MAX_DOC_CHARS]

            text_bytes = len(text.encode("utf-8"))
            if max_bytes is not None and (bytes_read + text_bytes) > max_bytes:
                break

            bytes_read += text_bytes
            lines_read += 1

            bucket = lang if lang else _dominant_script(text)
            yield bucket, text


def batch_iterator(
    corpus_path: Path,
    batch_size: int = 10_000,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> Iterator[List[str]]:
    """Stream batches of clean text documents from disk without metadata columns."""
    batch: List[str] = []
    for _, text in iter_corpus_records(corpus_path, max_lines=max_lines, max_bytes=max_bytes):
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def build_bpe_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def build_unigram_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = Metaspace(prepend_scheme="never", split=False)
    tokenizer.decoder = DecoderSequence([
        ByteFallback(),
        MetaspaceDecoder(prepend_scheme="never"),
    ])
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
    """Build a byte-level BPE tokenizer for parity-aware vocabulary balancing."""
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


def train_unigram(
    tokenizer: Tokenizer,
    corpus_path: Path,
    vocab_size: int,
    special_tokens: List[str],
    seed: int,
    batch_size: int,
    max_lines: Optional[int],
    max_bytes: Optional[int],
    shrinking_factor: float = 0.75,
    max_piece_length: int = 16,
) -> None:
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        unk_token="<unk>",
        seed=seed,
        shrinking_factor=shrinking_factor,
        max_piece_length=max_piece_length,
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


def parity_balanced_iterator(
    corpus_path: Path,
    batch_size: int = 10_000,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
    alpha: float = 0.2,
) -> Iterator[List[str]]:
    """Two-pass constant-memory streaming iterator for cross-language parity balancing.

    Uses exact `language` labels when reading a `.tsv` corpus (falling back to Unicode
    script detection for plain `.txt` files), and re-weights sampling via `bytes ** alpha`.
    """
    raw_bytes: dict[str, int] = defaultdict(int)
    for bucket, text in iter_corpus_records(corpus_path, max_lines=max_lines, max_bytes=max_bytes):
        raw_bytes[bucket] += len(text.encode("utf-8"))

    if not raw_bytes:
        return

    weights = {b_key: (b_val ** alpha) for b_key, b_val in raw_bytes.items()}
    total_w = sum(weights.values())
    total_target_bytes = sum(raw_bytes.values())

    repeats: dict[str, int] = {}
    for b_key, b_val in raw_bytes.items():
        target_b = total_target_bytes * (weights[b_key] / total_w)
        repeats[b_key] = max(1, round(target_b / max(1, b_val)))

    logger.info("Parity-aware byte counts per bucket: %s", dict(raw_bytes))
    logger.info("Parity-aware repeat multipliers: %s", repeats)

    batch: List[str] = []
    for bucket, text in iter_corpus_records(corpus_path, max_lines=max_lines, max_bytes=max_bytes):
        for _ in range(repeats[bucket]):
            batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def train_parity_aware(
    tokenizer: Tokenizer,
    corpus_path: Path,
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str],
    batch_size: int,
    max_lines: Optional[int],
    max_bytes: Optional[int],
    parity_alpha: float = 0.2,
) -> None:
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    iterator = parity_balanced_iterator(
        corpus_path,
        batch_size=batch_size,
        max_lines=max_lines,
        max_bytes=max_bytes,
        alpha=parity_alpha,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tokenization model on a TSV or plain-text corpus via streaming."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Input corpus (.tsv with bytes\tlanguage\ttext or plain .txt with one document per line).",
    )
    parser.add_argument(
        "--tokenizer-type",
        choices=["bpe", "unigram", "wordpiece", "parity_aware_bpe"],
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
        help="Maximum UTF-8 text bytes to read from the corpus (default: 500MB).",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Maximum number of documents to read from the corpus.",
    )
    parser.add_argument(
        "--parity-alpha",
        type=float,
        default=0.2,
        help="Temperature exponent for parity_aware_bpe sampling (0.0 = uniform, 1.0 = natural).",
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

    logger.info("Training tokenizer %s (vocab_size=%d, corpus=%s)", tokenizer_type, args.vocab_size, corpus_path)
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
    elif tokenizer_type == "unigram":
        tokenizer = build_unigram_tokenizer()
        train_unigram(
            tokenizer,
            corpus_path,
            args.vocab_size,
            special_tokens,
            args.seed,
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
            batch_size=args.batch_size,
            max_lines=args.max_lines,
            max_bytes=args.max_bytes,
            parity_alpha=args.parity_alpha,
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
    logger.info("Target vocab size: %d | Actual vocab size: %d", args.vocab_size, tokenizer.get_vocab_size())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
