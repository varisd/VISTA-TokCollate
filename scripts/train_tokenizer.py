#!/usr/bin/env python3
"""Train a tokenizer on a corpus and save it under an output prefix.

Supported tokenizer families:
- bpe: standard byte-level BPE
- unigram: unigram LM tokenizer (with Metaspace pre-tokenization & decoding)
- wordpiece: WordPiece tokenizer (with WordPiece decoder)
- parity_aware_bpe: byte-level BPE with parity-aware vocabulary balancing within target vocab budget

Default save location:
    data/tokenizers/tokenizer.<tokenizer_type>.<vocab_size>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel, Metaspace, Whitespace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer


DEFAULT_OUTPUT_PREFIX = Path("data/tokenizers/tokenizer")
DEFAULT_SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]


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


def batch_iterator(
    corpus_path: Path,
    batch_size: int = 10_000,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> Iterator[List[str]]:
    """Stream batches of non-empty lines from disk to prevent OOM on large corpora."""
    batch: List[str] = []
    bytes_read = 0
    lines_read = 0

    with corpus_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if max_lines is not None and lines_read >= max_lines:
                break
            line_bytes = len(line.encode("utf-8"))
            if max_bytes is not None and (bytes_read + line_bytes) > max_bytes:
                break

            bytes_read += line_bytes
            lines_read += 1

            stripped = line.rstrip("\n")
            if stripped.strip():
                batch.append(stripped)
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
    tokenizer.pre_tokenizer = Metaspace()
    tokenizer.decoder = MetaspaceDecoder()
    return tokenizer


def build_wordpiece_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder(prefix="##")
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
        show_progress=True,
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
        show_progress=True,
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
        show_progress=True,
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
    batch_size: int,
    max_lines: Optional[int],
    max_bytes: Optional[int],
) -> None:
    # Reserve capacity inside vocab_size so adding booster tokens does not exceed the target budget
    per_parity_quota = max(2, vocab_size // 200)
    reserved_booster_slots = per_parity_quota * 2
    base_vocab_size = max(len(special_tokens) + 256, vocab_size - reserved_booster_slots)

    train_bpe(
        tokenizer,
        corpus_path,
        base_vocab_size,
        min_frequency,
        special_tokens,
        batch_size=batch_size,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )

    vocab = tokenizer.get_vocab()
    if not vocab:
        raise RuntimeError("Parity-aware training did not produce a vocabulary.")

    balanced_tokens: List[str] = []
    parity_map: dict[int, List[str]] = {0: [], 1: []}

    for token in vocab:
        if token in special_tokens:
            continue
        if not token or token.startswith("##"):
            continue
        if token.isascii() and token.isalpha():
            parity = sum(ord(ch) for ch in token) % 2
            parity_map[parity].append(token)

    remaining_slots = max(0, vocab_size - tokenizer.get_vocab_size())
    per_side = remaining_slots // 2

    for parity in (0, 1):
        bucket = sorted(parity_map[parity], key=lambda item: vocab[item], reverse=True)
        for token in bucket[:per_side]:
            candidate = f"<parity_{parity}_{token[:8]}>"
            if candidate not in vocab and candidate not in balanced_tokens:
                balanced_tokens.append(candidate)

    if balanced_tokens:
        tokenizer.add_special_tokens(balanced_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tokenization model on a large corpus via streaming and save it under an output prefix."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Plain-text corpus to train on. One document per line.",
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
        default=None,
        help="Maximum UTF-8 bytes to read from the corpus (e.g. 2000000000 for 2GB). Recommended for 50GB+ files, especially Unigram.",
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

    print(f"Saved tokenizer: {tokenizer_path}")
    print(f"Target vocab size: {args.vocab_size} | Actual vocab size: {tokenizer.get_vocab_size()}")
    print(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
