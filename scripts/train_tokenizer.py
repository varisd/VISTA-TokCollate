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
import logging
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, Replace
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel, Metaspace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)

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
                if len(stripped) > 50_000:
                    stripped = stripped[:50_000]
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
        ByteLevelDecoder()
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
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    iterator = batch_iterator(
        corpus_path, batch_size=batch_size, max_lines=max_lines, max_bytes=max_bytes
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)


def _dominant_script(line: str) -> str:
    """Classify a document by its dominant Unicode script (e.g., LATIN, CYRILLIC, ARABIC, HAN)."""
    counts: dict[str, int] = defaultdict(int)
    # Sampling from the first 512 characters in a doc should be enough to determine the document script.
    for ch in line[:512]:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            script = name.split(" ", 1)[0] if name else "OTHER"
            counts[script] += 1
    return max(counts, key=counts.get) if counts else "OTHER"


def parity_balanced_iterator(
    corpus_path: Path,
    batch_size: int = 10_000,
    max_lines: Optional[int] = None,
    max_bytes: Optional[int] = None,
    alpha: float = 0.2,
) -> Iterator[List[str]]:
    """Re-weight document frequency across scripts using temperature sampling (p_s^alpha)
    so low-resource scripts achieve token-length parity with high-resource scripts.
    """
    docs_by_script: dict[str, List[str]] = defaultdict(list)
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
                docs_by_script[_dominant_script(stripped)].append(stripped)

    if not docs_by_script:
        return

    raw_bytes = {s: sum(len(d.encode("utf-8")) for d in docs) for s, docs in docs_by_script.items()}
    weights = {s: (b ** alpha) for s, b in raw_bytes.items()}
    total_w = sum(weights.values())
    total_target_bytes = sum(raw_bytes.values())

    batch: List[str] = []
    for script, docs in docs_by_script.items():
        target_bytes = int(total_target_bytes * (weights[script] / total_w))
        emitted = 0
        idx = 0
        while emitted < target_bytes:
            doc = docs[idx % len(docs)]
            batch.append(doc)
            emitted += len(doc.encode("utf-8"))
            idx += 1
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
) -> None:
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    iterator = parity_balanced_iterator(
        corpus_path,
        batch_size=batch_size,
        max_lines=max_lines,
        max_bytes=max_bytes,
        alpha=0.2,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)


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
        default=500_000_000,
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

    logger.info("Training tokenizer %s, vocabulary size %i", tokenizer_type, args.vocab_size)
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

    logger.info("Saved tokenizer: %s", tokenizer_path)
    logger.info("Target vocab size: %i | Actual vocab size: %i", args.vocab_size, tokenizer.get_vocab_size())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
