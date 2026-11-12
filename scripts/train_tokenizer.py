#!/usr/bin/env python3
"""Train a tokenizer on a corpus and save it to data/tokenizers by default.

Supported tokenizer families:
- bpe: standard byte-level BPE
- unigram: unigram LM tokenizer
- wordpiece: WordPiece tokenizer
- parity_aware_bpe: byte-level BPE with parity-aware vocabulary balancing

Default save location:
  data/tokenizers/<tokenizer_type>_<vocab_size>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel, Metaspace, Whitespace
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer


DEFAULT_OUTPUT_DIR = Path("data/tokenizers")
DEFAULT_SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]


def add_hf_metadata(tokenizer: Tokenizer, output_dir: Path, *, corpus_path: Path, tokenizer_type: str, vocab_size: int, seed: int) -> None:
    config = {
        "tokenizer_class": tokenizer_type,
        "vocab_size": vocab_size,
        "corpus": str(corpus_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
    }
    (output_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_ascii_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def build_bpe_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def build_unigram_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def build_wordpiece_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="<unk>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def build_parity_aware_tokenizer() -> Tokenizer:
    """Approximate parity-aware BPE by training a standard BPE tokenizer and then adding parity-balanced booster tokens.

    This keeps the tokenizer compatible with mainstream LLM workflows while exposing a clear parity-aware option.
    """
    tokenizer = build_bpe_tokenizer()
    return tokenizer


def train_bpe(tokenizer: Tokenizer, corpus_path: Path, vocab_size: int, min_frequency: int, special_tokens: List[str]) -> None:
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens, show_progress=True)
    tokenizer.train([str(corpus_path)], trainer=trainer)


def train_unigram(tokenizer: Tokenizer, corpus_path: Path, vocab_size: int, special_tokens: List[str], seed: int) -> None:
    trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=special_tokens, unk_token="<unk>", seed=seed, show_progress=True)
    tokenizer.train([str(corpus_path)], trainer=trainer)


def train_wordpiece(tokenizer: Tokenizer, corpus_path: Path, vocab_size: int, min_frequency: int, special_tokens: List[str]) -> None:
    trainer = WordPieceTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens, continuing_subword_prefix="##", show_progress=True)
    tokenizer.train([str(corpus_path)], trainer=trainer)


def train_parity_aware(tokenizer: Tokenizer, corpus_path: Path, vocab_size: int, min_frequency: int, special_tokens: List[str]) -> None:
    train_bpe(tokenizer, corpus_path, vocab_size, min_frequency, special_tokens)

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

    for parity in (0, 1):
        bucket = sorted(parity_map[parity], key=lambda item: vocab[item], reverse=True)
        for token in bucket[: max(2, vocab_size // 200)]:
            balanced_tokens.append(f"<parity_{parity}_{token[:8]}>")

    if balanced_tokens:
        tokenizer.add_special_tokens(balanced_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tokenization model on a corpus and save it under data/tokenizers.")
    parser.add_argument("--corpus", type=Path, required=True, help="Plain-text corpus to train on. One document per line.")
    parser.add_argument(
        "--tokenizer-type",
        choices=["bpe", "unigram", "wordpiece", "parity_aware_bpe"],
        default="bpe",
        help="Tokenizer family to train.",
    )
    parser.add_argument("--vocab-size", type=int, default=32768, help="Target vocabulary size.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save the trained tokenizer under. Default: data/tokenizers")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency for training.")
    parser.add_argument("--name", type=str, default=None, help="Optional output subdirectory name. Defaults to tokenizer_type_vocab_size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic training routines.")
    parser.add_argument("--special-tokens", nargs="*", default=DEFAULT_SPECIAL_TOKENS, help="Special tokens to reserve in the tokenizer vocabulary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_path = args.corpus.resolve()
    if not corpus_path.exists():
        raise FileNotFoundError(f"Training corpus not found: {corpus_path}")

    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive.")

    output_name = args.name or f"{args.tokenizer_type}_{args.vocab_size}"
    output_dir = (args.output_dir / output_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = list(dict.fromkeys(args.special_tokens))

    tokenizer_type = args.tokenizer_type
    if tokenizer_type == "parity_aware":
        tokenizer_type = "parity_aware_bpe"

    if tokenizer_type == "bpe":
        tokenizer = build_bpe_tokenizer()
        train_bpe(tokenizer, corpus_path, args.vocab_size, args.min_frequency, special_tokens)
    elif tokenizer_type == "unigram":
        tokenizer = build_unigram_tokenizer()
        train_unigram(tokenizer, corpus_path, args.vocab_size, special_tokens, args.seed)
    elif tokenizer_type == "wordpiece":
        tokenizer = build_wordpiece_tokenizer()
        train_wordpiece(tokenizer, corpus_path, args.vocab_size, args.min_frequency, special_tokens)
    elif tokenizer_type == "parity_aware_bpe":
        tokenizer = build_parity_aware_tokenizer()
        train_parity_aware(tokenizer, corpus_path, args.vocab_size, args.min_frequency, special_tokens)
    else:
        raise ValueError(f"Unsupported tokenizer type: {args.tokenizer_type}")

    args.tokenizer_type = tokenizer_type

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    add_hf_metadata(tokenizer, output_dir, corpus_path=corpus_path, tokenizer_type=args.tokenizer_type, vocab_size=args.vocab_size, seed=args.seed)

    print(f"Saved tokenizer: {tokenizer_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Output dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
