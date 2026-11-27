#!/usr/bin/env python3
"""Train a tokenizer from a TSV corpus (bytes\tlanguage\ttext) and save it under an output prefix.

Supported tokenizer families:
- bpe: standard byte-level BPE
- wordpiece: byte-level WordPiece tokenizer
- parity_aware_bpe: byte-level Parity-Aware BPE (using native ParityBpeTrainer and --ratios JSON)
- evolutionary: evolutionary tokenizer (UFAL evolutionary-tokenizer submodule)

Default save location:
    data/tokenizers/tokenizer.<tokenizer_type>.<vocab_size>
"""
from __future__ import annotations

import argparse
import json
import logging
import re
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


def slugify_language(language: str) -> str:
    """Convert language labels to filesystem-safe names."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", language.strip())
    return value or "unknown"


def prepare_evolutionary_language_files(
    corpus_path: Path,
    output_dir: Path,
    *,
    max_lines: int | None,
    max_bytes: int | None,
) -> list[Path]:
    """Write one UTF-8 text file per language from TSV rows for evolutionary training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, object] = {}
    language_files: dict[str, Path] = {}

    try:
        for lang, text in iter_tsv_records(corpus_path, max_lines=max_lines, max_bytes=max_bytes):
            lang_key = slugify_language(lang)
            if lang_key not in handles:
                lang_path = output_dir / f"{lang_key}.txt"
                language_files[lang_key] = lang_path
                handles[lang_key] = lang_path.open("w", encoding="utf-8")
            handle = handles[lang_key]
            handle.write(text)
            handle.write("\n")
    finally:
        for handle in handles.values():
            handle.close()

    files = sorted(language_files.values())
    if not files:
        raise ValueError("No language data found in corpus for evolutionary training.")
    return files


def train_evolutionary(
    *,
    corpus_path: Path,
    output_dir: Path,
    vocab_size: int,
    max_lines: int | None,
    max_bytes: int | None,
    submodule_dir: Path,
    evo_tokenizer_type: str,
    evo_generations: int,
    evo_population_size: int,
    evo_elite_size: int,
    evo_mutation_rate: float,
    evo_crossover_rate: float,
    evo_mutation_count: int,
    evo_ngram_max_length: int,
    evo_ngram_top_k: int,
    evo_num_bpe_individuals: int | None,
    evo_num_random_ngram_individuals: int,
    evo_weight_compression: float,
    evo_weight_unk_penalty: float,
    evo_weight_token_usage: float,
    evo_weight_cross_lingual_consistency: float,
    evo_weight_diversity: float,
) -> int:
    """Train the UFAL evolutionary tokenizer and return resulting vocabulary size."""
    if not submodule_dir.exists():
        raise FileNotFoundError(
            f"Evolutionary tokenizer submodule directory not found: {submodule_dir}"
        )

    if str(submodule_dir) not in sys.path:
        sys.path.insert(0, str(submodule_dir))

    try:
        from evolutionary_training import train_evolutionary_tokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Could not import evolutionary_training from the evolutionary-tokenizer submodule."
        ) from exc

    corpora_dir = output_dir / "evolutionary_corpora"
    language_files = prepare_evolutionary_language_files(
        corpus_path,
        corpora_dir,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )
    logger.info(
        "Prepared %d language files for evolutionary training at %s",
        len(language_files),
        corpora_dir,
    )

    train_evolutionary_tokenizer(
        monolingual_data_files=[str(p) for p in language_files],
        parallel_data_files=[str(p) for p in language_files],
        vocab_size=vocab_size,
        generations=evo_generations,
        population_size=evo_population_size,
        elite_size=evo_elite_size,
        mutation_rate=evo_mutation_rate,
        crossover_rate=evo_crossover_rate,
        fitness_weights={
            "compression": evo_weight_compression,
            "unk_penalty": evo_weight_unk_penalty,
            "token_usage": evo_weight_token_usage,
            "cross_lingual_consistency": evo_weight_cross_lingual_consistency,
            "diversity": evo_weight_diversity,
        },
        tokenizer_type=evo_tokenizer_type,
        enabled_mutations={
            "replace_unused": True,
            "random_ngram": True,
        },
        mutation_count=evo_mutation_count,
        parallel_bpe_init=True,
        ngram_max_length=evo_ngram_max_length,
        ngram_top_k=evo_ngram_top_k,
        num_bpe_individuals=evo_num_bpe_individuals,
        num_random_ngram_individuals=evo_num_random_ngram_individuals,
        parity_bpe_files=None,
        experiment_dir=str(output_dir),
    )

    vocabulary_path = output_dir / "vocabulary.json"
    if not vocabulary_path.exists():
        raise FileNotFoundError(
            f"Evolutionary training finished without vocabulary.json at {vocabulary_path}"
        )

    data = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary = data.get("vocabulary", [])
    if not isinstance(vocabulary, list) or not vocabulary:
        raise ValueError(f"Invalid or empty vocabulary in {vocabulary_path}")
    return len(vocabulary)


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


def add_evolutionary_metadata(
    output_dir: Path,
    *,
    corpus_path: Path,
    target_vocab_size: int,
    actual_vocab_size: int,
    seed: int,
    max_bytes: int | None,
    max_lines: int | None,
    evolutionary_config: dict[str, int | float | str | None],
) -> None:
    config = {
        "tokenizer_class": "evolutionary",
        "target_vocab_size": target_vocab_size,
        "actual_vocab_size": actual_vocab_size,
        "corpus": str(corpus_path),
        "max_bytes": max_bytes,
        "max_lines": max_lines,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "evolutionary": evolutionary_config,
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
        choices=["bpe", "wordpiece", "parity_aware_bpe", "evolutionary"],
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

    # Evolutionary tokenizer options
    parser.add_argument(
        "--evolutionary-submodule-dir",
        type=Path,
        default=Path("evolutionary-tokenizer"),
        help="Path to the evolutionary-tokenizer submodule.",
    )
    parser.add_argument(
        "--evolutionary-tokenizer-type",
        choices=["greedy", "pathpiece"],
        default="pathpiece",
        help="Evolutionary tokenizer backend.",
    )
    parser.add_argument("--evolutionary-generations", type=int, default=20, help="Evolution generations.")
    parser.add_argument("--evolutionary-population-size", type=int, default=120, help="Evolution population size.")
    parser.add_argument("--evolutionary-elite-size", type=int, default=20, help="Elite individuals per generation.")
    parser.add_argument("--evolutionary-mutation-rate", type=float, default=0.7, help="Mutation rate.")
    parser.add_argument("--evolutionary-crossover-rate", type=float, default=0.7, help="Crossover rate.")
    parser.add_argument("--evolutionary-mutation-count", type=int, default=10, help="Tokens mutated per mutation step.")
    parser.add_argument("--evolutionary-ngram-max-length", type=int, default=12, help="Maximum n-gram length.")
    parser.add_argument("--evolutionary-ngram-top-k", type=int, default=300_000, help="Top-k n-grams retained in candidate pool.")
    parser.add_argument(
        "--evolutionary-num-bpe-individuals",
        type=int,
        default=None,
        help="Optional fixed number of BPE-initialized individuals.",
    )
    parser.add_argument(
        "--evolutionary-num-random-ngram-individuals",
        type=int,
        default=0,
        help="Number of random n-gram-initialized individuals.",
    )
    parser.add_argument("--evolutionary-weight-compression", type=float, default=1.0, help="Fitness weight: compression.")
    parser.add_argument("--evolutionary-weight-unk-penalty", type=float, default=0.0, help="Fitness weight: unknown-token penalty.")
    parser.add_argument("--evolutionary-weight-token-usage", type=float, default=0.0, help="Fitness weight: token usage.")
    parser.add_argument(
        "--evolutionary-weight-cross-lingual-consistency",
        type=float,
        default=10.0,
        help="Fitness weight: cross-lingual consistency.",
    )
    parser.add_argument("--evolutionary-weight-diversity", type=float, default=5.0, help="Fitness weight: diversity.")
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
    elif tokenizer_type == "evolutionary":
        actual_vocab_size = train_evolutionary(
            corpus_path=corpus_path,
            output_dir=output_dir,
            vocab_size=args.vocab_size,
            max_lines=args.max_lines,
            max_bytes=args.max_bytes,
            submodule_dir=args.evolutionary_submodule_dir.resolve(),
            evo_tokenizer_type=args.evolutionary_tokenizer_type,
            evo_generations=args.evolutionary_generations,
            evo_population_size=args.evolutionary_population_size,
            evo_elite_size=args.evolutionary_elite_size,
            evo_mutation_rate=args.evolutionary_mutation_rate,
            evo_crossover_rate=args.evolutionary_crossover_rate,
            evo_mutation_count=args.evolutionary_mutation_count,
            evo_ngram_max_length=args.evolutionary_ngram_max_length,
            evo_ngram_top_k=args.evolutionary_ngram_top_k,
            evo_num_bpe_individuals=args.evolutionary_num_bpe_individuals,
            evo_num_random_ngram_individuals=args.evolutionary_num_random_ngram_individuals,
            evo_weight_compression=args.evolutionary_weight_compression,
            evo_weight_unk_penalty=args.evolutionary_weight_unk_penalty,
            evo_weight_token_usage=args.evolutionary_weight_token_usage,
            evo_weight_cross_lingual_consistency=args.evolutionary_weight_cross_lingual_consistency,
            evo_weight_diversity=args.evolutionary_weight_diversity,
        )
        add_evolutionary_metadata(
            output_dir,
            corpus_path=corpus_path,
            target_vocab_size=args.vocab_size,
            actual_vocab_size=actual_vocab_size,
            seed=args.seed,
            max_bytes=args.max_bytes,
            max_lines=args.max_lines,
            evolutionary_config={
                "submodule_dir": str(args.evolutionary_submodule_dir.resolve()),
                "tokenizer_type": args.evolutionary_tokenizer_type,
                "generations": args.evolutionary_generations,
                "population_size": args.evolutionary_population_size,
                "elite_size": args.evolutionary_elite_size,
                "mutation_rate": args.evolutionary_mutation_rate,
                "crossover_rate": args.evolutionary_crossover_rate,
                "mutation_count": args.evolutionary_mutation_count,
                "ngram_max_length": args.evolutionary_ngram_max_length,
                "ngram_top_k": args.evolutionary_ngram_top_k,
                "num_bpe_individuals": args.evolutionary_num_bpe_individuals,
                "num_random_ngram_individuals": args.evolutionary_num_random_ngram_individuals,
                "weight_compression": args.evolutionary_weight_compression,
                "weight_unk_penalty": args.evolutionary_weight_unk_penalty,
                "weight_token_usage": args.evolutionary_weight_token_usage,
                "weight_cross_lingual_consistency": args.evolutionary_weight_cross_lingual_consistency,
                "weight_diversity": args.evolutionary_weight_diversity,
            },
        )
        logger.info("Saved evolutionary tokenizer artifacts under: %s", output_dir)
        logger.info("Target vocab size: %d | Actual vocab size: %d", args.vocab_size, actual_vocab_size)
        return 0
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
