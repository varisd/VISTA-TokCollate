#!/usr/bin/env python3
import argparse
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

DEFAULT_LANGS = ["eng_Latn", "spa_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hani"]

DEFAULT_RATIOS = {
    "eng_Latn": 0.2,
    "spa_Latn": 0.2,
    "deu_Latn": 0.2,
    "rus_Cyrl": 0.2,
    "cmn_Hani": 0.2,
}

TEMP_SHARD_SIZE_BYTES = 64 * 1024 * 1024
COPY_BUFFER_BYTES = 16 * 1024 * 1024
PARQUET_BATCH_SIZE = 1024
HF_RETRY_DELAYS = (2, 5, 10, 20, 40)


def get_available_subsets(fs: HfFileSystem):
    """Discover available FineWeb-2 language directories directly via HfFileSystem."""
    try:
        entries = fs.ls("datasets/HuggingFaceFW/fineweb-2/data", detail=False)
        subsets = [
            Path(entry).name
            for entry in entries
            if not Path(entry).name.endswith("_removed")
            and not Path(entry).name.startswith("und_")
        ]
    except Exception as e:
        logger.warning("Failed to list FineWeb-2 subsets from Hugging Face: %s", e)
        subsets = []

    if "eng_Latn" not in subsets:
        subsets.append("eng_Latn")
    return sorted(set(subsets))


def get_parquet_paths(fs: HfFileSystem, lang: str):
    """Return sorted list of train Parquet file paths on Hugging Face for a language."""
    if lang == "eng_Latn":
        pattern = "datasets/HuggingFaceFW/fineweb/data/**/*.parquet"
    else:
        pattern = f"datasets/HuggingFaceFW/fineweb-2/data/{lang}/train/*.parquet"

    for attempt, delay in enumerate(HF_RETRY_DELAYS + (None,), start=1):
        try:
            files = sorted(fs.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No parquet files found matching {pattern}")
            return files
        except Exception as exc:
            if delay is None:
                raise
            logger.warning(
                "Failed to list parquet files for %s (attempt %d): %s. Retrying in %ss...",
                lang,
                attempt,
                exc,
                delay,
            )
            time.sleep(delay)


def list_languages(available_subsets):
    logger.info("=== Available FineWeb-2 subsets ===")
    for subset in available_subsets:
        marker = " [DEFAULT IN MIX]" if subset in DEFAULT_LANGS else ""
        if subset == "eng_Latn":
            marker += " [FALLBACK: original FineWeb]"
        logger.info("  - %s%s", subset, marker)
    logger.info("Total found: %d script/language configs.", len(available_subsets))


def parse_mixing_ratios(ratio_str, selected_langs):
    """Parse custom mixing ratios in the form lang:weight,lang:weight."""
    if not ratio_str:
        chosen_defaults = {l: DEFAULT_RATIOS.get(l, 1.0) for l in selected_langs}
        total = sum(chosen_defaults.values())
        return {l: chosen_defaults[l] / total for l in selected_langs}

    try:
        pairs = ratio_str.split(",")
        ratios = {}
        for pair in pairs:
            lang, weight = pair.split(":")
            if lang not in selected_langs:
                logger.warning(
                    "Ratio for '%s' provided but language not selected. Ignoring.", lang
                )
                continue
            ratios[lang] = float(weight)

        total_weight = sum(ratios.values())
        if total_weight <= 0:
            raise ValueError("Sum of weights must be positive.")

        for lang in ratios:
            ratios[lang] /= total_weight

        for lang in selected_langs:
            ratios.setdefault(lang, 0.0)
        return ratios
    except Exception as e:
        logger.error("Error parsing ratios (%s). Falling back to uniform weighting.", e)
        return {l: 1.0 / len(selected_langs) for l in selected_langs}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream and mix FineWeb-2 corpora by byte budget with low memory overhead"
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List all available language subsets on Hugging Face and exit",
    )
    parser.add_argument(
        "--languages",
        type=str,
        help="Comma-separated FineWeb-2 subset names (e.g. eng_Latn,deu_Latn,ces_Latn)",
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=1_000_000,
        help="Target total byte budget for the final mixed corpus",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        help="Custom mixing ratios, e.g. deu_Latn:0.5,rus_Cyrl:0.5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="multilingual_corpus.tsv",
        help="Output TSV file path",
    )
    parser.add_argument(
        "--min-char-len",
        type=int,
        default=200,
        help="Minimum document length in characters",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Directory for temporary shards (defaults to $TMPDIR or output parent directory)",
    )
    args = parser.parse_args()
    if args.target_bytes <= 0:
        parser.error("--target-bytes must be a positive integer")
    return args


def format_output_record(lang, text, doc_bytes):
    safe_text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return f"{doc_bytes}\t{lang}\t{safe_text}\n".encode("utf-8")


def iter_parquet_texts_resilient(fs: HfFileSystem, remote_path: str):
    """Stream 'text' column from a remote Parquet file in small batches with mid-stream retry."""
    rows_yielded = 0
    while True:
        try:
            with fs.open(remote_path, "rb") as f:
                pf = pq.ParquetFile(f)
                current_row = 0
                for batch in pf.iter_batches(
                    batch_size=PARQUET_BATCH_SIZE, columns=["text"]
                ):
                    batch_len = len(batch)
                    if current_row + batch_len <= rows_yielded:
                        current_row += batch_len
                        continue

                    texts = batch.column("text").to_pylist()
                    start_offset = max(0, rows_yielded - current_row)
                    for text in texts[start_offset:]:
                        rows_yielded += 1
                        yield text
                    current_row += batch_len
            return
        except Exception as exc:
            logger.warning(
                "Stream interrupted on %s at row %d (%s). Reconnecting...",
                remote_path,
                rows_yielded,
                exc,
            )
            for attempt, delay in enumerate(HF_RETRY_DELAYS, start=1):
                time.sleep(delay)
                try:
                    with fs.open(remote_path, "rb") as test_f:
                        test_f.read(1)
                    break
                except Exception:
                    if attempt == len(HF_RETRY_DELAYS):
                        raise


def write_language_shards(
    fs: HfFileSystem,
    lang: str,
    target_bytes: int,
    temp_dir: Path,
    min_char_len: int,
):
    if target_bytes <= 0:
        return [], 0

    parquet_files = get_parquet_paths(fs, lang)
    shard_paths = []
    current_handle = None
    current_bytes = 0
    total_written_bytes = 0
    stop_threshold_bytes = max(min_char_len * 4, 4096)

    logger.info(
        "Streaming %s (%d parquet files available, target: %s bytes)...",
        lang,
        len(parquet_files),
        f"{target_bytes:,}",
    )

    for remote_path in parquet_files:
        remaining_budget = target_bytes - total_written_bytes
        if remaining_budget <= stop_threshold_bytes:
            break

        for text in iter_parquet_texts_resilient(fs, remote_path):
            remaining_budget = target_bytes - total_written_bytes
            if remaining_budget <= stop_threshold_bytes:
                break

            if not text or len(text) < min_char_len:
                continue

            doc_bytes = len(text.encode("utf-8"))
            if doc_bytes <= 0 or doc_bytes > remaining_budget:
                continue

            if current_handle is None or current_bytes + doc_bytes > TEMP_SHARD_SIZE_BYTES:
                if current_handle is not None:
                    current_handle.close()
                shard_path = temp_dir / f"{lang}_shard_{len(shard_paths):05d}.tmp"
                current_handle = shard_path.open("wb")
                current_bytes = 0
                shard_paths.append(shard_path)

            current_handle.write(format_output_record(lang, text, doc_bytes))
            current_bytes += doc_bytes
            total_written_bytes += doc_bytes

    if current_handle is not None:
        current_handle.close()

    if total_written_bytes < target_bytes - stop_threshold_bytes:
        logger.warning(
            "Dataset for %s ended early (%s / %s bytes written).",
            lang,
            f"{total_written_bytes:,}",
            f"{target_bytes:,}",
        )

    logger.info(
        "Finished %s: %s bytes across %d shard(s).",
        lang,
        f"{total_written_bytes:,}",
        len(shard_paths),
    )
    return shard_paths, total_written_bytes


def merge_shards(shard_paths, output_path):
    """Merge temporary shards using raw binary chunk copying (zero string decoding)."""
    logger.info("Merging %d shuffled shards into %s...", len(shard_paths), output_path)
    with open(output_path, "wb") as out_f:
        for shard_path in shard_paths:
            with open(shard_path, "rb") as shard_f:
                shutil.copyfileobj(shard_f, out_f, length=COPY_BUFFER_BYTES)


def resolve_temp_base(cli_temp_dir, output_path: Path):
    """Choose the best scratch directory (explicit flag -> HPC $TMPDIR -> output parent)."""
    if cli_temp_dir:
        base = Path(cli_temp_dir)
        base.mkdir(parents=True, exist_ok=True)
        return base

    hpc_tmpdir = os.environ.get("TMPDIR")
    if hpc_tmpdir and Path(hpc_tmpdir).is_dir():
        return Path(hpc_tmpdir)

    return output_path.parent


def main(args):
    fs = HfFileSystem()

    if args.list_languages:
        available_subsets = get_available_subsets(fs)
        list_languages(available_subsets)
        return 0

    if args.languages:
        selected_langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    else:
        logger.info("No languages provided. Using default mix: %s", ", ".join(DEFAULT_LANGS))
        selected_langs = list(DEFAULT_LANGS)

    available_subsets = get_available_subsets(fs)
    invalid_langs = [l for l in selected_langs if l not in available_subsets]
    if invalid_langs:
        logger.error(
            "Subset(s) %s do not exist in FineWeb-2. Run with --list-languages to see valid codes.",
            ", ".join(invalid_langs),
        )
        return 1

    mixing_ratios = parse_mixing_ratios(args.ratios, selected_langs)

    logger.info("Planned composition of the output corpus:")
    for lang in selected_langs:
        logger.info("  - %s: %.1f%%", lang, mixing_ratios[lang] * 100)
    logger.info("Target size: %s bytes", f"{args.target_bytes:,}")

    target_bytes_per_lang = {
        lang: int(args.target_bytes * mixing_ratios[lang]) for lang in selected_langs
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_base = resolve_temp_base(args.temp_dir, output_path)

    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_path.stem}_", dir=str(temp_base)))
        logger.info("Temporary shard directory: %s", temp_dir)

        shard_paths = []
        collected_bytes = {lang: 0 for lang in selected_langs}

        for lang in selected_langs:
            lang_shards, written_bytes = write_language_shards(
                fs=fs,
                lang=lang,
                target_bytes=target_bytes_per_lang[lang],
                temp_dir=temp_dir,
                min_char_len=args.min_char_len,
            )
            shard_paths.extend(lang_shards)
            collected_bytes[lang] = written_bytes

        if not shard_paths:
            logger.warning("No records were written; output file will be empty.")

        random.shuffle(shard_paths)
        merge_shards(shard_paths, output_path)

        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Deleted temporary shard directory: %s", temp_dir)
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("=== Done ===")
    logger.info("Saved to: %s", str(output_path.resolve()))
    logger.info(
        "Total UTF-8 bytes collected: %s / %s",
        f"{sum(collected_bytes.values()):,}",
        f"{args.target_bytes:,}",
    )
    for lang in selected_langs:
        logger.info(
            "  - %s: %s bytes (target: %s)",
            lang,
            f"{collected_bytes[lang]:,}",
            f"{target_bytes_per_lang[lang]:,}",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
