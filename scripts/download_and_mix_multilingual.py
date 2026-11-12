#!/usr/bin/env python3
import argparse
import json
import logging
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from datasets import get_dataset_config_names, load_dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Default Fineweb-2 language subset
DEFAULT_LANGS = ["eng_Latn", "spa_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hani"]

# Use uniform mixing ratios as a default
DEFAULT_RATIOS = {
    "eng_Latn": 0.2,
    "spa_Latn": 0.2,
    "deu_Latn": 0.2,
    "rus_Cyrl": 0.2,
    "cmn_Hani": 0.2
}


def get_available_subsets():
    """Query the live FineWeb-2 language metadata and add the English fallback."""
    try:
        subsets = sorted(get_dataset_config_names("HuggingFaceFW/fineweb-2"))
        subsets = [
            subset for subset in subsets
            if subset and not subset.endswith("_removed") and not subset.startswith("und_")
        ]
    except Exception as e:
        logger.warning("Failed to load FineWeb-2 metadata from Hugging Face: %s", e)
        subsets = []

    if "eng_Latn" not in subsets:
        subsets.append("eng_Latn")
    return sorted(set(subsets))


def resolve_language_dataset(lang):
    """Return the dataset name and optional config for a language.

    FineWeb-2 exposes language-specific configs; English is missing there and is instead
    sourced from the original FineWeb corpus.
    """
    if lang == "eng_Latn":
        return "HuggingFaceFW/fineweb", None
    return "HuggingFaceFW/fineweb-2", lang


def list_languages(available_subsets):
    logger.info("\n=== Available FineWeb-2 subsets ===")
    for subset in available_subsets:
        marker = " [DEFAULT IN MIX]" if subset in DEFAULT_LANGS else ""
        if subset == "eng_Latn":
            marker += " [FALLBACK: original FineWeb]"
        logger.info("  - %s%s", subset, marker)
    logger.info("\nTotal found: %d script/language configs.", len(available_subsets))
    logger.info("==============================================\n")


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
                logger.warning("Ratio for '%s' was provided but the language is not selected. Ignoring.", lang)
                continue
            ratios[lang] = float(weight)

        total_weight = sum(ratios.values())
        for lang in ratios:
            ratios[lang] /= total_weight

        for lang in selected_langs:
            if lang not in ratios:
                ratios[lang] = 0.0
        return ratios
    except Exception as e:
        logger.error("Error parsing ratios. Falling back to uniform weighting. Detail: %s", e)
        return {l: 1.0 / len(selected_langs) for l in selected_langs}


def parse_args():
    parser = argparse.ArgumentParser(description="Stream and mix FineWeb-2 corpora by byte budget, not token count")
    parser.add_argument(
        "--list-languages", action="store_true", help="List all available language subsets on Hugging Face and exit"
    )
    parser.add_argument(
        "--languages", type=str, help="Comma-separated FineWeb-2 subset names (e.g. eng_Latn,deu_Latn,ces_Latn)"
    )
    parser.add_argument(
        "--target-bytes", type=int, default=1_000_000, help="Target total byte budget for the final mixed corpus (may finish slightly under the cap)"
    )
    parser.add_argument("--ratios", type=str, help="Custom mixing ratios, e.g. deu_Latn:0.5,rus_Cyrl:0.5")
    parser.add_argument(
        "--output",
        type=str,
        default="multilingual_corpus.jsonl",
        help="Output file name (.jsonl or .tsv; default is JSONL)",
    )
    parser.add_argument(
        "--output-format",
        "--format",
        dest="output_format",
        choices=("jsonl", "tsv"),
        default="jsonl",
        help="Output record format: jsonl (default) or tsv",
    )
    parser.add_argument("--min-char-len", type=int, default=200, help="Minimum document length in characters")
    args = parser.parse_args()
    if args.target_bytes <= 0:
        parser.error("--target-bytes must be a positive integer")
    return args


TEMP_SHARD_SIZE_BYTES = 64 * 1024 * 1024
HF_RETRY_DELAYS = (1, 2, 4, 8, 16)


def load_dataset_with_retry(dataset_name, **load_kwargs):
    last_error = None
    for attempt, delay in enumerate(HF_RETRY_DELAYS + (None,), start=1):
        try:
            return load_dataset(dataset_name, **load_kwargs)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Dataset load failed for %s (attempt %d/%d): %s",
                dataset_name,
                attempt,
                len(HF_RETRY_DELAYS) + 1,
                exc,
            )
            if delay is None:
                break
            logger.info("Retrying dataset load for %s in %ss...", dataset_name, delay)
            time.sleep(delay)
    raise last_error


def format_output_record(lang, text, doc_bytes, output_format):
    if output_format == "jsonl":
        output_obj = {
            "text": text,
            "dataset_subset": lang,
            "byte_length": doc_bytes,
        }
        return json.dumps(output_obj, ensure_ascii=False) + "\n"

    safe_text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return f"{doc_bytes}\t{lang}\t{safe_text}\n"


def write_language_shards(lang, target_bytes, temp_dir, min_char_len, output_format):
    dataset_name, dataset_config = resolve_language_dataset(lang)
    load_kwargs = {"split": "train", "streaming": True}
    if dataset_config is not None:
        load_kwargs["name"] = dataset_config

    shard_paths = []
    current_shard_path = None
    current_handle = None
    current_bytes = 0
    total_written_bytes = 0

    logger.info("Opening network stream for: %s...", lang)
    try:
        ds = load_dataset_with_retry(dataset_name, **load_kwargs)
    except Exception as e:
        logger.error("Error while connecting to subset %s: %s", lang, e)
        raise

    for doc in ds:
        if total_written_bytes >= target_bytes:
            break

        text = doc.get("text", "")
        if len(text) < min_char_len:
            continue

        doc_bytes = len(text.encode("utf-8"))
        if doc_bytes <= 0:
            continue

        remaining_budget = target_bytes - total_written_bytes
        if doc_bytes > remaining_budget:
            logger.debug(
                "Skipping oversized document for %s (%s bytes > %s bytes remaining in quota).",
                lang,
                doc_bytes,
                remaining_budget,
            )
            continue

        if current_handle is None or current_bytes + doc_bytes > TEMP_SHARD_SIZE_BYTES:
            if current_handle is not None:
                current_handle.close()
            current_shard_path = temp_dir / f"{lang}_shard_{len(shard_paths):04d}.tmp"
            current_handle = current_shard_path.open("w", encoding="utf-8")
            current_bytes = 0
            shard_paths.append(current_shard_path)

        current_handle.write(format_output_record(lang, text, doc_bytes, output_format))
        current_bytes += doc_bytes
        total_written_bytes += doc_bytes

    if current_handle is not None:
        current_handle.close()

    if total_written_bytes < target_bytes:
        logger.warning(
            "Dataset for %s ended before reaching the requested target (%s bytes written, %s bytes target). Stopping for this language.",
            lang,
            f"{total_written_bytes:,}",
            f"{target_bytes:,}",
        )

    logger.info("  - %s: %s bytes written to %d shard(s)", lang, f"{total_written_bytes:,}", len(shard_paths))
    return shard_paths, total_written_bytes


def merge_shards(shard_paths, output_path):
    with open(output_path, "w", encoding="utf-8") as out_f:
        for shard_path in shard_paths:
            with open(shard_path, "r", encoding="utf-8") as shard_f:
                for line in shard_f:
                    out_f.write(line)


def main(args):
    if args.list_languages:
        available_subsets = get_available_subsets()
        list_languages(available_subsets)
        return 0

    if args.languages:
        selected_langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    else:
        logger.info("No languages were provided. Using the default mix: %s", ", ".join(DEFAULT_LANGS))
        selected_langs = list(DEFAULT_LANGS)

    available_subsets = get_available_subsets()
    invalid_langs = [l for l in selected_langs if l not in available_subsets]
    if invalid_langs:
        logger.error("Subset(s) %s do not exist in FineWeb-2. Run with --list-languages to see valid codes.", ", ".join(invalid_langs))
        return 1

    mixing_ratios = parse_mixing_ratios(args.ratios, selected_langs)

    logger.info("\nPlanned composition of the output corpus:")
    for lang in selected_langs:
        logger.info("  - %s: %.1f%%", lang, mixing_ratios[lang] * 100)
    logger.info("Target size: %s bytes", f"{args.target_bytes:,}")

    target_bytes_per_lang = {lang: int(args.target_bytes * mixing_ratios[lang]) for lang in selected_langs}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = None
    try:
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_path.stem}_", dir=str(output_path.parent)))
        except OSError:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"{output_path.stem}_"))

        logger.info("Temporary shard directory: %s", temp_dir)
        shard_paths = []
        collected_bytes = {lang: 0 for lang in selected_langs}

        logger.info("\nStarting sharded download and writing to %s...", str(output_path.resolve()))
        for lang in selected_langs:
            try:
                lang_shards, written_bytes = write_language_shards(
                    lang=lang,
                    target_bytes=target_bytes_per_lang[lang],
                    temp_dir=temp_dir,
                    min_char_len=args.min_char_len,
                    output_format=args.output_format,
                )
            except Exception as e:
                logger.error("Error while connecting to subset %s: %s", lang, e)
                return 1

            shard_paths.extend(lang_shards)
            collected_bytes[lang] = written_bytes

        if not shard_paths:
            logger.warning("No records were written for the selected languages; output file will be empty.")

        random.shuffle(shard_paths)
        merge_shards(shard_paths, str(output_path))
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("\n=== Done ===")
    logger.info("Saved to: %s", str(Path(args.output).resolve()))
    logger.info("Total UTF-8 bytes collected: %s / %s", f"{sum(collected_bytes.values()):,}", f"{args.target_bytes:,}")
    for lang in selected_langs:
        logger.info("  - %s: %s bytes (target: %s)", lang, f"{collected_bytes[lang]:,}", f"{target_bytes_per_lang[lang]:,}")


if __name__ == "__main__":
    sys.exit(main(parse_args()))
