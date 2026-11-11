#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import shutil
import sys
import tempfile
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


def fetch_available_subsets():
    """Fetch all available FineWeb-2 subsets and add the English FineWeb fallback used for eng_Latn."""
    try:
        subsets = sorted(get_dataset_config_names("HuggingFaceFW/fineweb-2"))
        subsets = [
            subset for subset in subsets
            if subset and not subset.endswith("_removed") and not subset.startswith("und_")
        ]
    except Exception as e:
        logger.warning("Failed to load FineWeb-2 metadata from Hugging Face: %s", e)
        subsets = []

    # English is not exposed as eng_Latn in FineWeb-2, but the original FineWeb corpus is the right fallback.
    if "eng_Latn" not in subsets:
        subsets.append("eng_Latn")
    return sorted(subsets)


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
    parser.add_argument(
        "--shuffle", action="store_true", help="Shuffle the output corpus after mixing"
    )
    parser.add_argument(
        "--shuffle-batch-size",
        type=int,
        default=10_000,
        help="Rows to keep in memory per shuffle batch; smaller values use less RAM",
    )
    parser.add_argument(
        "--random-seed", default=42, type=int, help="Random seed for shuffling (if --shuffle is used)"
    )

    args = parser.parse_args()
    if args.target_bytes <= 0:
        parser.error("--target-bytes must be a positive integer")
    if args.shuffle_batch_size <= 0:
        parser.error("--shuffle-batch-size must be a positive integer")
    return args


def shuffle_output_streaming(output_path, random_seed, batch_size):
    """Shuffle a large output file without holding the entire file in memory.

    The algorithm batches rows into fixed-size chunks, shuffles each chunk in memory,
    writes the chunk to a temporary file, and then concatenates the temporary files in
    a random order. This keeps memory bounded while still yielding a non-trivial shuffle.
    """
    if not os.path.exists(output_path):
        return

    tmp_dir = tempfile.mkdtemp(prefix="multilingual_shuffle_")
    chunk_paths = []
    batch = []
    try:
        with open(output_path, "r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                batch.append(line)
                if len(batch) >= batch_size:
                    chunk_path = os.path.join(tmp_dir, f"chunk_{len(chunk_paths):06d}.txt")
                    rng = random.Random(random_seed + len(chunk_paths))
                    rng.shuffle(batch)
                    with open(chunk_path, "w", encoding="utf-8") as chunk_file:
                        chunk_file.writelines(batch)
                    chunk_paths.append(chunk_path)
                    batch = []

        if batch:
            chunk_path = os.path.join(tmp_dir, f"chunk_{len(chunk_paths):06d}.txt")
            rng = random.Random(random_seed + len(chunk_paths))
            rng.shuffle(batch)
            with open(chunk_path, "w", encoding="utf-8") as chunk_file:
                chunk_file.writelines(batch)
            chunk_paths.append(chunk_path)

        if not chunk_paths:
            return

        rng = random.Random(random_seed)
        rng.shuffle(chunk_paths)

        with open(output_path, "w", encoding="utf-8") as dst:
            for chunk_path in chunk_paths:
                with open(chunk_path, "r", encoding="utf-8") as chunk_file:
                    shutil.copyfileobj(chunk_file, dst)
    finally:
        for chunk_path in chunk_paths:
            try:
                os.remove(chunk_path)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def main(args):
    available_subsets = fetch_available_subsets()

    if args.list_languages:
        list_languages(available_subsets)
        return 0

    if args.languages:
        selected_langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    else:
        logger.info("No languages were provided. Using the default mix: %s", ", ".join(DEFAULT_LANGS))
        selected_langs = [l for l in DEFAULT_LANGS if l in available_subsets]

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

    iterators = {}
    for lang in selected_langs:
        logger.info("Opening network stream for: %s...", lang)
        try:
            dataset_name, dataset_config = resolve_language_dataset(lang)
            load_kwargs = {"split": "train", "streaming": True}
            if dataset_config is not None:
                load_kwargs["name"] = dataset_config
            ds = load_dataset(dataset_name, **load_kwargs)
            iterators[lang] = iter(ds)
        except Exception as e:
            logger.error("Error while connecting to subset %s: %s", lang, e)
            return 1

    collected_bytes = {lang: 0 for lang in selected_langs}
    total_written_bytes = 0
    active_langs = list(selected_langs)
    reached_byte_cap = False

    logger.info("\nStarting sample download and writing to %s...", args.output)

    with open(args.output, "w", encoding="utf-8") as f_out:
        while active_langs and not reached_byte_cap:
            for lang in list(active_langs):
                if collected_bytes[lang] >= target_bytes_per_lang[lang]:
                    active_langs.remove(lang)
                    continue

                try:
                    doc = next(iterators[lang])
                    text = doc.get("text", "")

                    if len(text) < args.min_char_len:
                        continue

                    doc_bytes = len(text.encode("utf-8"))
                    if total_written_bytes + doc_bytes > args.target_bytes:
                        logger.info("\nReached byte cap: stopping at %s bytes (target: %s).",
                                    f"{total_written_bytes:,}", f"{args.target_bytes:,}")
                        reached_byte_cap = True
                        break

                    if args.output_format == "jsonl":
                        output_obj = {
                            "text": text,
                            "dataset_subset": lang,
                            "byte_length": doc_bytes
                        }
                        f_out.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
                    else:
                        safe_text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                        f_out.write(f"{doc_bytes}\t{lang}\t{safe_text}\n")

                    collected_bytes[lang] += doc_bytes
                    total_written_bytes += doc_bytes

                except StopIteration:
                    logger.warning("\nData for subset %s were exhausted before reaching the target quota.", lang)
                    active_langs.remove(lang)

    if args.shuffle and os.path.exists(args.output):
        shuffle_output_streaming(args.output, args.random_seed, args.shuffle_batch_size)

    logger.info("\n=== Done ===")
    logger.info("Saved to: %s", os.path.abspath(args.output))
    logger.info("Total UTF-8 bytes collected: %s / %s", f"{total_written_bytes:,}", f"{args.target_bytes:,}")
    for lang in selected_langs:
        logger.info("  - %s: %s bytes (target: %s)", lang, f"{collected_bytes[lang]:,}", f"{target_bytes_per_lang[lang]:,}")


if __name__ == "__main__":
    sys.exit(main(parse_args()))
