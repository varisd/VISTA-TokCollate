#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
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
    """Fetch all available language subsets directly from the FineWeb-2 dataset configs."""
    try:
        subsets = sorted(get_dataset_config_names("HuggingFaceFW/fineweb-2"))
        subsets = [
            subset for subset in subsets
            if subset and not subset.endswith("_removed") and not subset.startswith("und_")
        ]
        return subsets
    except Exception as e:
        logger.warning("Failed to load FineWeb-2 metadata from Hugging Face: %s", e)
        return DEFAULT_LANGS


def list_languages(available_subsets):
    logger.info("\n=== Available FineWeb-2 subsets ===")
    for subset in available_subsets:
        marker = " [DEFAULT IN MIX]" if subset in DEFAULT_LANGS else ""
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


def main():
    parser = argparse.ArgumentParser(description="Stream and mix FineWeb-2 corpora by byte budget, not token count")
    parser.add_argument("--list", action="store_true", help="List all available language subsets on Hugging Face and exit")
    parser.add_argument("--langs", type=str, help="Comma-separated FineWeb-2 subset names (e.g. eng_Latn,deu_Latn,ces_Latn)")
    parser.add_argument("--target_bytes", type=int, default=256_000_000, help="Target total byte budget for the final mixed corpus (may finish slightly under the cap)")
    parser.add_argument("--ratios", type=str, help="Custom mixing ratios, e.g. deu_Latn:0.5,rus_Cyrl:0.5")
    parser.add_argument("--output", type=str, default="multilingual_corpus.jsonl", help="Output file name (.jsonl)")
    parser.add_argument("--min_char_len", type=int, default=200, help="Minimum document length in characters")

    args = parser.parse_args()
    if args.target_bytes <= 0:
        parser.error("--target_bytes must be a positive integer")

    # Load the real list from Hugging Face.
    available_subsets = fetch_available_subsets()

    if args.list:
        list_languages(available_subsets)
        sys.exit(0)

    # 1. Validate and select languages.
    if args.langs:
        selected_langs = [l.strip() for l in args.langs.split(",")]
        for l in selected_langs:
            if l not in available_subsets:
                logger.error("Subset '%s' does not exist in FineWeb-2. Run with --list to see valid codes.", l)
                sys.exit(1)
    else:
        logger.info("No languages were provided. Using the default mix: %s", ", ".join(DEFAULT_LANGS))
        selected_langs = [l for l in DEFAULT_LANGS if l in available_subsets]

    # 2. Set the weights.
    mixing_ratios = parse_mixing_ratios(args.ratios, selected_langs)

    logger.info("\nPlanned composition of the output corpus:")
    for lang in selected_langs:
        logger.info("  - %s: %.1f%%", lang, mixing_ratios[lang] * 100)
    logger.info("Target size: %s bytes", f"{args.target_bytes:,}")

    target_bytes_per_lang = {lang: int(args.target_bytes * mixing_ratios[lang]) for lang in selected_langs}

    # 3. Initialize the streaming dataset iterators from Hugging Face.
    iterators = {}
    for lang in selected_langs:
        logger.info("Opening network stream for: %s...", lang)
        try:
            # stream=True ensures documents are pulled incrementally instead of downloading whole corpora to disk.
            ds = load_dataset("HuggingFaceFW/fineweb-2", name=lang, split="train", streaming=True)
            iterators[lang] = iter(ds)
        except Exception as e:
            logger.error("Error while connecting to subset %s: %s", lang, e)
            sys.exit(1)

    # 4. Proportional mixing and JSONL output by UTF-8 byte length.
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

                    output_obj = {
                        "text": text,
                        "dataset_subset": lang,
                        "byte_length": doc_bytes
                    }
                    f_out.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

                    collected_bytes[lang] += doc_bytes
                    total_written_bytes += doc_bytes

                except StopIteration:
                    logger.warning("\nData for subset %s were exhausted before reaching the target quota.", lang)
                    active_langs.remove(lang)

    logger.info("\n=== Done ===")
    logger.info("Saved to: %s", os.path.abspath(args.output))
    logger.info("Total UTF-8 bytes collected: %s / %s", f"{total_written_bytes:,}", f"{args.target_bytes:,}")
    for lang in selected_langs:
        logger.info("  - %s: %s bytes (target: %s)", lang, f"{collected_bytes[lang]:,}", f"{target_bytes_per_lang[lang]:,}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # `head` closes the pipe early. Exit quietly rather than crashing.
        pass
