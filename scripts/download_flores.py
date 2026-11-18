#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_LANGS = ["eng_Latn", "spa_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hani"]
SPLITS = ["dev", "devtest"]
REPO = "facebook/flores"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download FLORES-200 splits for specified languages"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_LANGS),
        help="Comma-separated FLORES language codes (e.g. eng_Latn,deu_Latn,ces_Latn)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="flores_data",
        help="Directory to save downloaded JSONL/TXT files",
    )
    return parser.parse_args()


def main(args):
    try:
        available_configs = get_dataset_config_names(REPO)
    except Exception as exc:
        logger.error(
            "Could not fetch configs from '%s': %s\n"
            "If using 'openlanguagedata/flores_plus', make sure you accepted the terms on Hugging Face and ran `huggingface-cli login`.",
            REPO,
            exc,
        )
        return 1

    if args.list_languages:
        logger.info("=== Available configs in %s (%d total) ===", REPO, len(available_configs))
        for cfg in sorted(available_configs):
            logger.info("  - %s", cfg)
        return 0

    selected_langs = [l.strip() for l in args.languages.split(",") if l.strip()]

    invalid = [l for l in selected_langs if l not in available_configs]
    if invalid:
        logger.error(
            "Invalid language code(s) for %s: %s. Run with --list-languages to see valid options.",
            REPO,
            ", ".join(invalid),
        )
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in selected_langs:
        logger.info("Downloading %s from %s...", lang, REPO)
        ds = load_dataset(REPO, lang)

        for split in SPLITS:
            if split not in ds:
                logger.warning("Split '%s' not found for %s (available: %s)", split, lang, list(ds.keys()))
                continue

            split_data = ds[split]
            text_col = "sentence" if "sentence" in split_data.column_names else "text"

            txt_path = out_dir / f"{lang}.{split}.txt"
            with open(txt_path, "w", encoding="utf-8") as tf:
                for row in split_data:
                    clean_line = row[text_col].replace("\n", " ").strip()
                    tf.write(clean_line + "\n")

            logger.info("  Saved %s (%s): %d sentences", lang, split, len(split_data))

    logger.info("=== Done! Files saved in %s ===", str(out_dir.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
