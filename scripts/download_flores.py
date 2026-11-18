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

# Ungated mirror of FLORES-200 in native Parquet format (no HF login or script execution needed)
DATASET_REPO = "Muennighoff/flores200"
DEFAULT_LANGS = ["eng_Latn", "spa_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hani"]
SPLITS = ["dev", "devtest"]
LANG_ALIASES = {
    "cmn_Hani": "zho_Hans",  # FLORES-200 uses zho_Hans / zho_Hant instead of cmn_Hani
}


def resolve_lang_code(lang: str) -> str:
    return LANG_ALIASES.get(lang, lang)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download FLORES-200 evaluation splits (dev and devtest) from Hugging Face."
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_LANGS),
        help="Comma-separated FLORES-200 language codes (e.g. eng_Latn,deu_Latn,ces_Latn).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="flores200",
        help="Directory where downloaded .txt and .jsonl files will be saved.",
    )
    return parser.parse_args()


def get_monolingual_configs() -> list[str]:
    configs = get_dataset_config_names(DATASET_REPO)
    # Keep only single-language configs (exclude 'all' and hyphenated pairs like 'eng_Latn-ces_Latn')
    return sorted(c for c in configs if c != "all" and "-" not in c)


def main(args) -> int:
    try:
        available_langs = get_monolingual_configs()
    except Exception as exc:
        logger.error("Failed to fetch language list from '%s': %s", DATASET_REPO, exc)
        return 1

    raw_langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    selected_langs = [resolve_lang_code(l) for l in raw_langs]

    invalid = [l for l in selected_langs if l not in available_langs]
    if invalid:
        logger.error(
            "Invalid FLORES-200 language code(s): %s.",
            ", ".join(invalid),
        )
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in selected_langs:
        logger.info("Downloading %s from %s...", lang, DATASET_REPO)
        ds = load_dataset(DATASET_REPO, lang)

        for split in SPLITS:
            if split not in ds:
                logger.warning("Split '%s' not found for %s. Skipping.", split, lang)
                continue

            txt_path = out_dir / f"{lang}.{split}.txt"
            sentences = [s.replace("\n", " ").strip() for s in ds[split]["sentence"]]
            txt_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")

            logger.info("  Saved %d sentences -> %s", len(split_data), txt_path.name)

    logger.info("=== Done! All files saved to %s ===", out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
