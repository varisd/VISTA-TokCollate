#!/usr/bin/env python3
import argparse
import logging
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATASET_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
DEFAULT_LANGS = ["eng_Latn", "spa_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hani"]
SPLITS = ["dev", "devtest"]
LANG_ALIASES = {
    "cmn_Hani": "zho_Hans",  # FLORES-200 uses zho_Hans / zho_Hant instead of cmn_Hani
    "ekk_Latn": "est_Latn",  # FLORES-200 uses est_Latn (macro-language) instead of ekk_Latn
    "fas_Arab": "pes_Arab",  # FLORES-200 uses pes_Arab (Western Persian) instead of fas_Arab
}


def resolve_lang_code(lang: str) -> str:
    return LANG_ALIASES.get(lang, lang)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download FLORES-200 evaluation splits (dev and devtest)."
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
        help="Directory where downloaded .txt files will be saved.",
    )
    return parser.parse_args()


def main(args) -> int:
    raw_langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    selected_langs = [resolve_lang_code(l) for l in raw_langs]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "flores200_dataset.tar.gz"
        logger.info("Downloading FLORES-200 archive from %s...", DATASET_URL)
        try:
            urllib.request.urlretrieve(DATASET_URL, archive_path)
        except Exception as exc:
            logger.error("Failed to download archive from '%s': %s", DATASET_URL, exc)
            return 1

        with tarfile.open(archive_path, "r:gz") as tar:
            members = {m.name.lstrip("./"): m for m in tar.getmembers()}
            available_langs = sorted(
                Path(name).stem
                for name in members
                if name.startswith("flores200_dataset/dev/") and name.endswith(".dev")
            )

            invalid = [l for l in selected_langs if l not in available_langs]
            if invalid:
                logger.error(
                    "Invalid FLORES-200 language code(s): %s.",
                    ", ".join(invalid),
                )
                return 1

            for lang in selected_langs:
                logger.info("Extracting %s...", lang)
                for split in SPLITS:
                    member_name = f"flores200_dataset/{split}/{lang}.{split}"
                    if member_name not in members:
                        logger.warning("Split '%s' not found for %s. Skipping.", split, lang)
                        continue

                    raw_text = tar.extractfile(members[member_name]).read().decode("utf-8")
                    sentences = [line.replace("\n", " ").strip() for line in raw_text.splitlines()]

                    txt_path = out_dir / f"{lang}.{split}.txt"
                    txt_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")

                    logger.info("  Saved %d sentences -> %s", len(sentences), txt_path.name)

    logger.info("=== Done! All files saved to %s ===", out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
