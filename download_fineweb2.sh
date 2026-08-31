#!/usr/bin/env bash
# Download FineWeb-2 raw text for Amharic, Thai, Urdu and Vietnamese.
#
# Amharic and Urdu are small enough to take whole. Thai and Vietnamese are huge,
# so they are capped at 1.02 x byte_premium x (English reference bytes), just
# above what the byte-premium split will actually consume.
#
# Requires the English FineWeb pipeline to have been tokenized first, since the
# cap is computed from the English reference file:
#   bash launch_350m_fineweb.sh bpe_unscaled 98304
#
# Usage:
#   bash download_fineweb2.sh                 # all four languages
#   bash download_fineweb2.sh amh urd         # a subset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/local.env" ]]; then
  echo "[ERROR] local.env not found. Copy local.env.example to local.env and set DATA_ROOT." >&2
  exit 1
fi
source "${SCRIPT_DIR}/local.env"

LANGUAGES=("$@")
if [[ ${#LANGUAGES[@]} -eq 0 ]]; then
  LANGUAGES=(amh urd tha vie)
fi

# Byte premiums for the capped languages
declare -A BYTE_PREMIUM_MAP=( [tha]=2.74 [vie]=1.35 )

ENG_REF="${DATA_ROOT}/monolingual_training_data_tokenized/bpe_eng_latn_98304_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"

for lang in "${LANGUAGES[@]}"; do
  echo "================================================================"
  bp="${BYTE_PREMIUM_MAP[$lang]:-}"

  if [[ -z "$bp" ]]; then
    # Amharic / Urdu: small enough to download in full
    echo "[Download] $lang (all available)"
    python3 "${SCRIPT_DIR}/scripts/download_fineweb2.py" --language "$lang"
  else
    # Thai / Vietnamese: capped
    if [[ ! -f "$ENG_REF" ]]; then
      echo "[ERROR] English reference file not found: $ENG_REF" >&2
      echo "Run: bash launch_350m_fineweb.sh bpe_unscaled 98304" >&2
      exit 1
    fi
    eng_bytes=$(stat -c%s "$ENG_REF")
    max_bytes=$(python3 -c "import math; print(math.floor(1.02 * $bp * $eng_bytes))")
    echo "[Download] $lang (capped at $max_bytes bytes = 1.02 x $bp x $eng_bytes)"
    python3 "${SCRIPT_DIR}/scripts/download_fineweb2.py" --language "$lang" --max_bytes "$max_bytes"
  fi
done

echo "Done. Raw text written to ${DATA_ROOT}/monolingual_training_data/fineweb2_{lang}.txt"
