#!/usr/bin/env bash
# Download the monolingual training corpora used by the 130M runs.
#
# Source: https://huggingface.co/datasets/goldfish-models/fish-food
# Files are named {language}_{script}.txt, which is exactly the layout the
# byte-premium and tokenization scripts expect.
#
# Usage:
#   bash download_monolingual_data.sh                 # all five paper languages
#   bash download_monolingual_data.sh eng_latn        # just one

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/local.env" ]]; then
  echo "[ERROR] local.env not found. Copy local.env.example to local.env and set DATA_ROOT." >&2
  exit 1
fi
source "${SCRIPT_DIR}/local.env"

LANGUAGES=("$@")
if [[ ${#LANGUAGES[@]} -eq 0 ]]; then
  LANGUAGES=(eng_latn tha_thai urd_arab amh_ethi vie_latn)
fi

PATTERNS=""
for lang in "${LANGUAGES[@]}"; do
  PATTERNS="${PATTERNS}'${lang}.txt',"
done

echo "[Download] ${LANGUAGES[*]} -> ${DATA_ROOT}/monolingual_training_data"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='goldfish-models/fish-food',
    repo_type='dataset',
    allow_patterns=[${PATTERNS}],
    local_dir='${DATA_ROOT}/monolingual_training_data'
)
print('Done!')
"
