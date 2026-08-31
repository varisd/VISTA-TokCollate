#!/usr/bin/env bash
# 130M monolingual sweep: tokenize, then train, one language x tokenizer across vocabulary sizes.
#
# Usage:
#   bash launch_monolingual.sh <dataset> <tokenizer_type> [vocab ...]
#
# Examples:
#   bash launch_monolingual.sh eng_latn bpe_unscaled                # full vocabulary sweep
#   bash launch_monolingual.sh tha_thai unigram_unscaled 32768      # a single cell
#
# Arguments:
#   dataset          eng_latn | tha_thai | urd_arab | amh_ethi | vie_latn
#   tokenizer_type   bpe_unscaled | unigram_unscaled
#   vocab ...        one or more vocabulary sizes; defaults to the full sweep
#
# Environment:
#   goldfish=True|False   use the Goldfish-matched recipe (default True)
#   MASTER_PORT           torchrun rendezvous port (default 29510)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/local.env" ]]; then
  echo "[ERROR] local.env not found. Copy local.env.example to local.env and set DATA_ROOT." >&2
  exit 1
fi
source "${SCRIPT_DIR}/local.env"

DATASET=${1:?usage: bash launch_monolingual.sh <dataset> <tokenizer_type> [vocab ...]}
TOKENIZER=${2:?usage: bash launch_monolingual.sh <dataset> <tokenizer_type> [vocab ...]}
shift 2

VOCABS=("$@")
if [[ ${#VOCABS[@]} -eq 0 ]]; then
  VOCABS=(8192 16384 32768 49152 65536 81920 98304 114688 262144)
fi

goldfish=${goldfish:-True}
TRAIN_SCRIPT="${SCRIPT_DIR}/monolingual_130m.sh"
if [[ "$goldfish" == "True" ]]; then
  TRAIN_SCRIPT="${SCRIPT_DIR}/monolingual_130m_gold.sh"
fi

MASTER_PORT=${MASTER_PORT:-29510}
MAX_SEQ_LEN=1024

echo "[Config] dataset=$DATASET tokenizer=$TOKENIZER vocabs=${VOCABS[*]}"
echo "[Config] goldfish=$goldfish, using training script: $TRAIN_SCRIPT"

tokenize_fn() {
  local dataset=$1
  local tokenizer_type=$2
  local vocab=$3
  python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
    --dataset "$dataset" --tokenizer_type "$tokenizer_type" --tokenizer_vocabulary "$vocab" \
    --split train --max_seq_len $MAX_SEQ_LEN --max_segments -1 --prepend_cls True --include_sep True --shuffle True
  python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
    --dataset "$dataset" --tokenizer_type "$tokenizer_type" --tokenizer_vocabulary "$vocab" \
    --split eval --max_seq_len $MAX_SEQ_LEN --max_segments -1 --prepend_cls True --include_sep True --shuffle False
}

for vocab in "${VOCABS[@]}"; do
  echo "================================================================"
  echo "[Tokenize] dataset=$DATASET, tok=$TOKENIZER, vocab=$vocab"
  tokenize_fn "$DATASET" "$TOKENIZER" "$vocab"

  echo "[Train] dataset=$DATASET, tok=$TOKENIZER, vocab=$vocab"
  bash "$TRAIN_SCRIPT" pre "$DATASET" "$vocab" "$TOKENIZER" 6 "$MASTER_PORT"
done
