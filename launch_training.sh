#!/usr/bin/env bash
# 350M FineWeb (English) sweep: tokenize -> split to 7B tokens -> train.
#
# Run this BEFORE launch_350m_fineweb2.sh. The tokenized English output is the
# byte-size reference that every other language is byte-premium matched against.
#
# Usage:
#   bash launch_350m_fineweb.sh <tokenizer_type> [vocab ...]
#
# Examples:
#   bash launch_350m_fineweb.sh bpe_unscaled                  # full vocabulary sweep
#   bash launch_350m_fineweb.sh unigram_unscaled 32768        # a single cell
#
# Environment:
#   NUM_GPUS      GPUs per run (default 4)
#   MASTER_PORT   torchrun rendezvous port (default 29510)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/local.env" ]]; then
  echo "[ERROR] local.env not found. Copy local.env.example to local.env and set DATA_ROOT." >&2
  exit 1
fi
source "${SCRIPT_DIR}/local.env"

TOKENIZER=${1:?usage: bash launch_350m_fineweb.sh <tokenizer_type> [vocab ...]}
shift 1

VOCABS=("$@")
if [[ ${#VOCABS[@]} -eq 0 ]]; then
  VOCABS=(8192 32768 65536 98304)
fi

TRAIN_SCRIPT="${SCRIPT_DIR}/monolingual_350m.sh"
DATASET="fineweb_eng"
BP="1.0"

MAX_SEQ_LEN=1024
TARGET_TOKENS=7000000000   # 7B tokens
EVAL_LINES=8000
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"
MASTER_PORT=${MASTER_PORT:-29510}

echo "[Config] dataset=$DATASET tokenizer=$TOKENIZER vocabs=${VOCABS[*]}"
echo "[Config] Using training script: $TRAIN_SCRIPT"

raw_file="${DATA_ROOT}/monolingual_training_data/${DATASET}.txt"
if [[ ! -f "$raw_file" ]]; then
  echo "[ERROR] Raw text file not found: $raw_file" >&2
  echo "Run: python scripts/download_fineweb.py" >&2
  exit 1
fi

tokenize_and_split_fn() {
  local tokenizer_type=$1
  local vocab=$2

  # fineweb_eng uses the eng_latn tokenizers
  local prefix
  if [[ "$tokenizer_type" == "bpe_unscaled" ]]; then
    prefix="bpe"
  else
    prefix="unigram"
  fi
  local tok_basename="${prefix}_eng_latn_${vocab}_300mb_unscaled"
  local tok_dir="${OUTPUT_ROOT}/${tok_basename}"
  local tok_file="${tok_dir}/${DATASET}_${BP}_tokenized.txt"
  local full_file="${tok_dir}/${DATASET}_${BP}_tokenized_full.txt"
  local eval_file="${tok_dir}/${DATASET}_${BP}_eval_tokenized.txt"

  # Skip if the final train + eval splits already exist
  if [[ -f "$tok_file" && -f "$eval_file" ]]; then
    echo "  [Skip] Tokenized splits already exist: $tok_file"
  else
    # Step 1: Tokenize ALL raw text if not already done
    if [[ -f "$full_file" ]]; then
      echo "  [Step 1] Full tokenized file already exists: $full_file"
    else
      echo "  [Step 1] Tokenizing full dataset: dataset=$DATASET, tok=$tokenizer_type, vocab=$vocab"
      python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
        --dataset "$DATASET" --tokenizer_type "$tokenizer_type" --tokenizer_vocabulary "$vocab" \
        --split train --max_seq_len $MAX_SEQ_LEN --max_segments -1 \
        --prepend_cls True --include_sep True --shuffle False \
        --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"
      # Rename to _full so tokenize_and_pack won't treat it as already done next run
      mv "$tok_file" "$full_file"
    fi

    # Step 2: Split by target token count
    echo "  [Step 2] Splitting: first ${TARGET_TOKENS} tokens for train, last ${EVAL_LINES} lines for eval"
    python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
      --input "$full_file" \
      --train_output "$tok_file" \
      --eval_output "$eval_file" \
      --target_tokens $TARGET_TOKENS \
      --eval_lines $EVAL_LINES
  fi
}

for vocab in "${VOCABS[@]}"; do
  echo "================================================================"
  echo "[Tokenize+Split] dataset=$DATASET, tok=$TOKENIZER, vocab=$vocab"
  tokenize_and_split_fn "$TOKENIZER" "$vocab"

  echo "[Train] dataset=$DATASET, tok=$TOKENIZER, vocab=$vocab"
  bash "$TRAIN_SCRIPT" pre "$DATASET" "$vocab" "$TOKENIZER" 6 "$MASTER_PORT"
done
