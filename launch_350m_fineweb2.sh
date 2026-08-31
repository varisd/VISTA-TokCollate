#!/usr/bin/env bash
# 350M FineWeb-2 sweep: tokenize -> byte-premium split -> train, for the non-English languages.
#
# Requires launch_350m_fineweb.sh to have been run first at the same vocabulary sizes:
# the tokenized English output is the byte-size reference each language is matched against.
#
# Usage:
#   bash launch_350m_fineweb2.sh <dataset|all> <tokenizer_type> [vocab ...]
#
# Examples:
#   bash launch_350m_fineweb2.sh all bpe_unscaled                    # all four languages, full sweep
#   bash launch_350m_fineweb2.sh fineweb2_tha unigram_unscaled 32768 # a single cell
#
# Arguments:
#   dataset          fineweb2_amh | fineweb2_tha | fineweb2_urd | fineweb2_vie | all
#   tokenizer_type   bpe_unscaled | unigram_unscaled
#   vocab ...        one or more vocabulary sizes; defaults to the full sweep
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

DATASET_ARG=${1:?usage: bash launch_350m_fineweb2.sh <dataset|all> <tokenizer_type> [vocab ...]}
TOKENIZER=${2:?usage: bash launch_350m_fineweb2.sh <dataset|all> <tokenizer_type> [vocab ...]}
shift 2

if [[ "$DATASET_ARG" == "all" ]]; then
  DATASETS=(fineweb2_amh fineweb2_tha fineweb2_urd fineweb2_vie)
else
  DATASETS=("$DATASET_ARG")
fi

VOCABS=("$@")
if [[ ${#VOCABS[@]} -eq 0 ]]; then
  VOCABS=(8192 32768 65536 98304)
fi

TRAIN_SCRIPT="${SCRIPT_DIR}/monolingual_350m.sh"
MAX_SEQ_LEN=1024
OUTPUT_ROOT="${DATA_ROOT}/monolingual_training_data_tokenized"
INDEX_PATH="${SCRIPT_DIR}/configs/monolingual_bp_index.json"
MASTER_PORT=${MASTER_PORT:-29510}
MAX_EPOCHS=20

echo "[Config] datasets=${DATASETS[*]} tokenizer=$TOKENIZER vocabs=${VOCABS[*]}"
echo "[Config] Using training script: $TRAIN_SCRIPT"

# Per-language config: tokenizer dataset, byte premium, eval lines
declare -A TOK_DATASET_MAP=( [fineweb2_amh]=amh_ethi [fineweb2_tha]=tha_thai [fineweb2_urd]=urd_arab [fineweb2_vie]=vie_latn )
declare -A BYTE_PREMIUM_MAP=( [fineweb2_amh]=1.72 [fineweb2_tha]=2.74 [fineweb2_urd]=1.71 [fineweb2_vie]=1.35 )
declare -A EVAL_LINES_MAP=( [fineweb2_amh]=13760 [fineweb2_tha]=21920 [fineweb2_urd]=13680 [fineweb2_vie]=10800 )

# Verify raw text files exist (run download_fineweb2.sh first)
for dataset in "${DATASETS[@]}"; do
  if [[ -z "${BYTE_PREMIUM_MAP[$dataset]:-}" ]]; then
    echo "[ERROR] Unknown dataset: $dataset (expected one of ${!BYTE_PREMIUM_MAP[*]}, or 'all')" >&2
    exit 1
  fi
  raw_file="${DATA_ROOT}/monolingual_training_data/${dataset}.txt"
  if [[ ! -f "$raw_file" ]]; then
    echo "[ERROR] Raw text file not found: $raw_file" >&2
    echo "Run download_fineweb2.sh first." >&2
    exit 1
  fi
done

tok_prefix() {
  if [[ "$1" == "bpe_unscaled" ]]; then echo "bpe"; else echo "unigram"; fi
}

eng_ref_bytes_for() {
  local tokenizer_type=$1 vocab=$2
  local prefix; prefix=$(tok_prefix "$tokenizer_type")
  local eng_ref_file="${OUTPUT_ROOT}/${prefix}_eng_latn_${vocab}_300mb_unscaled/fineweb_eng_1.0_tokenized.txt"
  if [[ ! -f "$eng_ref_file" ]]; then
    echo "  [ERROR] English reference file not found: $eng_ref_file" >&2
    echo "  Run the English FineWeb pipeline first: bash launch_350m_fineweb.sh $tokenizer_type $vocab" >&2
    return 1
  fi
  stat -c%s "$eng_ref_file"
}

tokenize_and_split_fn() {
  local dataset=$1
  local tokenizer_type=$2
  local vocab=$3

  local tok_dataset="${TOK_DATASET_MAP[$dataset]}"
  local bp="${BYTE_PREMIUM_MAP[$dataset]}"
  local eval_lines="${EVAL_LINES_MAP[$dataset]}"

  local prefix; prefix=$(tok_prefix "$tokenizer_type")
  local tok_dir="${OUTPUT_ROOT}/${prefix}_${tok_dataset}_${vocab}_300mb_unscaled"
  local tok_file="${tok_dir}/${dataset}_${bp}_tokenized.txt"
  local full_file="${tok_dir}/${dataset}_${bp}_tokenized_full.txt"
  local eval_file="${tok_dir}/${dataset}_${bp}_eval_tokenized.txt"

  local eng_ref_bytes target_bytes
  eng_ref_bytes=$(eng_ref_bytes_for "$tokenizer_type" "$vocab") || return 1
  target_bytes=$(python3 -c "import math; print(math.floor($bp * $eng_ref_bytes))")
  echo "  [Config] eng_ref=$eng_ref_bytes bytes, BP=$bp, target_bytes=$target_bytes, eval_lines=$eval_lines"

  # Skip if the final train + eval splits already exist
  if [[ -f "$tok_file" && -f "$eval_file" ]]; then
    echo "  [Skip] Tokenized splits already exist: $tok_file"
  else
    # Step 1: Tokenize ALL raw text if not already done
    if [[ -f "$full_file" ]]; then
      echo "  [Step 1] Full tokenized file already exists: $full_file"
    else
      echo "  [Step 1] Tokenizing full dataset: dataset=$dataset, tok=$tokenizer_type, vocab=$vocab"
      python "${SCRIPT_DIR}/scripts/tokenize_and_pack.py" \
        --dataset "$dataset" --tokenizer_type "$tokenizer_type" --tokenizer_vocabulary "$vocab" \
        --split train --max_seq_len $MAX_SEQ_LEN --max_segments -1 \
        --prepend_cls True --include_sep True --shuffle False \
        --index_path "$INDEX_PATH" --output_root "$OUTPUT_ROOT"
      mv "$tok_file" "$full_file"
    fi

    # Step 2: Split by target bytes (with disjointness guarantee)
    echo "  [Step 2] Splitting: first ${target_bytes} bytes for train, last ${eval_lines} lines for eval"
    python "${SCRIPT_DIR}/scripts/split_tokenized.py" \
      --input "$full_file" \
      --train_output "$tok_file" \
      --eval_output "$eval_file" \
      --target_bytes "$target_bytes" \
      --eval_lines "$eval_lines"
  fi
}

for dataset in "${DATASETS[@]}"; do
  for vocab in "${VOCABS[@]}"; do
    echo "================================================================"
    echo "[Tokenize+Split] dataset=$dataset, tok=$TOKENIZER, vocab=$vocab"
    tokenize_and_split_fn "$dataset" "$TOKENIZER" "$vocab"

    # Data-scarce languages hold less than the byte target; repeat epochs to compensate.
    bp="${BYTE_PREMIUM_MAP[$dataset]}"
    tok_dataset="${TOK_DATASET_MAP[$dataset]}"
    prefix=$(tok_prefix "$TOKENIZER")
    tok_file="${OUTPUT_ROOT}/${prefix}_${tok_dataset}_${vocab}_300mb_unscaled/${dataset}_${bp}_tokenized.txt"

    eng_ref_bytes=$(eng_ref_bytes_for "$TOKENIZER" "$vocab")
    target_bytes=$(python3 -c "import math; print(math.floor($bp * $eng_ref_bytes))")
    actual_bytes=$(stat -c%s "$tok_file")

    if (( actual_bytes < target_bytes )); then
      NUM_EPOCHS=$(python3 -c "print(min($MAX_EPOCHS, $target_bytes // $actual_bytes))")
    else
      NUM_EPOCHS=1
    fi
    echo "  [Epochs] target=$target_bytes actual=$actual_bytes -> epochs=$NUM_EPOCHS"

    echo "[Train] dataset=$dataset, tok=$TOKENIZER, vocab=$vocab, epochs=$NUM_EPOCHS"
    bash "$TRAIN_SCRIPT" pre "$dataset" "$vocab" "$TOKENIZER" 6 "$MASTER_PORT" "$NUM_EPOCHS"
  done
done
