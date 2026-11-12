<div align="center">

# VISTA Model Suite | Training Pipeline

### No Single Tokenizer Feature Reliably Predicts<br>Downstream Language Model Performance

Antonio Lopardo &nbsp;·&nbsp; Luis Frentzen Salim &nbsp;·&nbsp; Shikhar Srivastava<br>
Kevin Jiang &nbsp;·&nbsp; Mridul Sharma &nbsp;·&nbsp; Catherine Arnett

*Findings of the Association for Computational Linguistics: EMNLP 2026*

Correspondence: [catherine@eleuther.ai](mailto:catherine@eleuther.ai)

<br>

<p>
🤗 <b>Models:</b>
<a href="https://huggingface.co/collections/shikhar-srivastava/vista-130m">VISTA-130M</a> &nbsp;·&nbsp;
<a href="https://huggingface.co/collections/shikhar-srivastava/vista-350m">VISTA-350M</a>
&nbsp;&nbsp;|&nbsp;&nbsp;
🤗 <b>Data:</b>
<a href="https://huggingface.co/datasets/catherinearnett/montok">Tokenizers</a> &nbsp;·&nbsp;
<a href="https://huggingface.co/datasets/goldfish-models/fish-food">Corpora</a>
</p>

</div>

<br>

This repository is the end-to-end training pipeline for the **VISTA model suite**, covering data
acquisition, byte-premium matching, tokenization and pretraining. It reproduces every checkpoint
released above.

VISTA models are LLaMA-style decoders trained from scratch across a grid of **languages ×
tokenization algorithms × vocabulary sizes**. Each language's training data is matched in *bytes*
rather than tokens, so that models differing only in their tokenizer stay comparable.

## VISTA Suite

| | 130M models | 350M models |
|---|---|---|
| Corpus | Monolingual, byte-premium matched | FineWeb (English), FineWeb-2 (others) |
| Languages | English, Thai, Urdu, Amharic, Vietnamese | English, Thai, Urdu, Amharic, Vietnamese |
| Algorithms | BPE, Unigram | BPE, Unigram |
| Vocabulary sizes | 8192, 16384, 32768, 49152, 65536, 81920, 98304, 114688, 262144 | 8192, 32768, 65536, 98304 |

### Byte-premium matching

Languages are not equally compressible into UTF-8, so an equal number of bytes is not an equal
amount of text. Each language is allotted `byte_premium ×` the English budget, so every model sees
a comparable quantity of content:

| Language | Code | Script | Byte premium | 130M train split |
|---|---|---|---|---|
| English | `eng_latn` | Latin | 1.00 | 1.00 GiB |
| Vietnamese | `vie_latn` | Latin | 1.35 | 1.35 GiB |
| Urdu | `urd_arab` | Arabic | 1.71 | 1.71 GiB |
| Amharic | `amh_ethi` | Ethiopic | 1.72 | 1.72 GiB |
| Thai | `tha_thai` | Thai | 2.74 | 2.74 GiB |

Premiums and split sizes live in `configs/monolingual_bp_index.json` and are applied by
`scripts/create_bp_splits.py`.

## Setup

```bash
conda create -n vista python=3.9 -y
conda activate vista
pip install -r exp_requirements.txt
```

All data paths resolve from a single `local.env` at the repo root, which is git-ignored so each
machine keeps its own:

```bash
cp local.env.example local.env
$EDITOR local.env          # set DATA_ROOT=/path/to/your/data
```

`DATA_ROOT` is the only variable. The pipeline reads and writes these subdirectories under it:

```
$DATA_ROOT/
├── monolingual-tokenizers/               tokenizers (downloaded)
│   ├── bpe_unscaled_tokenizers/
│   └── unigram_unscaled_tokenizers/
├── monolingual_training_data/            raw text, one document per line
├── monolingual_training_data_bp/         byte-premium train/eval splits
└── monolingual_training_data_tokenized/  packed token-id files consumed by the trainer
```

Everything inside the repo resolves relative to the checkout, so it works from any location.

## Data

**1. Tokenizers.** All 90 tokenizers (5 languages × 2 algorithms × 9 vocabulary sizes) are published
as a Hugging Face dataset:

```bash
bash download_tokenizers.sh
```

**2. Raw text.**

The 130M runs use per-language monolingual corpora from
[`goldfish-models/fish-food`](https://huggingface.co/datasets/goldfish-models/fish-food):

```bash
bash download_monolingual_data.sh          # all five languages
```

The 350M runs use FineWeb and FineWeb-2:

```bash
python scripts/download_fineweb.py                       # HuggingFaceFW/fineweb, sample-10BT
python scripts/download_fineweb2_multilingual.py --list-languages
python scripts/download_fineweb2_multilingual.py --languages eng_Latn,deu_Latn,rus_Cyrl,cmn_Hani --target-bytes 1000000000 --output data/fineweb2_multilingual.jsonl
```

The new script replaces the old single-purpose FineWeb-2 shell downloader. It streams remote Parquet
files directly from Hugging Face, lets you select a custom language mix and ratio, and writes either
JSONL or TSV output with temp-shard buffering and retry handling. The older workflow was a simple
language-by-language download; this version is a mixed-corpus builder for multilingual training data.

**3. Byte-premium splits** (130M path only). Carves each corpus into a `byte_premium × 1 GiB` train
split and a held-out eval split of `8000 × byte_premium` lines:

```bash
bash create_byte_premium_sets.sh
```

Run this only once all five corpora are present: it regenerates
`configs/monolingual_bp_index.json` from whichever files it finds on disk.

**4. Tokenization.** Text is tokenized once per (language, algorithm, vocabulary) and packed into a
token-ids-per-line format the trainer streams directly. The sweep scripts do this automatically; to
run it by hand:

```bash
python scripts/tokenize_and_pack.py \
  --dataset eng_latn --tokenizer_type bpe_unscaled --tokenizer_vocabulary 32768 \
  --split train --max_seq_len 1024 --max_segments -1 \
  --prepend_cls True --include_sep True --shuffle True
```

## Training

### 130M: monolingual, byte-premium matched

One script covers the grid, tokenizing each cell and then training it:

```bash
# One language, one algorithm, the full vocabulary sweep
bash launch_monolingual.sh eng_latn bpe_unscaled

# A single cell
bash launch_monolingual.sh tha_thai unigram_unscaled 32768
```

By default this uses the **Goldfish-matched** recipe (`monolingual_130m_gold.sh`), which is what the
released models were trained with. Set `goldfish=False` for the higher-throughput recipe
(`monolingual_130m.sh`):

| | Goldfish-matched (default) | Alternate |
|---|---|---|
| Learning rate | 1e-4 | 1e-3 |
| Batch size / device | 16 | 32 |
| Total batch size | 64 | 512 |
| Attention dropout | 0.1 | 0.0 |
| Adam eps | 1e-6 | 1e-8 |

Shared by both: 10 epochs, cosine schedule, 10% warmup, no weight decay, bfloat16, sequence length
1024, 4 GPUs via `torchrun`.

To train a single model directly:

```bash
bash monolingual_130m_gold.sh pre eng_latn 32768 bpe_unscaled 6 29510
#                            │   │        │     │             │ └─ master port
#                            │   │        │     │             └─── post_num
#                            │   │        │     └─────────────────  bpe_unscaled | unigram_unscaled
#                            │   │        └───────────────────────  vocabulary size
#                            │   └────────────────────────────────  language code
#                            └────────────────────────────────────  norm type (use "pre")
```

### 350M: FineWeb and FineWeb-2

English must be built first: its tokenized output is the byte-size reference every other language is
matched against.

```bash
bash launch_350m_fineweb.sh  bpe_unscaled           # English
bash launch_350m_fineweb2.sh all bpe_unscaled       # Amharic, Thai, Urdu, Vietnamese
```

Both accept an explicit vocabulary list, and `launch_350m_fineweb2.sh` accepts a single language:

```bash
bash launch_350m_fineweb2.sh fineweb2_tha unigram_unscaled 32768 65536
```

English is trained on its first 7B tokens. For the data-scarce languages (Amharic, Urdu) the full
corpus is smaller than the byte target, so epochs are repeated to compensate:
`epochs = min(20, target_bytes / actual_bytes)`, computed per cell.

The 350M recipe uses lr 1e-3, batch size 16 per device, total batch 512, gradient clipping 1.0.

### GPUs

All training scripts use 4 GPUs by default. Override with `NUM_GPUS`:

```bash
NUM_GPUS=8 bash launch_350m_fineweb.sh bpe_unscaled
```

### Logging

Training logs to Weights & Biases under the project name in the `WANDB_PROJECT` environment
variable, defaulting to `vista`. Set it to your own project before launching:

```bash
export WANDB_PROJECT=my-project
```

(It is read in `torchrun_main.py`, at the `wandb.init` call.) Checkpoints are written locally to
`./{run_name}/`. Nothing is uploaded to the Hugging Face Hub unless you pass `--hf_repo_name` and
`--hf_push_final` / `--hf_push_checkpoints` to `torchrun_main.py` yourself.

### Architecture

| | 130M | 350M |
|---|---|---|
| Layers | 12 | 24 |
| Hidden size | 768 | 1024 |
| Attention heads | 12 | 16 |
| Intermediate size | 2048 | 2736 |
| Sequence length | 1024 | 1024 |

Both are pre-norm LLaMA with RMSNorm, SwiGLU (SiLU) and rotary position embeddings. The `vocab_size`
in the config files is a placeholder; the trainer sets it from the tokenizer in use.

## Repository layout

```
├── torchrun_main.py               Training entry point
├── configs/                       Model architectures and the byte-premium index
├── peft_pretraining/              LLaMA implementation, dataloader, training utilities
├── utils/                         Path resolution and training-time instrumentation
├── scripts/                       Download, byte-premium splits, tokenization
├── evals/                         Downstream evaluation suite (see evals/README.md)
│
├── download_tokenizers.sh         Fetch the 90 tokenizers
├── download_monolingual_data.sh   Fetch the monolingual corpora
├── download_fineweb2.sh           Backward-compatible wrapper to the multilingual FineWeb-2 downloader
├── scripts/
│   ├── download_fineweb2_multilingual.py  Mixed-language FineWeb-2 corpus builder
│   ├── split_tokenized.py
│   └── tokenize_and_pack.py
├── create_byte_premium_sets.sh    Build byte-premium train/eval splits
│
├── launch_monolingual.sh          130M sweep
├── launch_350m_fineweb.sh         350M English sweep
├── launch_350m_fineweb2.sh        350M multilingual sweep
│
├── monolingual_130m.sh            130M trainer
├── monolingual_130m_gold.sh       130M trainer, Goldfish-matched (sweep default)
└── monolingual_350m.sh            350M trainer
```

## Acknowledgments

Built on [LayerNorm-Scaling](https://github.com/lmsdss/LayerNorm-Scaling) for the training
infrastructure and
[word-acquisition-language-models](https://github.com/tylerachang/word-acquisition-language-models)
for the tokenization and packing pipeline. The monolingual corpora, the 130M training recipe and the
FLORES evaluation follow [Goldfish](https://github.com/tylerachang/goldfish). We thank the authors of
all three.
